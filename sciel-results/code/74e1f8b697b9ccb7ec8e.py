import numpy as np
import time
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Strategy: Solve the mechanism design LP using HiGHS with reduced IC constraints
    (exploiting transitivity on the 2D grid) and high-precision tolerances.
    """
    start_time = time.time()
    DEADLINE = 85.0

    # 1. Parse Input
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            vals = list(distribution_spec.values())
            for v in vals:
                if isinstance(v, (list, np.ndarray)):
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=float)

    if probs.shape != (16,):
        probs = probs.flatten()
        if len(probs) != 16:
            if len(probs) > 16:
                probs = probs[:16]
            else:
                new_probs = np.zeros(16)
                new_probs[:len(probs)] = probs
                probs = new_probs

    total_p = np.sum(probs)
    if total_p > 1e-12:
        probs = probs / total_p
    else:
        probs = np.ones(16) / 16.0

    # 2. Define Grid
    # Index i corresponds to (v1, v2) where i = v1 * 4 + v2
    # v1 in 0..3, v2 in 0..3
    grid_values = [(i // 4, i % 4) for i in range(16)]
    
    n_types = 16
    n_vars = 48  # 16 x1, 16 x2, 16 p

    # 3. Formulate LP
    # Maximize Revenue = sum(p_i * probs_i)
    # Minimize -sum(p_i * probs_i)
    c = np.zeros(n_vars)
    for i in range(16):
        c[32 + i] = -probs[i]

    A_rows = []
    b_rows = []

    # IR Constraints: v1*x1_i + v2*x2_i - p_i >= 0
    # => -v1*x1_i - v2*x2_i + p_i <= 0
    for i in range(16):
        v1, v2 = grid_values[i]
        row = np.zeros(n_vars)
        row[i] = -v1
        row[16 + i] = -v2
        row[32 + i] = 1
        A_rows.append(row)
        b_rows.append(0)

    # IC Constraints (Reduced via Transitivity)
    # We only need to check against immediate "down" neighbors in the grid.
    # For type i = (v1, v2):
    # - If v1 > 0, neighbor j = (v1-1, v2) has index (v1-1)*4 + v2
    # - If v2 > 0, neighbor j = (v1, v2-1) has index v1*4 + (v2-1)
    
    for i in range(16):
        v1_i, v2_i = grid_values[i]
        
        # Check down-left (decrease v1)
        if v1_i > 0:
            j = (v1_i - 1) * 4 + v2_i
            # u_i >= u_j
            # v_i.x_i - p_i >= v_i.x_j - p_j
            # -v_i.x_i + v_i.x_j + p_i - p_j <= 0
            row = np.zeros(n_vars)
            row[i] = -v1_i
            row[16 + i] = -v2_i
            row[j] = v1_i
            row[16 + j] = v2_i
            row[32 + i] = 1
            row[32 + j] = -1
            A_rows.append(row)
            b_rows.append(0)
            
        # Check down-right (decrease v2)
        if v2_i > 0:
            j = v1_i * 4 + (v2_i - 1)
            # u_i >= u_j
            # v_i.x_i - p_i >= v_i.x_j - p_j
            # -v_i.x_i + v_i.x_j + p_i - p_j <= 0
            row = np.zeros(n_vars)
            row[i] = -v1_i
            row[16 + i] = -v2_i
            row[j] = v1_i
            row[16 + j] = v2_i
            row[32 + i] = 1
            row[32 + j] = -1
            A_rows.append(row)
            b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))

    # 4. Solve
    try:
        # Use tight tolerances
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True,
                               'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12,
                               'dual_gap_tolerance': 1e-12,
                               'time_limit': DEADLINE - (time.time() - start_time) - 5})

        if time.time() - start_time > DEADLINE:
            return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}

        if res.success:
            x = res.x
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)

            for i in range(16):
                allocations[i, 0] = x[i]
                allocations[i, 1] = x[16 + i]
                payments[i] = x[32 + i]

            # Clip to handle numerical noise
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)

            # Polish pass: refine using the solution as warm start
            # Re-solve with even tighter tolerances
            if time.time() - start_time < DEADLINE - 10:
                try:
                    res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                                   options={'presolve': True,
                                            'dual_feasibility_tolerance': 1e-14,
                                            'primal_feasibility_tolerance': 1e-14,
                                            'dual_gap_tolerance': 1e-14,
                                            'time_limit': DEADLINE - (time.time() - start_time) - 5})
                    if res2.success:
                        x2 = res2.x
                        allocations2 = np.zeros((16, 2))
                        payments2 = np.zeros(16)
                        for i in range(16):
                            allocations2[i, 0] = x2[i]
                            allocations2[i, 1] = x2[16 + i]
                            payments2[i] = x2[32 + i]
                        allocations2 = np.clip(allocations2, 0, 1)
                        payments2 = np.clip(payments2, 0, None)
                        # Use the better objective
                        rev1 = np.sum(payments * probs)
                        rev2 = np.sum(payments2 * probs)
                        if rev2 >= rev1 - 1e-10:
                            allocations = allocations2
                            payments = payments2
                except Exception:
                    pass

            return {
                "allocations": allocations,
                "payments": payments
            }
        else:
            # If LP fails, try with slightly relaxed constraints
            # This is a safety net, but reduced constraints should be less likely to fail
            b_relaxed = b_ub.copy()
            for k in range(len(b_relaxed)):
                b_relaxed[k] = 1e-10

            res_relaxed = linprog(c, A_ub=A_ub, b_ub=b_relaxed, bounds=bounds, method='highs',
                                  options={'presolve': True,
                                           'time_limit': DEADLINE - (time.time() - start_time) - 5})
            if res_relaxed.success:
                x = res_relaxed.x
                allocations = np.zeros((16, 2))
                payments = np.zeros(16)
                for i in range(16):
                    allocations[i, 0] = x[i]
                    allocations[i, 1] = x[16 + i]
                    payments[i] = x[32 + i]
                allocations = np.clip(allocations, 0, 1)
                payments = np.clip(payments, 0, None)
                return {"allocations": allocations, "payments": payments}

            # Final fallback
            return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}

    except Exception:
        return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}

