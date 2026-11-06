import numpy as np
import time
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline to be safe before 90s kill

    # 1. Parse Input
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            vals = list(distribution_spec.values())
            probs = None
            for v in vals:
                if isinstance(v, (list, np.ndarray)):
                    probs = np.array(v, dtype=float)
                    break
            if probs is None:
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
    grid_values = []
    for v1 in range(4):
        for v2 in range(4):
            grid_values.append((v1, v2))

    n_types = 16
    # Variables: x1 (16), x2 (16), u (16) -> 48 vars
    n_vars = 48

    # 3. Formulate LP
    # Maximize Revenue = sum(p_i * probs_i)
    # p_i = v1_i * x1_i + v2_i * x2_i - u_i
    # Minimize -Revenue = -sum(probs_i * (v1_i * x1_i + v2_i * x2_i - u_i))
    # = sum(-probs_i * v1_i * x1_i - probs_i * v2_i * x2_i + probs_i * u_i)

    c = np.zeros(n_vars)
    for i in range(16):
        v1, v2 = grid_values[i]
        c[i] = -probs[i] * v1
        c[16 + i] = -probs[i] * v2
        c[32 + i] = probs[i]

    A_rows = []
    b_rows = []

    # IC Constraints:
    # u_i - u_j >= (v1_i - v1_j)*x1_j + (v2_i - v2_j)*x2_j
    # => u_j - u_i - (v1_i - v1_j)*x1_j - (v2_i - v2_j)*x2_j <= 0
    # Wait, standard IC is:
    # v_i . x_i - p_i >= v_i . x_j - p_j
    # (v_i . x_i - u_i) >= (v_i . x_j - u_j)  [since p = v.x - u]
    # v_i . x_i - v_i . x_j >= u_i - u_j
    # (v1_i - v1_j)x1_i + (v2_i - v2_j)x2_i >= u_i - u_j
    # This is the "local" or "global" IC depending on which x is used.
    # For finite support, global IC is:
    # u_i - u_j >= (v_i - v_j) . x_j  is WRONG.
    # Correct IC:
    # v_i . x_i - p_i >= v_i . x_j - p_j
    # Let u_i = v_i . x_i - p_i (utility)
    # u_i >= v_i . x_j - p_j
    # p_j = v_j . x_j - u_j
    # u_i >= v_i . x_j - (v_j . x_j - u_j) = (v_i - v_j) . x_j + u_j
    # u_i - u_j >= (v_i - v_j) . x_j
    # So: u_i - u_j - (v1_i - v1_j)*x1_j - (v2_i - v2_j)*x2_j >= 0
    # => -(u_i - u_j) + (v1_i - v1_j)*x1_j + (v2_i - v2_j)*x2_j <= 0
    # => -u_i + u_j + (v1_i - v1_j)*x1_j + (v2_i - v2_j)*x2_j <= 0

    for i in range(16):
        v1_i, v2_i = grid_values[i]
        for j in range(16):
            if i == j:
                continue
            v1_j, v2_j = grid_values[j]
            row = np.zeros(n_vars)
            row[32 + i] = -1
            row[32 + j] = 1
            row[j] = v1_i - v1_j
            row[16 + j] = v2_i - v2_j
            A_rows.append(row)
            b_rows.append(0)

    # Payment Non-negativity: p_i >= 0
    # p_i = v1_i * x1_i + v2_i * x2_i - u_i >= 0
    # => u_i - v1_i * x1_i - v2_i * x2_i <= 0
    for i in range(16):
        v1, v2 = grid_values[i]
        row = np.zeros(n_vars)
        row[32 + i] = 1
        row[i] = -v1
        row[16 + i] = -v2
        A_rows.append(row)
        b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds:
    # x in [0, 1]
    # u in [0, inf)
    bounds = [(0, 1)] * 32 + [(0, None)] * 16

    def extract_solution(x):
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(16):
            allocations[i, 0] = x[i]
            allocations[i, 1] = x[16 + i]
            u_i = x[32 + i]
            v1, v2 = grid_values[i]
            payments[i] = v1 * allocations[i, 0] + v2 * allocations[i, 1] - u_i
        
        allocations = np.clip(allocations, 0, 1)
        payments = np.clip(payments, 0, None)
        return allocations, payments

    def check_feasibility(allocations, payments, probs):
        # Check IR
        for i in range(16):
            v1, v2 = grid_values[i]
            utility = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
            if utility < -1e-8:
                return False
        # Check IC
        for i in range(16):
            v1_i, v2_i = grid_values[i]
            u_i = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
            for j in range(16):
                if i == j:
                    continue
                u_j_as_if_i = v1_i * allocations[j, 0] + v2_i * allocations[j, 1] - payments[j]
                if u_i < u_j_as_if_i - 1e-8:
                    return False
        # Check bounds
        if np.any(allocations < -1e-12) or np.any(allocations > 1 + 1e-12):
            return False
        if np.any(payments < -1e-12):
            return False
        return True

    # 4. Solve
    best_result = None
    best_rev = -1

    # Try with tight tolerances first
    tolerances = [1e-10, 1e-9, 1e-8]

    for tol in tolerances:
        if time.time() > deadline:
            break
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                          options={'presolve': True,
                                   'dual_feasibility_tolerance': tol,
                                   'primal_feasibility_tolerance': tol})
            if res.success:
                x = res.x
                allocations, payments = extract_solution(x)
                if check_feasibility(allocations, payments, probs):
                    rev = np.sum(payments * probs)
                    if rev > best_rev:
                        best_rev = rev
                        best_result = {"allocations": allocations, "payments": payments}
                # If feasible and tight tolerance, we can break
                if tol == 1e-10:
                    break
        except Exception:
            pass

    if best_result is None:
        # Fallback: Try a simpler LP without presolve or with different method?
        # Or just return zero mechanism
        return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
        
    return best_result
