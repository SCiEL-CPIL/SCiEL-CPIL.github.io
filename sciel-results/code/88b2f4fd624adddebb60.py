import numpy as np
from scipy.optimize import linprog
import time
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 58.0  # Safety margin before 60s

    # Construct the full support grid: values in {0,1,2,3} for two goods
    values = [0, 1, 2, 3]
    support_grid = [(v1, v2) for v1 in values for v2 in values]
    n_states = len(support_grid)
    state_to_idx = {s: i for i, s in enumerate(support_grid)}

    # Extract probabilities robustly
    probs = np.zeros(n_states)
    if isinstance(distribution_spec, dict):
        for s, p in distribution_spec.items():
            if s in state_to_idx:
                probs[state_to_idx[s]] = float(p)
    elif isinstance(distribution_spec, (np.ndarray, list)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (4, 4):
            for i in range(4):
                for j in range(4):
                    idx = state_to_idx[(i, j)]
                    probs[idx] = arr[i, j]
        else:
            flat = arr.flatten()
            if len(flat) == n_states:
                probs = flat.copy()
            else:
                probs = np.ones(n_states) / n_states
    else:
        probs = np.ones(n_states) / n_states

    # Normalize
    total_prob = probs.sum()
    if total_prob > 1e-12:
        probs = probs / total_prob
    else:
        probs = np.ones(n_states) / n_states

    # --- Attempt 1: Solve the full LP for the optimal mechanism ---
    
    # LP variables: x1[0..15], x2[0..15], p[0..15]
    n_vars = 3 * n_states
    c = np.zeros(n_vars)
    for i in range(n_states):
        c[2 * n_states + i] = -probs[i]

    # Build constraint matrix
    A_ub_rows = []
    b_ub_vals = []

    # IC constraints: for all pairs (i, j), p_i - p_j <= v_i·x_i - v_i·x_j
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        for j in range(n_states):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[2 * n_states + i] = 1.0
            row[2 * n_states + j] = -1.0
            row[i] = -v1_i
            row[n_states + i] = -v2_i
            row[j] = v1_i
            row[n_states + j] = v2_i
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    # IR constraints: p_i <= v1_i*x1_i + v2_i*x2_i
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        row = np.zeros(n_vars)
        row[2 * n_states + i] = 1.0
        row[i] = -v1_i
        row[n_states + i] = -v2_i
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    if len(A_ub_rows) > 0:
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
    else:
        A_ub = np.empty((0, n_vars))
        b_ub = np.empty(0)

    # Bounds
    bounds = [(0, 1)] * n_states + [(0, 1)] * n_states + [(0, None)] * n_states

    x_opt = None
    if time.time() - start_time < deadline - 5:
        try:
            # Use highs with tighter tolerances for better numerical stability
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12,
                                  'dual_edge_tolerances': [1e-12, 1e-12],
                                  'primal_edge_tolerances': [1e-12, 1e-12]})
            if res.status == 0:
                x_opt = res.x
            else:
                # Fallback to interior-point if highs fails
                res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point',
                              options={'maxiter': 1000, 'disp': False})
                if res2.status == 0:
                    x_opt = res2.x
        except Exception:
            x_opt = None

    if x_opt is not None:
        x1 = x_opt[:n_states]
        x2 = x_opt[n_states:2 * n_states]
        p = x_opt[2 * n_states:]

        allocations = np.column_stack((x1, x2))
        payments = p

        # Clean up numerical noise
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)

        # Verify IC (regret check)
        max_regret = 0.0
        for i in range(n_states):
            v1_i, v2_i = support_grid[i]
            utility_truthful = v1_i * x1[i] + v2_i * x2[i] - p[i]
            for j in range(n_states):
                utility_deviate = v1_i * x1[j] + v2_i * x2[j] - p[j]
                regret = utility_deviate - utility_truthful
                if regret > max_regret:
                    max_regret = regret

        if max_regret <= 1e-8:
            return {"allocations": allocations, "payments": payments}
        else:
            # If IC is violated by a small amount, try to fix by slightly increasing payments
            # This is safe because IR has slack for most states
            violation = max_regret
            if violation < 1e-4:
                # Try adding a small epsilon to all payments to create margin
                for eps in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
                    p_adjusted = p + eps
                    # Check IR still holds
                    ir_ok = True
                    for i in range(n_states):
                        v1_i, v2_i = support_grid[i]
                        if p_adjusted[i] > v1_i * x1[i] + v2_i * x2[i] + 1e-10:
                            ir_ok = False
                            break
                    if ir_ok:
                        # Re-check IC
                        max_regret_adj = 0.0
                        for i in range(n_states):
                            v1_i, v2_i = support_grid[i]
                            utility_truthful = v1_i * x1[i] + v2_i * x2[i] - p_adjusted[i]
                            for j in range(n_states):
                                utility_deviate = v1_i * x1[j] + v2_i * x2[j] - p_adjusted[j]
                                regret = utility_deviate - utility_truthful
                                if regret > max_regret_adj:
                                    max_regret_adj = regret
                        if max_regret_adj <= 1e-8:
                            return {"allocations": allocations, "payments": p_adjusted}
            
            # Try a secondary LP with slightly tightened IC constraints to get a more robust solution
            if time.time() - start_time < deadline - 5:
                try:
                    A_ub_tight = A_ub.copy()
                    b_ub_tight = b_ub.copy()
                    # Tighten IC constraints (first n_states*(n_states-1) rows)
                    n_ic = n_states * (n_states - 1)
                    b_ub_tight[:n_ic] -= 1e-6
                    res_tight = linprog(c, A_ub=A_ub_tight, b_ub=b_ub_tight, bounds=bounds, 
                                       method='highs',
                                       options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                                'primal_feasibility_tolerance': 1e-12})
                    if res_tight.status == 0:
                        x_opt_tight = res_tight.x
                        x1_t = x_opt_tight[:n_states]
                        x2_t = x_opt_tight[n_states:2 * n_states]
                        p_t = x_opt_tight[2 * n_states:]
                        alloc_t = np.column_stack((x1_t, x2_t))
                        alloc_t = np.clip(alloc_t, 0.0, 1.0)
                        p_t = np.maximum(p_t, 0.0)
                        
                        max_regret_t = 0.0
                        for i in range(n_states):
                            v1_i, v2_i = support_grid[i]
                            utility_truthful = v1_i * x1_t[i] + v2_i * x2_t[i] - p_t[i]
                            for j in range(n_states):
                                utility_deviate = v1_i * x1_t[j] + v2_i * x2_t[j] - p_t[j]
                                regret = utility_deviate - utility_truthful
                                if regret > max_regret_t:
                                    max_regret_t = regret
                        
                        if max_regret_t <= 1e-8:
                            return {"allocations": alloc_t, "payments": p_t}
                except Exception:
                    pass

    # --- Attempt 2: Robust Fallback (Separable Threshold Mechanism) ---
    # This mechanism is guaranteed IC and IR.
    # It assumes the allocation of good 1 depends only on v1, and good 2 only on v2.
    # x_g(v) = 1 if v_g >= T_g, else 0.
    # p_g(v) = T_g * x_g(v).
    
    best_rev = -1
    best_alloc = np.zeros((n_states, 2))
    best_pay = np.zeros(n_states)
    
    # Thresholds can be 0, 1, 2, 3, or 4 (never allocate)
    # If threshold is 0, we always allocate. Payment is 0.
    # If threshold is 4, we never allocate. Payment is 0.
    possible_thresholds = [0, 1, 2, 3, 4]
    
    for t1 in possible_thresholds:
        for t2 in possible_thresholds:
            alloc = np.zeros((n_states, 2))
            pay = np.zeros(n_states)
            
            for i in range(n_states):
                v1, v2 = support_grid[i]
                a1 = 1.0 if v1 >= t1 else 0.0
                a2 = 1.0 if v2 >= t2 else 0.0
                alloc[i, 0] = a1
                alloc[i, 1] = a2
                # Payment is the threshold value if allocated
                pay[i] = a1 * t1 + a2 * t2
                
            # Calculate expected revenue
            exp_rev = np.sum(probs * pay)
            
            if exp_rev > best_rev:
                best_rev = exp_rev
                best_alloc = alloc
                best_pay = pay

    # Verify the fallback mechanism just in case (it should be perfect)
    max_regret = 0.0
    for i in range(n_states):
        v1_i, v2_i = support_grid[i]
        utility_truthful = v1_i * best_alloc[i, 0] + v2_i * best_alloc[i, 1] - best_pay[i]
        for j in range(n_states):
            utility_deviate = v1_i * best_alloc[j, 0] + v2_i * best_alloc[j, 1] - best_pay[j]
            regret = utility_deviate - utility_truthful
            if regret > max_regret:
                max_regret = regret
                
    if max_regret <= 1e-8:
        return {"allocations": best_alloc, "payments": best_pay}
    else:
        # Last resort: Zero mechanism
        return {"allocations": np.zeros((n_states, 2)), "payments": np.zeros(n_states)}
