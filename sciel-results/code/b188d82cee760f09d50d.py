import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0  # safety margin before 90s kill

    # Parse distribution_spec to get probabilities over 16 types
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.zeros(16)
            for k, v in distribution_spec.items():
                if isinstance(k, tuple) and len(k) == 2:
                    v1, v2 = k
                    idx = v1 * 4 + v2
                    if 0 <= idx < 16:
                        probs[idx] = v
                elif isinstance(k, int) and 0 <= k < 16:
                    probs[k] = v
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) != 16:
            if len(probs) > 16:
                probs = probs[:16]
            else:
                tmp = np.zeros(16)
                tmp[:len(probs)] = probs
                probs = tmp
    else:
        probs = np.ones(16) / 16.0

    # Normalize
    probs = np.clip(probs, 0, None)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(16) / 16.0

    n_types = 16
    values = np.zeros((n_types, 2))
    for i in range(n_types):
        values[i, 0] = i // 4
        values[i, 1] = i % 4

    # LP formulation:
    # Variables: x[i,j] for i in 0..15, j in 0..1 (allocation probabilities, 0<=x<=1)
    #            p[i] for i in 0..15 (payments, p >= 0)
    # Total: 32 + 16 = 48 variables

    n_vars = 48
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[32 + i] = -probs[i]

    # Build constraint matrix
    A_ub_list = []
    b_ub_list = []

    # IR constraints: p_i - sum_j v_ij x_ij <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[32 + i] = 1.0
        for j in range(2):
            row[i * 2 + j] = -values[i, j]
        A_ub_list.append(row)
        b_ub_list.append(0.0)

    # IC constraints: for each pair (i, j), i != j
    # p_i - p_j - sum_k v_ik x_ik + sum_k v_ik x_jk <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[32 + i] = 1.0
            row[32 + j] = -1.0
            for k in range(2):
                row[i * 2 + k] = -values[i, k]
                row[j * 2 + k] = values[i, k]
            A_ub_list.append(row)
            b_ub_list.append(0.0)

    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[32:] = 6.0  # max possible payment is sum of max values = 3+3=6

    def solve_lp(c_vec, A_ub_mat, b_ub_vec, bounds_list, time_limit):
        try:
            res = linprog(c_vec, A_ub=A_ub_mat, b_ub=b_ub_vec, bounds=bounds_list,
                         method='highs',
                         options={'presolve': True,
                                  'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12,
                                  'dual_staleness_threshold': 1e-12,
                                  'primal_staleness_threshold': 1e-12,
                                  'time_limit': time_limit})
            return res
        except Exception:
            return None

    best_allocations = None
    best_payments = None
    best_revenue = -np.inf

    # First pass: solve with standard tolerances
    remaining_time = deadline - time.time() - 5
    if remaining_time > 10:
        res = solve_lp(c, A_ub, b_ub, list(zip(lb, ub)), remaining_time)
        if res is not None and res.status == 0 and res.x is not None:
            x_sol = res.x[:32]
            p_sol = res.x[32:]
            x_sol = np.clip(x_sol, 0, 1)
            p_sol = np.maximum(p_sol, 0)
            allocations = x_sol.reshape(16, 2)
            payments = p_sol

            # Verify constraints
            valid = True
            for i in range(n_types):
                if payments[i] > np.sum(values[i] * allocations[i]) + 1e-8:
                    valid = False
                    break
            if valid:
                for i in range(n_types):
                    for j in range(n_types):
                        if i == j:
                            continue
                        lhs = payments[i] - payments[j]
                        rhs = np.sum(values[i] * (allocations[i] - allocations[j]))
                        if lhs > rhs + 1e-8:
                            valid = False
                            break
                    if not valid:
                        break

            if valid:
                revenue = np.sum(probs * payments)
                best_allocations = allocations.copy()
                best_payments = payments.copy()
                best_revenue = revenue

    # Polish pass: if we have a feasible solution, try to improve with tighter tolerances
    if best_allocations is not None and time.time() < deadline - 10:
        # Create a tighter problem around the current solution
        # Use the current solution as a warm start by solving again with very tight tolerances
        remaining_time = deadline - time.time() - 3
        if remaining_time > 5:
            res2 = solve_lp(c, A_ub, b_ub, list(zip(lb, ub)), remaining_time)
            if res2 is not None and res2.status == 0 and res2.x is not None:
                x_sol2 = res2.x[:32]
                p_sol2 = res2.x[32:]
                x_sol2 = np.clip(x_sol2, 0, 1)
                p_sol2 = np.maximum(p_sol2, 0)
                allocations2 = x_sol2.reshape(16, 2)
                payments2 = p_sol2

                valid2 = True
                for i in range(n_types):
                    if payments2[i] > np.sum(values[i] * allocations2[i]) + 1e-10:
                        valid2 = False
                        break
                if valid2:
                    for i in range(n_types):
                        for j in range(n_types):
                            if i == j:
                                continue
                            lhs = payments2[i] - payments2[j]
                            rhs = np.sum(values[i] * (allocations2[i] - allocations2[j]))
                            if lhs > rhs + 1e-10:
                                valid2 = False
                                break
                        if not valid2:
                            break

                if valid2:
                    revenue2 = np.sum(probs * payments2)
                    if revenue2 > best_revenue:
                        best_allocations = allocations2.copy()
                        best_payments = payments2.copy()
                        best_revenue = revenue2

    # If LP failed or we're running out of time, use a simple valid fallback
    if best_allocations is None or time.time() > deadline - 2:
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)

    return {"allocations": best_allocations, "payments": best_payments}
