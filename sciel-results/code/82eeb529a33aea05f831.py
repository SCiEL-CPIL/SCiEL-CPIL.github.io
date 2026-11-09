import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 58.0  # safety margin before 60s

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

    # Precompute value vectors
    v1 = np.array([s[0] for s in support_grid], dtype=float)
    v2 = np.array([s[1] for s in support_grid], dtype=float)

    # LP variables: x1[0..15], x2[0..15], p[0..15]
    n_vars = 3 * n_states
    c = np.zeros(n_vars)
    c[2 * n_states:] = -probs

    # Build constraint matrix efficiently using vectorized operations
    # IC constraints: p_i - p_j <= v_i·x_i - v_i·x_j for all i != j
    # IR constraints: p_i <= v1_i*x1_i + v2_i*x2_i

    # Number of IC constraints: n_states * (n_states - 1)
    n_ic = n_states * (n_states - 1)
    n_ir = n_states
    n_constraints = n_ic + n_ir

    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)

    # Fill IC constraints
    idx = 0
    for i in range(n_states):
        vi1, vi2 = v1[i], v2[i]
        for j in range(n_states):
            if i == j:
                continue
            A_ub[idx, 2 * n_states + i] = 1.0
            A_ub[idx, 2 * n_states + j] = -1.0
            A_ub[idx, i] = -vi1
            A_ub[idx, n_states + i] = -vi2
            A_ub[idx, j] = vi1
            A_ub[idx, n_states + j] = vi2
            idx += 1

    # Fill IR constraints
    for i in range(n_states):
        row_idx = n_ic + i
        A_ub[row_idx, 2 * n_states + i] = 1.0
        A_ub[row_idx, i] = -v1[i]
        A_ub[row_idx, n_states + i] = -v2[i]

    # Bounds
    bounds = [(0, 1)] * n_states + [(0, 1)] * n_states + [(0, None)] * n_states

    # Check time before solving
    if time.time() - start_time > deadline - 10:
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}

    # Solve LP with high precision settings
    x_opt = None
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'primal_feasibility_tolerance': 1e-12,
                             'dual_feasibility_tolerance': 1e-12})
        if res.status == 0:
            x_opt = res.x
        else:
            # Try interior-point as fallback
            res2 = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='interior-point')
            if res2.status == 0:
                x_opt = res2.x
    except Exception:
        x_opt = None

    if x_opt is None:
        allocations = np.zeros((n_states, 2))
        payments = np.zeros(n_states)
        return {"allocations": allocations, "payments": payments}

    x1 = x_opt[:n_states]
    x2 = x_opt[n_states:2 * n_states]
    p = x_opt[2 * n_states:]

    allocations = np.column_stack((x1, x2))
    payments = p

    # Clean up numerical noise
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)

    # Verify IC (regret check) with vectorized computation
    # utility_truthful[i] = v1[i]*x1[i] + v2[i]*x2[i] - p[i]
    # utility_deviate[i,j] = v1[i]*x1[j] + v2[i]*x2[j] - p[j]
    # regret[i,j] = utility_deviate[i,j] - utility_truthful[i]

    # Vectorized regret computation
    val_truthful = v1 * x1 + v2 * x2 - p  # shape (n_states,)
    val_deviate = v1[:, None] * x1[None, :] + v2[:, None] * x2[None, :] - p[None, :]  # shape (n_states, n_states)
    regret = val_deviate - val_truthful[:, None]
    max_regret = regret.max()

    # If regret exceeds tolerance, attempt to fix payments for the given allocation
    if max_regret > 1e-8:
        # Re-optimize payments for fixed allocations
        n_p = n_states
        c_p = -probs.copy()

        # Build payment LP constraints
        # IC: p_i - p_j <= val_i - (v1_i*x1_j + v2_i*x2_j) for all i != j
        # IR: p_i <= val_i

        n_constraints_p = n_ic + n_ir
        A_ub_p = np.zeros((n_constraints_p, n_p))
        b_ub_p = np.zeros(n_constraints_p)

        idx = 0
        for i in range(n_states):
            vi1, vi2 = v1[i], v2[i]
            val_i = vi1 * x1[i] + vi2 * x2[i]
            for j in range(n_states):
                if i == j:
                    continue
                val_j_at_i = vi1 * x1[j] + vi2 * x2[j]
                A_ub_p[idx, i] = 1.0
                A_ub_p[idx, j] = -1.0
                b_ub_p[idx] = val_i - val_j_at_i
                idx += 1

        for i in range(n_states):
            row_idx = n_ic + i
            A_ub_p[row_idx, i] = 1.0
            b_ub_p[row_idx] = v1[i] * x1[i] + v2[i] * x2[i]

        bounds_p = [(0, None)] * n_p

        try:
            res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs',
                           options={'presolve': True, 'primal_feasibility_tolerance': 1e-12,
                                   'dual_feasibility_tolerance': 1e-12})
            if res_p.status == 0:
                payments = res_p.x
                payments = np.maximum(payments, 0.0)

                # Re-verify with vectorized computation
                val_truthful_new = v1 * x1 + v2 * x2 - payments
                val_deviate_new = v1[:, None] * x1[None, :] + v2[:, None] * x2[None, :] - payments[None, :]
                regret_new = val_deviate_new - val_truthful_new[:, None]
                max_regret_new = regret_new.max()

                if max_regret_new <= 1e-8:
                    pass
                else:
                    # Still failed, try a second polish with even tighter tolerances
                    if time.time() - start_time < deadline - 5:
                        try:
                            res_p2 = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p,
                                            method='interior-point')
                            if res_p2.status == 0:
                                payments2 = np.maximum(res_p2.x, 0.0)
                                val_t2 = v1 * x1 + v2 * x2 - payments2
                                val_d2 = v1[:, None] * x1[None, :] + v2[:, None] * x2[None, :] - payments2[None, :]
                                reg2 = val_d2 - val_t2[:, None]
                                if reg2.max() <= 1e-8:
                                    payments = payments2
                        except Exception:
                            pass

                if regret_new.max() > 1e-8:
                    # Final fallback: use the original LP solution if it's close enough
                    # or return zero
                    if max_regret <= 1e-6:
                        # Keep original payments
                        pass
                    else:
                        allocations = np.zeros((n_states, 2))
                        payments = np.zeros(n_states)
            else:
                # LP for payments failed, try interior point
                try:
                    res_p2 = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p,
                                    method='interior-point')
                    if res_p2.status == 0:
                        payments = np.maximum(res_p2.x, 0.0)
                        val_t = v1 * x1 + v2 * x2 - payments
                        val_d = v1[:, None] * x1[None, :] + v2[:, None] * x2[None, :] - payments[None, :]
                        if (val_d - val_t[:, None]).max() > 1e-8:
                            allocations = np.zeros((n_states, 2))
                            payments = np.zeros(n_states)
                    else:
                        allocations = np.zeros((n_states, 2))
                        payments = np.zeros(n_states)
                except Exception:
                    allocations = np.zeros((n_states, 2))
                    payments = np.zeros(n_states)
        except Exception:
            allocations = np.zeros((n_states, 2))
            payments = np.zeros(n_states)

    # Final check: ensure non-negative payments and valid allocations
    payments = np.maximum(payments, 0.0)
    allocations = np.clip(allocations, 0.0, 1.0)

    return {"allocations": allocations, "payments": payments}
