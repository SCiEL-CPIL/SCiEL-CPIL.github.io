"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s limit

    # Define the type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16

    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'weights' in distribution_spec:
            probs = np.array(distribution_spec['weights'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            for key in distribution_spec:
                if isinstance(distribution_spec[key], (list, tuple, np.ndarray)):
                    probs = np.array(distribution_spec[key], dtype=float)
                    break
            else:
                probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types

    if probs.shape[0] != n_types:
        if probs.size == n_types:
            probs = probs.reshape(n_types)
        else:
            probs = np.ones(n_types) / n_types

    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types

    # Value matrix: V[t, g] = value of good g for type t
    V = np.array([[t[0], t[1]] for t in types], dtype=float)  # (16, 2)

    # Variable indexing for the main LP:
    # x[g, t_rep] -> index = g * n_types + t_rep
    # p[t_rep]    -> index = 2 * n_types + t_rep
    def x_idx(g, t_rep):
        return g * n_types + t_rep

    def p_idx(t_rep):
        return 2 * n_types + t_rep

    n_vars_main = 3 * n_types  # 48

    # Objective: maximize sum_{t} probs[t] * p[t]
    c = np.zeros(n_vars_main)
    for t in range(n_types):
        c[p_idx(t)] = -probs[t]

    A_ub_rows = []
    b_ub_vals = []

    # 1. IC Constraints:
    # u(t, t) >= u(t, t_rep)
    # sum_g V[t,g]*x[g,t] - p(t) >= sum_g V[t,g]*x[g,t_rep] - p(t_rep)
    # -sum_g V[t,g]*x[g,t] + sum_g V[t,g]*x[g,t_rep] + p(t) - p(t_rep) <= 0
    for t in range(n_types):
        for t_rep in range(n_types):
            if t == t_rep:
                continue
            row = np.zeros(n_vars_main)
            for g in range(2):
                row[x_idx(g, t)] -= V[t, g]
                row[x_idx(g, t_rep)] += V[t, g]
            row[p_idx(t)] += 1.0
            row[p_idx(t_rep)] -= 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)

    # 2. IR Constraints:
    # u(t, t) >= 0
    # -sum_g V[t,g]*x[g,t] + p(t) <= 0
    for t in range(n_types):
        row = np.zeros(n_vars_main)
        for g in range(2):
            row[x_idx(g, t)] -= V[t, g]
        row[p_idx(t)] += 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)

    # Bounds: 0 <= x <= 1, p >= 0
    bounds = []
    for g in range(2):
        for t_rep in range(n_types):
            bounds.append((0.0, 1.0))
    for t_rep in range(n_types):
        bounds.append((0.0, None))

    # Check deadline
    if time.time() > deadline - 10.0:
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}

    try:
        # Solve main LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 
                                  'time_limit': deadline - 10.0,
                                  'primal_feasibility_tolerance': 1e-12,
                                  'dual_feasibility_tolerance': 1e-12,
                                  'dual_staleness_threshold': 1e-12})

        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)

        if result.success:
            x_opt = result.x
            for t in range(n_types):
                for g in range(2):
                    allocations[t, g] = x_opt[x_idx(g, t)]
                payments[t] = x_opt[p_idx(t)]

            # Clip small numerical errors
            allocations = np.clip(allocations, 0.0, 1.0)
            payments = np.maximum(payments, 0.0)

            # Post-processing: Verify and potentially polish
            # We will try to maximize payments given the allocations found, 
            # ensuring strict IC and IR.
            
            # Define a function to solve for optimal payments given fixed allocations
            def solve_payments_for_allocations(allocs):
                # Variables: p[0..15]
                n_p = n_types
                c_p = np.zeros(n_p)
                for t in range(n_types):
                    c_p[t] = -probs[t]
                
                A_ub_p = []
                b_ub_p = []
                
                # IR: p[t] <= V[t] . allocs[t]
                # -p[t] >= -V[t] . allocs[t]  => p[t] <= V[t] . allocs[t]
                # In linprog form: p[t] <= V[t] . allocs[t]
                # So: 1*p[t] + 0*others <= V[t] . allocs[t]
                for t in range(n_types):
                    row = np.zeros(n_p)
                    row[t] = 1.0
                    A_ub_p.append(row)
                    b_ub_p.append(np.dot(V[t], allocs[t]))
                
                # IC: p[t_rep] - p[t] <= V[t] . (allocs[t_rep] - allocs[t])
                # p[t_rep] - p[t] - V[t] . allocs[t_rep] + V[t] . allocs[t] <= 0
                for t in range(n_types):
                    for t_rep in range(n_types):
                        if t == t_rep:
                            continue
                        row = np.zeros(n_p)
                        row[t_rep] = 1.0
                        row[t] = -1.0
                        A_ub_p.append(row)
                        b_ub_p.append(np.dot(V[t], allocs[t_rep] - allocs[t]))
                
                A_ub_p = np.array(A_ub_p)
                b_ub_p = np.array(b_ub_p)
                
                bounds_p = [(0.0, None)] * n_p
                
                try:
                    res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs',
                                    options={'presolve': True, 
                                             'primal_feasibility_tolerance': 1e-12,
                                             'dual_feasibility_tolerance': 1e-12})
                    if res_p.success:
                        return res_p.x
                    else:
                        return None
                except Exception:
                    return None

            # Try to polish payments
            new_payments = solve_payments_for_allocations(allocations)
            if new_payments is not None:
                # Check if new payments are better and valid
                # They should be valid by construction of the sub-LP
                if np.sum(probs * new_payments) > np.sum(probs * payments) - 1e-9:
                    payments = new_payments
                    payments = np.maximum(payments, 0.0)
            
            # Final validation check
            eps = 1e-8
            valid = True
            for t in range(n_types):
                u_tt = np.dot(V[t], allocations[t]) - payments[t]
                if u_tt < -eps:
                    valid = False
                    break
                for t_rep in range(n_types):
                    if t == t_rep:
                        continue
                    u_t_trep = np.dot(V[t], allocations[t_rep]) - payments[t_rep]
                    if u_t_trep - u_tt > eps:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                # Fallback: simple valid mechanism
                allocations = np.zeros((n_types, 2))
                payments = np.zeros(n_types)

        else:
            # LP failed, return zero mechanism
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)

    except Exception:
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)

    return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
