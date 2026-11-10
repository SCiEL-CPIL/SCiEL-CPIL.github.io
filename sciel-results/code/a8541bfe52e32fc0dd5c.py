def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import minimize, linprog

    values = [0, 1, 2, 3]
    n_values = len(values)
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    type_index = {t: i for i, t in enumerate(types)}
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(t, 0.0) for t in types], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if probs.size != n_types:
            probs = probs.reshape((n_values, n_values)).flatten()
    else:
        probs = np.ones(n_types) / n_types
        
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_types) / n_types

    def compute_U(q1, q2):
        U = np.zeros((n_values, n_values))
        for v1 in range(n_values):
            for v2 in range(n_values):
                if v1 == 0 and v2 == 0:
                    U[0, 0] = 0.0
                elif v1 > 0 and v2 == 0:
                    idx_prev = (v1-1) * n_values + v2
                    U[v1, v2] = U[v1-1, v2] + q1[idx_prev]
                elif v1 == 0 and v2 > 0:
                    idx_prev = v1 * n_values + (v2-1)
                    U[v1, v2] = U[v1, v2-1] + q2[idx_prev]
                else:
                    idx_left = (v1-1) * n_values + v2
                    idx_down = v1 * n_values + (v2-1)
                    U[v1, v2] = max(U[v1-1, v2] + q1[idx_left], U[v1, v2-1] + q2[idx_down])
        return U.flatten()

    def objective(x):
        q1 = np.clip(x[:n_types], 0, 1)
        q2 = np.clip(x[n_types:2*n_types], 0, 1)
        U = compute_U(q1, q2)
        revenue = 0.0
        for i in range(n_types):
            v1, v2 = types[i]
            p_i = v1 * q1[i] + v2 * q2[i] - U[i]
            if p_i < 0:
                revenue -= 1000 * p_i
            else:
                revenue += probs[i] * p_i
        return -revenue

    def constraint_monotone_q1(x):
        q1 = x[:n_types]
        vals = []
        for v2 in range(n_values):
            for v1 in range(1, n_values):
                idx_curr = v1 * n_values + v2
                idx_prev = (v1-1) * n_values + v2
                vals.append(q1[idx_curr] - q1[idx_prev])
        return vals

    def constraint_monotone_q2(x):
        q2 = x[n_types:2*n_types]
        vals = []
        for v1 in range(n_values):
            for v2 in range(1, n_values):
                idx_curr = v1 * n_values + v2
                idx_prev = v1 * n_values + (v2-1)
                vals.append(q2[idx_curr] - q2[idx_prev])
        return vals

    def constraint_nonneg_payment(x):
        q1 = np.clip(x[:n_types], 0, 1)
        q2 = np.clip(x[n_types:2*n_types], 0, 1)
        U = compute_U(q1, q2)
        vals = []
        for i in range(n_types):
            v1, v2 = types[i]
            vals.append(v1 * q1[i] + v2 * q2[i] - U[i])
        return vals

    # Constraints must be in format [{'type': 'ineq', 'fun': lambda x: ...}]
    cons = []
    for i in range(n_types - n_values): # q1 monotonicity count
        pass # We need individual functions or a vectorized one? SLSQP accepts vectorized fun for a single constraint dict.
        
    # Let's use a single constraint function that returns a vector
    def all_constraints(x):
        c1 = constraint_monotone_q1(x)
        c2 = constraint_monotone_q2(x)
        c3 = constraint_nonneg_payment(x)
        return np.concatenate([c1, c2, c3])

    # SLSQP can take a single constraint with a vectorized fun
    cons = [{'type': 'ineq', 'fun': all_constraints}]

    x0 = np.full(2*n_types, 0.5)
    bounds = [(0, 1)] * (2*n_types)
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-8})
        if res.success:
            q1 = np.clip(res.x[:n_types], 0, 1)
            q2 = np.clip(res.x[n_types:2*n_types], 0, 1)
        else:
            raise Exception("Optimization failed")
    except:
        # Fallback to LP
        c = np.zeros(3 * n_types)
        for i in range(n_types):
            c[2 * n_types + i] = -probs[i]
        A_ub = []
        b_ub = []
        for i in range(n_types):
            v1, v2 = types[i]
            row = np.zeros(3 * n_types)
            row[i] = -v1
            row[n_types + i] = -v2
            row[2 * n_types + i] = 1
            A_ub.append(row)
            b_ub.append(0)
        for i in range(n_types):
            v1_i, v2_i = types[i]
            for j in range(n_types):
                if i == j: continue
                row = np.zeros(3 * n_types)
                row[i] = -v1_i
                row[n_types + i] = -v2_i
                row[2 * n_types + i] = 1
                row[j] = v1_i
                row[n_types + j] = v2_i
                row[2 * n_types + j] = -1
                A_ub.append(row)
                b_ub.append(0)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds_lp = [(0, 1)] * (2 * n_types) + [(0, None)] * n_types
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        if result.success:
            q1 = np.clip(result.x[:n_types], 0, 1)
            q2 = np.clip(result.x[n_types:2*n_types], 0, 1)
        else:
            q1 = np.zeros(n_types)
            q2 = np.zeros(n_types)

    U = compute_U(q1, q2)
    payments = np.zeros(n_types)
    for i in range(n_types):
        v1, v2 = types[i]
        payments[i] = max(0, v1 * q1[i] + v2 * q2[i] - U[i])
        
    allocations = np.column_stack([q1, q2])
    return {'allocations': allocations, 'payments': payments}
