import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    # Determine the support size and values
    # Support is given, but task says {0,1,2,3}
    # We assume support is a list or set of values
    # If support is not explicitly a list of 4 values, we infer {0,1,2,3}
    if isinstance(support, (list, tuple, np.ndarray)):
        vals = sorted(list(support))
    else:
        vals = [0, 1, 2, 3]
        
    n_vals = len(vals)
    n_types = n_vals * n_vals
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(vals)}
    
    # Get probabilities f(v)
    # distribution_spec can be:
    # 1. A dictionary { (v1, v2): prob }
    # 2. A dictionary { "v1,v2": prob }
    # 3. A list/array of length 16 (flattened)
    # 4. A string name ("uniform")
    # 5. A 2D array
    
    probs = np.zeros(n_types)
    
    def get_prob(v1, v2):
        idx = val_to_idx[v1] * n_vals + val_to_idx[v2]
        return idx
        
    if isinstance(distribution_spec, dict):
        for key, prob in distribution_spec.items():
            if isinstance(key, tuple) and len(key) == 2:
                v1, v2 = key
                if v1 in val_to_idx and v2 in val_to_idx:
                    probs[get_prob(v1, v2)] = prob
            elif isinstance(key, str):
                parts = key.replace('(', '').replace(')', '').split(',')
                if len(parts) == 2:
                    try:
                        v1 = float(parts[0])
                        v2 = float(parts[1])
                        v1 = int(v1) if v1 == int(v1) else v1
                        v2 = int(v2) if v2 == int(v2) else v2
                        if v1 in val_to_idx and v2 in val_to_idx:
                            probs[get_prob(v1, v2)] = prob
                    except:
                        pass
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        spec = np.array(distribution_spec)
        if spec.ndim == 1 and spec.size == n_types:
            probs = spec
        elif spec.ndim == 2 and spec.shape == (n_vals, n_vals):
            # Flatten row-major: v1 is row, v2 is col?
            # Standard C order: index = v1 * n_vals + v2
            probs = spec.flatten()
    elif isinstance(distribution_spec, str):
        if distribution_spec.lower() in ["uniform", "uniform_dist", "u"]:
            probs[:] = 1.0 / n_types
        elif distribution_spec.lower() in ["independent", "indep"]:
            probs[:] = 1.0 / n_types
        else:
            # Default to uniform if unknown
            probs[:] = 1.0 / n_types
    else:
        probs[:] = 1.0 / n_types
        
    # Normalize probs just in case
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs /= total_prob
    else:
        probs[:] = 1.0 / n_types

    # Setup LP
    # Variables: x_1[0..15], x_2[0..15]
    # Total 32 variables
    n_vars = 2 * n_types
    
    # Objective: Maximize Revenue
    # c = -coefficients (for minimization)
    c = np.zeros(n_vars)
    
    # Calculate coefficients for x_1 and x_2
    # C_1(j) = f_j * v1_j - sum_{k: v2_k == v2_j, v1_k >= v1_j} f_k
    # C_2(j) = f_j * v2_j - sum_{k: v1_k == v1_j, v2_k >= v2_j} f_k
    
    # Precompute sums for efficiency
    # sum_1[v1, v2] = sum_{k: v2_k == v2, v1_k >= v1} f_k
    # sum_2[v1, v2] = sum_{k: v1_k == v1, v2_k >= v2} f_k
    
    sum_1 = np.zeros((n_vals, n_vals))
    sum_2 = np.zeros((n_vals, n_vals))
    
    # Create index arrays
    v1s = [vals[i] for i in range(n_vals)]
    v2s = [vals[j] for j in range(n_vals)]
    
    # Reshape probs to 2D for easier manipulation
    # P[i, j] = f(vals[i], vals[j])
    P = probs.reshape((n_vals, n_vals))
    
    for i in range(n_vals):
        for j in range(n_vals):
            # sum_1[i, j] = sum_{i' >= i} P[i', j]
            sum_1[i, j] = np.sum(P[i:, j])
            # sum_2[i, j] = sum_{j' >= j} P[i, j']
            sum_2[i, j] = np.sum(P[i, j:])
            
    for i in range(n_vals):
        for j in range(n_vars): # Wait, j is type index
            pass
            
    # Let's iterate over types
    for i in range(n_vals):
        for j in range(n_vals):
            idx = i * n_vals + j
            v1 = vals[i]
            v2 = vals[j]
            f_ij = P[i, j]
            
            # Coeff for x_1[idx]
            c_1 = f_ij * v1 - sum_1[i, j]
            # Coeff for x_2[idx]
            c_2 = f_ij * v2 - sum_2[i, j]
            
            c[idx] = -c_1 # Minimize -Revenue
            c[n_types + idx] = -c_2

    # Constraints
    # 1. Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * n_vars
    
    # 2. Monotonicity
    # A_ub @ x <= b_ub
    
    A_ub_rows = []
    b_ub_rows = []
    
    # Monotonicity for x_1
    # For each v2 (j), and for each i < i', x_1(i, j) <= x_1(i', j)
    # => x_1(i, j) - x_1(i', j) <= 0
    for j in range(n_vals):
        for i in range(n_vals - 1):
            for i_prime in range(i + 1, n_vals):
                row = np.zeros(n_vars)
                idx_curr = i * n_vals + j
                idx_next = i_prime * n_vals + j
                row[idx_curr] = 1.0
                row[n_types + idx_next] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
                
    # Monotonicity for x_2
    # For each v1 (i), and for each j < j', x_2(i, j) <= x_2(i, j')
    # => x_2(i, j) - x_2(i, j') <= 0
    for i in range(n_vals):
        for j in range(n_vals - 1):
            for j_prime in range(j + 1, n_vals):
                row = np.zeros(n_vars)
                idx_curr = i * n_vals + j
                idx_next = i * n_vals + j_prime
                row[n_types + idx_curr] = 1.0
                row[n_types + idx_next] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(0.0)
                
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            # Fallback to uniform allocation if LP fails
            x_1 = np.ones(n_types)
            x_2 = np.ones(n_types)
        else:
            x_1 = res.x[:n_types]
            x_2 = res.x[n_types:]
    except Exception:
        x_1 = np.ones(n_types)
        x_2 = np.ones(n_types)
        
    # Clip to [0, 1] to handle numerical issues
    x_1 = np.clip(x_1, 0, 1)
    x_2 = np.clip(x_2, 0, 1)
    
    # Calculate Payments
    # p(v) = v1*x1(v) + v2*x2(v) - rent(v)
    # rent(v) = sum_{t: v1_min < t <= v1} x1(t, v2) + sum_{t: v2_min < t <= v2} x2(v1, t)
    # Since values are discrete and we sum over the support steps.
    # The "step" in the integral corresponds to the difference between adjacent support points.
    # Here support is {0, 1, 2, 3}. Steps are 1.
    
    payments = np.zeros(n_types)
    
    # Precompute cumulative sums for rent
    # rent_1[i, j] = sum_{k=0}^{i-1} x_1[k, j]  (sum over v1 < v1_i)
    # rent_2[i, j] = sum_{k=0}^{j-1} x_2[i, k]  (sum over v2 < v2_j)
    
    X1 = x_1.reshape((n_vals, n_vals))
    X2 = x_2.reshape((n_vals, n_vals))
    
    # Cumulative sum along axis 0 (v1) for X1, excluding current
    # cumsum includes current, so we shift or subtract
    cum_X1 = np.cumsum(X1, axis=0)
    cum_X2 = np.cumsum(X2, axis=1)
    
    for i in range(n_vals):
        for j in range(n_vals):
            idx = i * n_vals + j
            v1 = vals[i]
            v2 = vals[j]
            
            # Rent from good 1: sum of x_1 for types with v1' < v1 and same v2
            # This is sum_{k=0}^{i-1} X1[k, j]
            if i > 0:
                rent_1 = np.sum(X1[:i, j])
            else:
                rent_1 = 0.0
                
            # Rent from good 2: sum of x_2 for types with v2' < v2 and same v1
            if j > 0:
                rent_2 = np.sum(X2[i, :j])
            else:
                rent_2 = 0.0
                
            payments[idx] = v1 * X1[i, j] + v2 * X2[i, j] - rent_1 - rent_2
            
    # Ensure non-negative payments? 
    # IR requires p(v) <= v.x(v). 
    # If p(v) becomes negative due to numerical error, it's still feasible (p>=0 is required? 
    # "nonnegative payments" is a constraint).
    # So we must clip payments to 0?
    # Wait, if p(v) < 0, it violates "nonnegative payments".
    # However, with the optimal LP solution, p(v) should be >= 0 because 
    # the rent is the area under the curve, and v*x(v) is the value.
    # For the lowest type (0,0), p(0,0) = 0.
    # For others, p(v) >= 0 is generally true for monotone allocations.
    # Let's clip to 0 to be safe against floating point errors.
    payments = np.clip(payments, 0, None)
    
    # Format Output
    # allocations: (16, 2)
    # payments: (16,)
    
    allocations = np.column_stack((x_1, x_2))
    
    return {
        'allocations': allocations,
        'payments': payments
    }
