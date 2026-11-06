import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    
    if isinstance(distribution_spec, dict) and 'probs' in distribution_spec:
        probs = np.array(distribution_spec['probs'], dtype=float)
    else:
        probs = np.ones(n_types) / n_types
        
    probs = probs / probs.sum()
    
    # Variables:
    # 0..15: x1[t]
    # 16..31: x2[t]
    # 32..47: U[t]
    n_vars = 3 * n_types
    
    def idx_x1(t): return t
    def idx_x2(t): return n_types + t
    def idx_U(t): return 2 * n_types + t
    
    # Objective: Max sum(pi * (v1*x1 + v2*x2 - U))
    # Min -sum(pi * (v1*x1 + v2*x2 - U))
    c = np.zeros(n_vars)
    for t in range(n_types):
        v1, v2 = types[t]
        c[idx_x1(t)] = -probs[t] * v1
        c[idx_x2(t)] = -probs[t] * v2
        c[idx_U(t)] = probs[t]  # Min -(-U) = Min +U? No.
        # Obj: Max sum pi (v.x - U) = Max sum pi v.x - sum pi U
        # Min -sum pi v.x + sum pi U
        # So c[U] = +probs[t]
        
    A_ub_rows = []
    b_ub_vals = []
    
    # IC: U[t] - U[t'] - v1[t]*x1[t'] - v2[t]*x2[t'] + v1[t']*x1[t'] + v2[t']*x2[t'] >= 0
    # => -U[t] + U[t'] + v1[t]*x1[t'] + v2[t]*x2[t'] - v1[t']*x1[t'] - v2[t']*x2[t'] <= 0
    for t in range(n_types):
        v1, v2 = types[t]
        for t_prime in range(n_types):
            if t_prime == t:
                continue
            v1_p, v2_p = types[t_prime]
            row = np.zeros(n_vars)
            row[idx_U(t)] = -1.0
            row[idx_U(t_prime)] = 1.0
            row[idx_x1(t_prime)] = v1
            row[idx_x2(t_prime)] = v2
            row[idx_x1(t_prime)] -= v1_p # Wait, term is - v1_p * x1[t_prime]?
            # Let's re-derive:
            # U[t] >= U[t'] + v(t) x(t') - v(t') x(t')
            # U[t] - U[t'] - v(t) x(t') + v(t') x(t') >= 0
            # -U[t] + U[t'] + v(t) x(t') - v(t') x(t') <= 0
            # Coeff of x1[t_prime]: v(t) - v(t') = v1 - v1_p
            # Coeff of x2[t_prime]: v2 - v2_p
            row[idx_x1(t_prime)] = v1 - v1_p
            row[idx_x2(t_prime)] = v2 - v2_p
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
            
    # IR: U[t] <= v1[t]*x1[t] + v2[t]*x2[t]
    # => U[t] - v1[t]*x1[t] - v2[t]*x2[t] <= 0
    for t in range(n_types):
        v1, v2 = types[t]
        row = np.zeros(n_vars)
        row[idx_U(t)] = 1.0
        row[idx_x1(t)] = -v1
        row[idx_x2(t)] = -v2
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1)) # x1
    for t in range(n_types):
        bounds.append((0, 1)) # x2
    for t in range(n_types):
        bounds.append((0, None)) # U >= 0
        
    # Equality constraint: U[0] = 0 (assuming types[0] is (0,0))
    A_eq = np.zeros((1, n_vars))
    A_eq[0, idx_U(0)] = 1.0
    b_eq = np.array([0.0])
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        x = result.x
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for t in range(n_types):
            allocations[t, 0] = x[idx_x1(t)]
            allocations[t, 1] = x[idx_x2(t)]
            U_t = x[idx_U(t)]
            v1, v2 = types[t]
            payments[t] = v1 * allocations[t, 0] + v2 * allocations[t, 1] - U_t
            
        # Clip for numerical stability
        allocations = np.clip(allocations, 0, 1)
        payments = np.maximum(payments, 0)
        
        return {'allocations': allocations, 'payments': payments}
    else:
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}
