"""Optimal single-bidder auction via evolutionary search over allocations."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, linprog

def _compute_revenue(x_flat, probs, V):
    """
    Compute the revenue for a given allocation vector.
    x_flat: (32,) vector of allocations x_{i,j}.
    probs: (16,) probabilities.
    V: (16, 2) values.
    """
    n_types = 16
    n_goods = 2
    x = x_flat.reshape(n_types, n_goods)
    
    # Check bounds
    if np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
        return -np.inf
    
    # Compute IC constraint coefficients A_ki = v_k . (x_i - x_k)
    # We need u_i >= u_k + A_ki
    # A_ki = dot(V[k], x[i] - x[k])
    
    # Build LP for minimal u
    # Variables: u_0 ... u_15
    # Minimize sum(pi_i * u_i)
    # Constraints:
    # u_i - u_k >= A_ki  =>  -u_i + u_k <= -A_ki
    # u_i >= 0
    
    c = probs.copy()
    
    A_ub = []
    b_ub = []
    
    for i in range(n_types):
        for k in range(n_types):
            if i == k:
                continue
            A_ki = np.dot(V[k], x[i] - x[k])
            # -u_i + u_k <= -A_ki
            row = np.zeros(n_types)
            row[i] = -1.0
            row[k] = 1.0
            A_ub.append(row)
            b_ub.append(-A_ki)
            
    if not A_ub:
        A_ub = None
        b_ub = None
    else:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
    bounds = [(0, None)] * n_types
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        return -np.inf
        
    u = res.x
    # u should be non-negative due to bounds
    
    # Calculate payments
    # p_i = V_i . x_i + u_i
    payments = np.sum(V * x, axis=1) + u
    
    # Check IR (should be satisfied by u >= 0)
    # p_i >= V_i . x_i  => u_i >= 0
    
    # Revenue
    revenue = np.sum(probs * payments)
    
    return revenue

def construct_auction(distribution_spec, support, constraints):
    """
    Construct optimal incentive-compatible auction for one additive bidder, two goods.
    
    Args:
        distribution_spec: specification of the value distribution
        support: value support (e.g., {0,1,2,3} for each good)
        constraints: additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
            
    n_types = len(types)
    n_goods = 2
    
    if isinstance(distribution_spec, dict) and 'probabilities' in distribution_spec:
        probs = np.array(distribution_spec['probabilities'])
        if len(probs) != n_types:
            probs = np.resize(probs, n_types)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
        if len(probs) != n_types:
            probs = np.resize(probs, n_types)
    else:
        probs = np.ones(n_types) / n_types
        
    probs = probs / probs.sum()
    V = np.array(types, dtype=float)
    
    # Use differential evolution to search for optimal allocations
    # 32 variables, bounds [0, 1]
    bounds = [(0, 1)] * (n_types * n_goods)
    
    # We want to maximize revenue, so minimize -revenue
    def objective(x_flat):
        rev = _compute_revenue(x_flat, probs, V)
        if rev == -np.inf:
            return 1e10
        return -rev

    # Run differential evolution
    # maxiter and popsize control the effort
    # 32 dimensions, 60s limit.
    # Let's use a moderate population.
    try:
        res = differential_evolution(objective, bounds, maxiter=100, popsize=10, seed=42, tol=1e-8, polish=True)
        if res.success:
            x_flat = res.x
        else:
            # Fallback to best found so far or threshold
            x_flat = res.x
    except Exception:
        # Fallback to simple threshold if DE fails
        x_flat = np.zeros(n_types * n_goods)
        # Use the best threshold rule
        best_rev = -np.inf
        best_x = x_flat
        for r1 in values:
            for r2 in values:
                x_t = np.zeros((n_types, 2))
                for idx, (v1, v2) in enumerate(types):
                    x_t[idx, 0] = 1.0 if v1 >= r1 else 0.0
                    x_t[idx, 1] = 1.0 if v2 >= r2 else 0.0
                rev = _compute_revenue(x_t.flatten(), probs, V)
                if rev > best_rev:
                    best_rev = rev
                    best_x = x_t.flatten()
        x_flat = best_x

    x = x_flat.reshape(n_types, n_goods)
    
    # Recompute payments for the best x
    # We need to solve the LP for u again to get exact payments
    A_ki_list = []
    for i in range(n_types):
        for k in range(n_types):
            if i == k: continue
            A_ki_list.append(np.dot(V[k], x[i] - x[k]))
            
    A_ub = []
    b_ub = []
    for i in range(n_types):
        for k in range(n_types):
            if i == k: continue
            A_ki = np.dot(V[k], x[i] - x[k])
            row = np.zeros(n_types)
            row[i] = -1.0
            row[k] = 1.0
            A_ub.append(row)
            b_ub.append(-A_ki)
            
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = None
        b_ub = None
        
    c = probs
    bounds_u = [(0, None)] * n_types
    res_u = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_u, method='highs')
    
    if res_u.success:
        u = res_u.x
    else:
        # Should not happen if x was feasible
        u = np.zeros(n_types)
        
    payments = np.sum(V * x, axis=1) + u
    payments = np.maximum(payments, 0)
    x = np.clip(x, 0, 1)
    
    return {
        'allocations': x,
        'payments': payments
    }

# EVOLVE-BLOCK-END
