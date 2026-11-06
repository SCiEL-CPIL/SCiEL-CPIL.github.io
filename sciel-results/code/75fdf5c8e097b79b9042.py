"""Optimal single-bidder auction via Myerson's Lemma (independent) or LP (dependent)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

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
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    n_goods = 2
    
    # Parse probabilities
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
    
    # Check for independence
    is_independent = True
    # Marginal P(v1)
    p1_marg = np.zeros(4)
    p2_marg = np.zeros(4)
    for i, (v1, v2) in enumerate(types):
        p1_marg[v1] += probs[i]
        p2_marg[v2] += probs[i]
        
    for i, (v1, v2) in enumerate(types):
        expected = p1_marg[v1] * p2_marg[v2]
        if abs(probs[i] - expected) > 1e-9:
            is_independent = False
            break
            
    if is_independent:
        # Myerson's Lemma for independent goods
        # Compute virtual values for each good
        def get_virtual_values(marg_prob, vals):
            # marg_prob: [P(0), P(1), P(2), P(3)]
            # vals: [0, 1, 2, 3]
            virt = np.zeros(4)
            for v in vals:
                # F(v) = sum_{u<=v} P(u)
                # f(v) = P(v)
                # phi(v) = v - (1-F(v))/f(v)
                F_v = np.sum(marg_prob[:v+1])
                f_v = marg_prob[v]
                if f_v > 1e-12:
                    virt[v] = v - (1 - F_v) / f_v
                else:
                    virt[v] = -np.inf
            return virt

        phi1 = get_virtual_values(p1_marg, values)
        phi2 = get_virtual_values(p2_marg, values)
        
        # Allocation: x=1 if phi >= 0
        x1_vals = np.zeros(4)
        x2_vals = np.zeros(4)
        for v in values:
            x1_vals[v] = 1.0 if phi1[v] >= -1e-9 else 0.0
            x2_vals[v] = 1.0 if phi2[v] >= -1e-9 else 0.0
            
        # Construct allocation matrix
        alloc = np.zeros((n_types, 2))
        for i, (v1, v2) in enumerate(types):
            alloc[i, 0] = x1_vals[v1]
            alloc[i, 1] = x2_vals[v2]
            
        # Compute payments
        # p1(v1) = v1 * x1(v1) - sum_{u < v1} x1(u)
        p1_vals = np.zeros(4)
        p2_vals = np.zeros(4)
        for v in values:
            # Sum of allocations for lower types
            sum_x1 = np.sum(x1_vals[:v])
            sum_x2 = np.sum(x2_vals[:v])
            p1_vals[v] = v * x1_vals[v] - sum_x1
            p2_vals[v] = v * x2_vals[v] - sum_x2
            
        # Ensure non-negative payments (IR)
        # In Myerson, if we set U(0)=0, p(0)=0.
        # If p(v) < 0, it means IR is violated.
        # However, with virtual values, the optimal mechanism satisfies IR.
        # We clip to 0 just in case of numerical issues.
        p1_vals = np.maximum(p1_vals, 0)
        p2_vals = np.maximum(p2_vals, 0)
        
        payments = np.zeros(n_types)
        for i, (v1, v2) in enumerate(types):
            payments[i] = p1_vals[v1] + p2_vals[v2]
            
        return {
            'allocations': alloc,
            'payments': payments
        }
    else:
        # Fallback to LP for dependent distributions
        return _solve_lp(probs, types, values)

def _solve_lp(probs, types, values):
    n_types = len(types)
    n_goods = 2
    V = np.array(types, dtype=float)
    
    n_vars = n_types * n_goods + n_types
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_types * n_goods + i] = -probs[i]
        
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
        
    A_ic = []
    b_ic = []
    for i in range(n_types):
        for k in range(n_types):
            if i == k: continue
            row = np.zeros(n_vars)
            row[n_types * n_goods + i] = 1.0
            row[n_types * n_goods + k] = -1.0
            for j in range(n_goods):
                row[i * n_goods + j] = -V[i, j]
                row[k * n_goods + j] = V[i, j]
            A_ic.append(row)
            b_ic.append(0.0)
            
    A_ir = []
    b_ir = []
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[n_types * n_goods + i] = 1.0
        for j in range(n_goods):
            row[i * n_goods + j] = -V[i, j]
        A_ir.append(row)
        b_ir.append(0.0)
        
    A_ub = np.vstack([A_ic, A_ir])
    b_ub = np.concatenate([b_ic, b_ir])
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        x_vals = res.x
        alloc = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for i in range(n_types):
            for j in range(n_goods):
                alloc[i, j] = x_vals[i * n_goods + j]
            payments[i] = x_vals[n_types * n_goods + i]
        alloc = np.clip(alloc, 0, 1)
        payments = np.maximum(payments, 0)
        return {'allocations': alloc, 'payments': payments}
    else:
        # Fallback to zero allocation
        return {'allocations': np.zeros((n_types, 2)), 'payments': np.zeros(n_types)}

# EVOLVE-BLOCK-END
