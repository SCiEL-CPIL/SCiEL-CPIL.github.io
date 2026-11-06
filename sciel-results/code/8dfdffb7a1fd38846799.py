"""Incentive-compatible finite auction design via direct LP solve."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    DEADLINE = 85  # 5s safety margin before 90s kill
    
    # Parse inputs
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Try to find a list/array value
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, np.ndarray)) and len(val) == 16:
                    probs = np.array(val, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16
    elif isinstance(distribution_spec, (list, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16
    
    probs = probs / probs.sum()  # Normalize
    
    # Support: 16 type profiles (v1, v2) with vi in {0,1,2,3}
    if support is None:
        support = [(i // 4, i % 4) for i in range(16)]
    elif isinstance(support, dict):
        # Try to extract
        for key in support:
            if isinstance(support[key], (list, tuple)) and len(support[key]) == 16:
                support = support[key]
                break
        else:
            support = [(i // 4, i % 4) for i in range(16)]
    
    # Ensure support is a list of 16 tuples
    if len(support) != 16:
        support = [(i // 4, i % 4) for i in range(16)]
    
    # Map type index to (v1, v2)
    type_map = {}
    for i, (v1, v2) in enumerate(support):
        type_map[i] = (int(v1), int(v2))
    
    # Build LP for optimal incentive-compatible mechanism
    # Variables: x[i][g] for i in 0..15, g in 0..1 (allocation probabilities)
    #             p[i] for i in 0..15 (payments)
    # Total variables: 16*2 + 16 = 48
    
    n_types = 16
    n_goods = 2
    n_vars = n_types * n_goods + n_types  # 48
    
    def x_idx(i, g):
        return i * n_goods + g
    
    def p_idx(i):
        return n_types * n_goods + i
    
    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    # In linprog: minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_idx(i)] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 1. Individual rationality: p[i] - sum_g x[i][g]*v[i][g] <= 0 for all i
    # 2. Regret constraint: for all i, i':
    #    sum_g x[i'][g]*v[i][g] - p[i'] <= sum_g x[i][g]*v[i][g] - p[i] + 1e-8
    #    => sum_g x[i'][g]*v[i][g] - p[i'] - sum_g x[i][g]*v[i][g] + p[i] <= 1e-8
    # 3. Feasibility: 0 <= x[i][g] <= 1 (handled by bounds)
    # 4. Nonnegative payments: p[i] >= 0 (handled by bounds)
    
    A_ub_rows = []
    b_ub_rows = []
    
    # IR constraints: p[i] - sum_g x[i][g]*v[i][g] <= 0
    for i in range(n_types):
        row = np.zeros(n_vars)
        v1, v2 = type_map[i]
        row[x_idx(i, 0)] = -v1
        row[x_idx(i, 1)] = -v2
        row[p_idx(i)] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
    
    # Regret constraints: for all i, i' (i != i' is enough but do all for safety)
    # regret(i, i') = [sum_g x[i'][g]*v[i][g] - p[i']] - [sum_g x[i][g]*v[i][g] - p[i]] <= 1e-8
    # => sum_g x[i'][g]*v[i][g] - p[i'] - sum_g x[i][g]*v[i][g] + p[i] <= 1e-8
    for i in range(n_types):
        v1_i, v2_i = type_map[i]
        for ip in range(n_types):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            v1_ip, v2_ip = type_map[ip]
            # sum_g x[ip][g]*v_i[g]
            row[x_idx(ip, 0)] += v1_i
            row[x_idx(ip, 1)] += v2_i
            # - p[ip]
            row[p_idx(ip)] += -1.0
            # - sum_g x[i][g]*v_i[g]
            row[x_idx(i, 0)] += -v1_i
            row[x_idx(i, 1)] += -v2_i
            # + p[i]
            row[p_idx(i)] += 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(1e-8)
    
    A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_ub_rows) if b_ub_rows else np.zeros(0)
    
    # Bounds: 0 <= x[i][g] <= 1, p[i] >= 0 (unbounded above)
    bounds = []
    for i in range(n_types):
        for g in range(n_goods):
            bounds.append((0, 1))
        bounds.append((0, None))
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: simple uniform allocation with zero payments
            alloc = np.ones((16, 2))
            pays = np.zeros(16)
            return {'allocations': alloc, 'payments': pays}
        
        x_opt = result.x
        
        # Extract allocations and payments
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        for i in range(n_types):
            for g in range(n_goods):
                allocations[i, g] = x_opt[x_idx(i, g)]
            payments[i] = x_opt[p_idx(i)]
        
        # Clamp small numerical errors
        allocations = np.clip(allocations, 0, 1)
        payments = np.clip(payments, 0, None)
        
        # Verify constraints and fix if needed
        if time.time() < DEADLINE:
            # Check IR
            for i in range(n_types):
                v1, v2 = type_map[i]
                utility = v1 * allocations[i, 0] + v2 * allocations[i, 1] - payments[i]
                if utility < -1e-6:
                    # Fix by reducing payment
                    payments[i] = max(0, v1 * allocations[i, 0] + v2 * allocations[i, 1])
            
            # Check regret
            for i in range(n_types):
                v1_i, v2_i = type_map[i]
                util_i = v1_i * allocations[i, 0] + v2_i * allocations[i, 1] - payments[i]
                for ip in range(n_types):
                    if i == ip:
                        continue
                    v1_ip, v2_ip = type_map[ip]
                    # Utility of type i if reporting ip
                    util_i_as_ip = v1_i * allocations[ip, 0] + v2_i * allocations[ip, 1] - payments[ip]
                    if util_i_as_ip > util_i + 1e-8:
                        # Violation: increase payment of ip or decrease allocation
                        # Simple fix: increase payment[ip] to close the gap
                        gap = util_i_as_ip - util_i
                        payments[ip] += gap + 1e-10
                        # But this might break IR for ip
                        v1_ip, v2_ip = type_map[ip]
                        util_ip = v1_ip * allocations[ip, 0] + v2_ip * allocations[ip, 1] - payments[ip]
                        if util_ip < -1e-6:
                            # Reduce allocation
                            scale = max(0, (payments[ip] - 1e-10) / (v1_ip * allocations[ip, 0] + v2_ip * allocations[ip, 1] + 1e-15))
                            allocations[ip] *= scale
                            payments[ip] = max(0, v1_ip * allocations[ip, 0] + v2_ip * allocations[ip, 1])
            
            # Final clamp
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)
        
        return {'allocations': allocations, 'payments': payments}
    
    except Exception as e:
        # Fallback: simple mechanism
        alloc = np.ones((16, 2))
        pays = np.zeros(16)
        return {'allocations': alloc, 'payments': pays}

# EVOLVE-BLOCK-END
