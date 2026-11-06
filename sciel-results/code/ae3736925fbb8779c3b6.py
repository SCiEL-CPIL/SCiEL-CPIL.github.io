import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    # Define the type space
    values = [0, 1, 2, 3]
    num_types = 16
    
    # Generate all type combinations (v1, v2) in C-order
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
            
    # Extract joint probabilities
    probs = np.zeros(num_types)
    
    if isinstance(distribution_spec, np.ndarray) and distribution_spec.shape == (4, 4):
        probs = distribution_spec.flatten()
    elif isinstance(distribution_spec, dict):
        if 'joint' in distribution_spec:
            joint = distribution_spec['joint']
            if isinstance(joint, np.ndarray) and joint.shape == (4, 4):
                probs = joint.flatten()
            elif isinstance(joint, list) and len(joint) == 16:
                probs = np.array(joint)
        elif 'independent' in distribution_spec:
            ind = distribution_spec['independent']
            p1 = np.array(ind[0])
            p2 = np.array(ind[1])
            joint = np.outer(p1, p2)
            probs = joint.flatten()
        elif 'p1' in distribution_spec and 'p2' in distribution_spec:
            p1 = np.array(distribution_spec['p1'])
            p2 = np.array(distribution_spec['p2'])
            joint = np.outer(p1, p2)
            probs = joint.flatten()
    elif isinstance(distribution_spec, (list, tuple)) and len(distribution_spec) == 2:
        p1 = np.array(distribution_spec[0])
        p2 = np.array(distribution_spec[1])
        joint = np.outer(p1, p2)
        probs = joint.flatten()
    else:
        probs = np.ones(num_types) / num_types
        
    # Normalize probabilities
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(num_types) / num_types

    # Variables:
    # x_{k,0}, x_{k,1} for k in 0..15 (32 vars)
    # u_k for k in 0..15 (16 vars)
    # Total: 48
    
    n_vars = 48
    x_offset = 0
    u_offset = 32
    
    # Objective: Maximize Revenue = sum_k probs[k] * (v1_k * x_{k,0} + v2_k * x_{k,1} - u_k)
    # Minimize -Revenue
    c = np.zeros(n_vars)
    
    for k in range(num_types):
        v1, v2 = types[k]
        p_k = probs[k]
        
        c[x_offset + 2*k + 0] = -p_k * v1
        c[x_offset + 2*k + 1] = -p_k * v2
        c[u_offset + k] = p_k

    # Constraints
    # 1. IR: u_k >= 0  =>  -u_k <= 0
    # 2. IC: u_v - u_v' - (v - v') . x_v' >= 0
    #    => -u_v + u_v' + (v - v') . x_v' <= 0
    # 3. Non-negative Payment: u_k - v1*x_{k,0} - v2*x_{k,1} <= 0
    
    n_ic = num_types * (num_types - 1)
    n_ir = num_types
    n_pay = num_types
    n_constraints = n_ic + n_ir + n_pay
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    # IR Constraints
    for k in range(num_types):
        row = n_ic + k
        A_ub[row, u_offset + k] = -1.0
        b_ub[row] = 0.0
        
    # IC Constraints
    constraint_idx = 0
    for v_idx in range(num_types):
        v1, v2 = types[v_idx]
        for vp_idx in range(num_types):
            if v_idx == vp_idx:
                continue
            vp1, vp2 = types[vp_idx]
            
            row = constraint_idx
            constraint_idx += 1
            
            # -u_v + u_v' + (v1 - vp1) * x_{vp,0} + (v2 - vp2) * x_{vp,1} <= 0
            A_ub[row, u_offset + v_idx] = -1.0
            A_ub[row, u_offset + vp_idx] = 1.0
            A_ub[row, x_offset + 2*vp_idx + 0] = v1 - vp1
            A_ub[row, x_offset + 2*vp_idx + 1] = v2 - vp2
            b_ub[row] = 0.0
            
    # Non-negative Payment Constraints
    for k in range(num_types):
        row = n_ic + n_ir + k
        v1, v2 = types[k]
        
        A_ub[row, u_offset + k] = 1.0
        A_ub[row, x_offset + 2*k + 0] = -v1
        A_ub[row, x_offset + 2*k + 1] = -v2
        b_ub[row] = 0.0
        
    # Bounds
    # x in [0, 1]
    # u in [0, max_possible_utility]
    # Max utility is bounded by max value * max allocation = 3 * 1 + 3 * 1 = 6
    # We can set a tighter bound if needed, but 10 is safe.
    bounds = []
    for k in range(num_types):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for k in range(num_types):
        bounds.append((0.0, 10.0))

    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            # Fallback to zero mechanism
            return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
        
        z = res.x
        
        allocations = np.zeros((16, 2))
        for k in range(num_types):
            allocations[k, 0] = z[x_offset + 2*k + 0]
            allocations[k, 1] = z[x_offset + 2*k + 1]
            
        utilities = np.zeros(16)
        for k in range(num_types):
            utilities[k] = z[u_offset + k]
            
        payments = np.zeros(16)
        for k in range(num_types):
            v1, v2 = types[k]
            payments[k] = v1 * allocations[k, 0] + v2 * allocations[k, 1] - utilities[k]
            
        # Clean up numerical noise
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.clip(payments, 0.0, None)
        
        # Verify validity
        # Check IR: u >= 0 (handled by bounds)
        # Check IC: u(v) >= u(v') + (v-v')x(v')
        # Check Payment >= 0 (handled by constraints)
        
        # Simple check for regret
        for v_idx in range(num_types):
            v1, v2 = types[v_idx]
            for vp_idx in range(num_types):
                if v_idx == vp_idx: continue
                vp1, vp2 = types[vp_idx]
                # Utility of reporting vp when true type is v
                util_report = v1 * allocations[vp_idx, 0] + v2 * allocations[vp_idx, 1] - payments[vp_idx]
                # Utility of reporting truthfully
                util_truth = v1 * allocations[v_idx, 0] + v2 * allocations[v_idx, 1] - payments[v_idx]
                if util_report > util_truth + 1e-8:
                    # Incentive violation
                    return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
                    
        return {"allocations": allocations, "payments": payments}
        
    except Exception:
        return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
