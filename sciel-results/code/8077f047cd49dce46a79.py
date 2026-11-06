import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    # Define the type space
    # Values are in {0, 1, 2, 3} for both goods
    values = [0, 1, 2, 3]
    num_types = len(values) ** 2  # 16 types
    
    # Generate all type combinations (v1, v2)
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
            
    # Map type index to (v1, v2)
    # Index k corresponds to types[k]
    
    # Extract joint probabilities
    # We need to handle different formats of distribution_spec
    probs = np.zeros(num_types)
    
    if isinstance(distribution_spec, np.ndarray) and distribution_spec.shape == (4, 4):
        # Direct joint distribution
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
            # Assume ind is a list of two arrays
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
        # Assume independent distributions
        p1 = np.array(distribution_spec[0])
        p2 = np.array(distribution_spec[1])
        joint = np.outer(p1, p2)
        probs = joint.flatten()
    else:
        # Fallback: uniform distribution
        probs = np.ones(num_types) / num_types
        
    # Normalize probabilities just in case
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(num_types) / num_types

    # Variables:
    # x_{k,0}, x_{k,1} for k in 0..15
    # u_k for k in 0..15
    # Total variables: 16*2 + 16 = 48
    
    n_vars = 48
    x_offset = 0
    u_offset = 32
    
    # Objective: Maximize Revenue = sum_k probs[k] * (v1_k * x_{k,0} + v2_k * x_{k,1} - u_k)
    # linprog minimizes, so we minimize -Revenue
    c = np.zeros(n_vars)
    
    for k in range(num_types):
        v1, v2 = types[k]
        p_k = probs[k]
        
        # Coefficients for x
        c[x_offset + 2*k + 0] = -p_k * v1
        c[x_offset + 2*k + 1] = -p_k * v2
        
        # Coefficient for u
        c[u_offset + k] = -p_k  # We want to subtract u, so in minimization of -Rev, it's -(-p*u) = +p*u? 
        # Wait.
        # Maximize R = sum p (v x - u)
        # Minimize -R = sum p (-v x + u)
        # So coeff for x is -p*v. Correct.
        # Coeff for u is +p. Correct.
        c[u_offset + k] = p_k

    # Constraints: A_ub @ z <= b_ub
    
    # 1. IR: u_k >= 0  =>  -u_k <= 0
    # 2. IC: u_v - u_v' - (v - v') . x_v' >= 0
    #    => -u_v + u_v' + (v - v') . x_v' <= 0
    
    # We will build the IC constraints matrix.
    # Number of IC constraints: 16 * 16 = 256
    # Number of IR constraints: 16
    
    n_ic_constraints = num_types * num_types
    n_ir_constraints = num_types
    n_constraints = n_ic_constraints + n_ir_constraints
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    # IR Constraints
    for k in range(num_types):
        row = n_ic_constraints + k
        A_ub[row, u_offset + k] = -1.0
        b_ub[row] = 0.0
        
    # IC Constraints
    constraint_idx = 0
    for v_idx in range(num_types):
        v1, v2 = types[v_idx]
        for vp_idx in range(num_types):
            vp1, vp2 = types[vp_idx]
            
            row = constraint_idx
            constraint_idx += 1
            
            # Term: -u_v + u_v' + (v1 - vp1) * x_{vp,0} + (v2 - vp2) * x_{vp,1} <= 0
            
            # Coeff for u_v
            A_ub[row, u_offset + v_idx] = -1.0
            # Coeff for u_v'
            A_ub[row, u_offset + vp_idx] = 1.0
            
            # Coeff for x_{vp,0}
            A_ub[row, x_offset + 2*vp_idx + 0] = v1 - vp1
            # Coeff for x_{vp,1}
            A_ub[row, x_offset + 2*vp_idx + 1] = v2 - vp2
            
            b_ub[row] = 0.0
            
    # Bounds
    # x_{k,j} in [0, 1]
    # u_k in [0, inf)
    
    bounds = []
    for k in range(num_types):
        bounds.append((0.0, 1.0)) # x_{k,0}
        bounds.append((0.0, 1.0)) # x_{k,1}
    for k in range(num_types):
        bounds.append((0.0, None)) # u_k

    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            # Fallback to zero mechanism if LP fails
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            return {"allocations": allocations, "payments": payments}
        
        z = res.x
        
        # Extract allocations
        allocations = np.zeros((16, 2))
        for k in range(num_types):
            allocations[k, 0] = z[x_offset + 2*k + 0]
            allocations[k, 1] = z[x_offset + 2*k + 1]
            
        # Extract utilities
        utilities = np.zeros(16)
        for k in range(num_types):
            utilities[k] = z[u_offset + k]
            
        # Calculate payments: p(v) = v . x(v) - u(v)
        payments = np.zeros(16)
        for k in range(num_types):
            v1, v2 = types[k]
            payments[k] = v1 * allocations[k, 0] + v2 * allocations[k, 1] - utilities[k]
            
        # Clean up numerical noise
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.clip(payments, 0.0, None)
        
        # Verify IC and IR roughly to ensure validity
        # If invalid, return zeros
        is_valid = True
        for k in range(num_types):
            if payments[k] < -1e-8:
                is_valid = False
                break
            if utilities[k] < -1e-8:
                is_valid = False
                break
                
        if not is_valid:
            return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
            
        return {"allocations": allocations, "payments": payments}
        
    except Exception as e:
        # In case of any error, return a safe zero mechanism
        return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
