def coefficients(stages):
    """
    Implements a 3-step 4th-order symmetric composition using 12 stages.
    Each 4th-order step uses 4 stages.
    The 4th-order step is a composition of three 2nd-order Strang steps:
    S_4(t) = S_2(theta*t) * S_2((1-2*theta)*t) * S_2(theta*t)
    where theta = 1 / (2 * (1 - 2^(-1/3)))
    
    This maps to 4 non-zero (b, a) pairs per step:
    1. B(theta/2), A(theta)
    2. B(1/2 - theta/2), A(1 - 2*theta)
    3. B(1/2 - theta/2), A(theta)
    4. B(theta/2), A(0)
    """
    if stages != 12:
        # Fallback to uniform if stages is not 12
        # We assume stages=12 based on task description
        # If stages is different, we might need to adjust, but task specifies 12.
        # For safety, if stages != 12, we return a valid 2nd order Strang for that number of stages if possible,
        # or just uniform. But the task says "twelve-stage".
        pass

    a = [0.0] * stages
    b = [0.0] * stages

    num_steps = stages // 4
    if num_steps == 0:
        # Fallback if stages < 4
        dt = 1.0 / max(1, stages // 2)
        for j in range(stages // 2):
            idx1 = 2 * j
            idx2 = 2 * j + 1
            b[idx1] = dt / 2.0
            a[idx1] = dt
            b[idx2] = dt / 2.0
            a[idx2] = 0.0
        return a, b

    dt = 1.0 / num_steps
    
    # Calculate theta for 4th order
    # theta = 1 / (2 * (1 - 2^(-1/3)))
    theta = 1.0 / (2.0 * (1.0 - 2.0 ** (-1.0 / 3.0)))
    
    # Coefficients for one 4th-order step of size dt
    # b1, a1
    b1 = dt * theta / 2.0
    a1 = dt * theta
    
    # b2, a2
    b2 = dt * (0.5 - theta / 2.0)
    a2 = dt * (1.0 - 2.0 * theta)
    
    # b3, a3
    b3 = dt * (0.5 - theta / 2.0)
    a3 = dt * theta
    
    # b4, a4
    b4 = dt * theta / 2.0
    a4 = 0.0
    
    for step in range(num_steps):
        base_idx = step * 4
        b[base_idx + 0] = b1
        a[base_idx + 0] = a1
        
        b[base_idx + 1] = b2
        a[base_idx + 1] = a2
        
        b[base_idx + 2] = b3
        a[base_idx + 2] = a3
        
        b[base_idx + 3] = b4
        a[base_idx + 3] = a4
        
    return a, b
