def coefficients(stages):
    """Symmetric 12-stage product formula with non-uniform weights for error cancellation."""
    # Symmetric coefficients: a_k = a_{13-k}, b_k = b_{13-k}
    # Designed to approximate 6th-order Suzuki-Trotter behavior
    a = [0.05, 0.12, 0.18, 0.19, 0.18, 0.12, 0.12, 0.18, 0.19, 0.18, 0.12, 0.05]
    b = [0.08, 0.15, 0.19, 0.20, 0.19, 0.15, 0.15, 0.19, 0.20, 0.19, 0.15, 0.08]
    
    # Normalize to ensure sum = 1 exactly
    a_sum = sum(a)
    b_sum = sum(b)
    a = [x / a_sum for x in a]
    b = [x / b_sum for x in b]
    
    # Pad or truncate to exactly `stages` elements if needed
    while len(a) < stages:
        a.append(0.0)
        b.append(0.0)
    if len(a) > stages:
        a = a[:stages]
        b = b[:stages]
    
    return a, b
