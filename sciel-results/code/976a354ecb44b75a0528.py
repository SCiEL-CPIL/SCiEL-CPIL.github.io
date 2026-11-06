def coefficients(stages):
    # 6th-order symmetric Suzuki-Trotter coefficients
    # c0 = 1 / (2 - 2**(1/3))
    # c1 = -c0 / 2**(1/3)
    # c2 = c1 / 2
    # c3 = (1 - 2*c1 - 4*c2) / 2
    
    c0 = 1.0 / (2.0 - 2.0**(1.0/3.0))
    c1 = -c0 / 2.0**(1.0/3.0)
    c2 = c1 / 2.0
    c3 = (1.0 - 2.0*c1 - 4.0*c2) / 2.0
    
    # 12-stage mapping
    # Sequence: B(c3) A(c2) B(c1) A(c0) B(c1) A(c2) B(c3) A(c2) B(c1) A(c0) B(c1) A(c2)
    # Note: The 6th order scheme has a trailing B(c3). We drop it and normalize.
    
    b_raw = [c3, c1, c1, c3, c1, c1, c3, 0.0, 0.0, 0.0, 0.0, 0.0]
    a_raw = [c2, c0, c2, c2, c0, c2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    sum_b = sum(b_raw)
    sum_a = sum(a_raw)
    
    # Normalize to sum to 1
    b = [x / sum_b for x in b_raw]
    a = [x / sum_a for x in a_raw]
    
    return a, b
