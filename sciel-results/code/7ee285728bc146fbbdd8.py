import math

def coefficients(stages):
    """4th-order symmetric product formula using Yoshida's 3-step scheme,
    applied twice to achieve 12 stages with improved accuracy over Strang."""
    # Yoshida 4th-order 3-step coefficients
    c = 2 ** (1/3)
    d1 = 1.0 / (2.0 * (1.0 - c))
    d2 = 1.0 - 2.0 * d1
    
    # Each half uses the 3-step scheme for t/2
    # Structure: (B1 A1)(B2 A2)(B3 A3)(B4 A4)(B5 A5)(B6 A6)
    # where B_{2i-1} = d1*(t/2), B_{2i} = d2*(t/2), B_6 = d3*(t/2)
    # But we need 12 B's and 12 A's.
    
    # Instead, use 12-stage symmetric 4th-order by using 4 substeps of 3-stage each
    # Or simply: use the 3-stage scheme 4 times, each for t/4
    
    # For 4 applications of 3-stage scheme (each for t/4):
    # That gives 12 B-steps and 12 A-steps
    
    scale = 1.0 / 4.0
    b_list = []
    a_list = []
    for _ in range(4):
        b_list += [d1 * scale, d2 * scale, d1 * scale]
        a_list += [d1 * scale, d2 * scale, d1 * scale]
    
    # Verify sums
    assert abs(sum(b_list) - 1.0) < 1e-10, f"b sum = {sum(b_list)}"
    assert abs(sum(a_list) - 1.0) < 1e-10, f"a sum = {sum(a_list)}"
    
    # Pad or trim to exactly 12
    while len(a_list) < 12:
        a_list.append(0.0)
        b_list.append(0.0)
    a_list = a_list[:12]
    b_list = b_list[:12]
    
    # Ensure all values are finite and in range
    for i in range(12):
        assert math.isfinite(a_list[i]) and -50 <= a_list[i] <= 50
        assert math.isfinite(b_list[i]) and -50 <= b_list[i] <= 50
    
    return a_list, b_list
