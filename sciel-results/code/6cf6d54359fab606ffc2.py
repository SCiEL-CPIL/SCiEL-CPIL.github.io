import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # Define the 7 Strassen terms compactly
    # Each term is (u_coeffs, v_coeffs, w_coeffs)
    # Coeffs are for [00, 01, 10, 11]
    
    # Term 0: (A00+A11)(B00+B11) -> C00, C11
    t0_u = [1, 0, 0, 1]
    t0_v = [1, 0, 0, 1]
    t0_w = [1, 0, 0, 1]
    
    # Term 1: (A10+A11)B00 -> C10, C11
    # Note: Standard Strassen M2 contributes to C10 (+1) and C11 (-1)
    t1_u = [0, 0, 1, 1]
    t1_v = [1, 0, 0, 0]
    t1_w = [0, 0, 1, -1]
    
    # Term 2: A00(B01-B11) -> C01, C11
    t2_u = [1, 0, 0, 0]
    t2_v = [0, 1, 0, -1]
    t2_w = [0, 1, 0, 1]
    
    # Term 3: A11(B10-B00) -> C00, C10
    t3_u = [0, 0, 0, 1]
    t3_v = [-1, 0, 1, 0]
    t3_w = [1, 0, 1, 0]
    
    # Term 4: (A00+A01)B11 -> C00, C01
    t4_u = [1, 1, 0, 0]
    t4_v = [0, 0, 0, 1]
    t4_w = [-1, 1, 0, 0]
    
    # Term 5: (A10-A00)(B00+B01) -> C11
    t5_u = [-1, 0, 1, 0]
    t5_v = [1, 1, 0, 0]
    t5_w = [0, 0, 0, 1]
    
    # Term 6: (A01-A11)(B10+B11) -> C00
    t6_u = [0, 1, 0, -1]
    t6_v = [0, 0, 1, 1]
    t6_w = [1, 0, 0, 0]
    
    terms = [
        (t0_u, t0_v, t0_w),
        (t1_u, t1_v, t1_w),
        (t2_u, t2_v, t2_w),
        (t3_u, t3_v, t3_w),
        (t4_u, t4_v, t4_w),
        (t5_u, t5_v, t5_w),
        (t6_u, t6_v, t6_w)
    ]
    
    U = [[t[0][i] for t in terms] for i in range(4)]
    V = [[t[1][i] for t in terms] for i in range(4)]
    W = [[t[2][i] for t in terms] for i in range(4)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
