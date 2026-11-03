"""Rewritten program for Matrix multiplication 2,4,4 using Strassen-like decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 4
    # Rank is 4 * 7 = 28
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen multiplications for a 2x2 matrix product
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11) Y00
    # M3 = X00 (Y10 - Y11)
    # M4 = X11 (Y01 - Y00)
    # M5 = (X00 + X10) Y11
    # M6 = (X01 - X00)(Y00 + Y10)
    # M7 = (X10 - X11)(Y11 + Y01)
    
    # U factors (coefficients for X inputs)
    # V factors (coefficients for Y inputs)
    # W factors (coefficients for Z outputs, added to C)
    
    # Local indices for X (2x2): 0:00, 1:01, 2:10, 3:11
    # Local indices for Y (2x2): 0:00, 1:01, 2:10, 3:11
    # Local indices for Z (2x2): 0:00, 1:01, 2:10, 3:11
    
    u_coeffs = [
        [1, 0, 0, 1],  # M1: X00 + X11
        [0, 0, 1, 1],  # M2: X10 + X11
        [1, 0, 0, 0],  # M3: X00
        [0, 0, 0, 1],  # M4: X11
        [1, 0, 1, 0],  # M5: X00 + X10
        [0, 1, -1, 0], # M6: X01 - X00
        [0, 0, 1, -1]  # M7: X10 - X11
    ]
    
    v_coeffs = [
        [1, 0, 0, 1],  # M1: Y00 + Y11
        [1, 0, 0, 0],  # M2: Y00
        [0, -1, 1, 0], # M3: Y10 - Y11
        [0, 1, 0, -1], # M4: Y01 - Y00
        [0, 0, 0, 1],  # M5: Y11
        [1, 0, 1, 0],  # M6: Y00 + Y10
        [0, 1, 0, 1]   # M7: Y11 + Y01
    ]
    
    w_coeffs = [
        [1, 0, 0, -1, 1, 0, 1],  # Z00 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],   # Z01 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # Z10 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0]   # Z11 = M1 - M2 + M3 + M6
    ]
    # Note: w_coeffs[z_idx][m_idx] is the coefficient of M_m in Z_z
    
    # Define the 4 sub-problems
    # Each sub-problem maps local X indices to global U indices
    # Each sub-problem maps local Y indices to global V indices
    # Each sub-problem maps local Z indices to global W indices
    
    # Global U index = i * m + j
    # Global V index = j * p + k
    # Global W index = k * n + i
    
    # Sub-problem 1: P1 = A1 B11
    # A1 is A[:, 0:2] -> U indices: 0, 1 (i=0), 4, 5 (i=1)
    # Local X: 0->00, 1->01, 2->10, 3->11
    # Global U: 0->0, 1->1, 2->4, 3->5
    u_map_1 = [0, 1, 4, 5]
    
    # B11 is B[0:2, 0:2] -> V indices: 0, 1 (j=0), 4, 5 (j=1)
    # Local Y: 0->00, 1->01, 2->10, 3->11
    # Global V: 0->0, 1->1, 2->4, 3->5
    v_map_1 = [0, 1, 4, 5]
    
    # Output adds to C[:, 0:2]
    # Local Z: 0->00, 1->01, 2->10, 3->11
    # Global W: C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
    # W index = k * 2 + i
    # Z00 (k=0, i=0) -> 0
    # Z01 (k=1, i=0) -> 2
    # Z10 (k=0, i=1) -> 1
    # Z11 (k=1, i=1) -> 3
    w_map_1 = [0, 2, 1, 3]
    
    # Sub-problem 2: P2 = A2 B21
    # A2 is A[:, 2:4] -> U indices: 2, 3 (i=0), 6, 7 (i=1)
    # Local X: 0->00, 1->01, 2->10, 3->11
    # Global U: 0->2, 1->3, 2->6, 3->7
    u_map_2 = [2, 3, 6, 7]
    
    # B21 is B[2:4, 0:2] -> V indices: 8, 9 (j=2), 12, 13 (j=3)
    # Local Y: 0->00, 1->01, 2->10, 3->11
    # Global V: 0->8, 1->9, 2->12, 3->13
    v_map_2 = [8, 9, 12, 13]
    
    # Output adds to C[:, 0:2]
    # Same W mapping as P1
    w_map_2 = [0, 2, 1, 3]
    
    # Sub-problem 3: P3 = A1 B12
    # A1 is A[:, 0:2] -> U indices: 0, 1, 4, 5
    u_map_3 = [0, 1, 4, 5]
    
    # B12 is B[0:2, 2:4] -> V indices: 2, 3 (j=0), 6, 7 (j=1)
    # Local Y: 0->00, 1->01, 2->10, 3->11
    # Global V: 0->2, 1->3, 2->6, 3->7
    v_map_3 = [2, 3, 6, 7]
    
    # Output adds to C[:, 2:4]
    # Z00 (k=2, i=0) -> 2*2+0 = 4
    # Z01 (k=3, i=0) -> 3*2+0 = 6
    # Z10 (k=2, i=1) -> 2*2+1 = 5
    # Z11 (k=3, i=1) -> 3*2+1 = 7
    w_map_3 = [4, 6, 5, 7]
    
    # Sub-problem 4: P4 = A2 B22
    # A2 is A[:, 2:4] -> U indices: 2, 3, 6, 7
    u_map_4 = [2, 3, 6, 7]
    
    # B22 is B[2:4, 2:4] -> V indices: 10, 11 (j=2), 14, 15 (j=3)
    # Local Y: 0->00, 1->01, 2->10, 3->11
    # Global V: 0->10, 1->11, 2->14, 3->15
    v_map_4 = [10, 11, 14, 15]
    
    # Output adds to C[:, 2:4]
    # Same W mapping as P3
    w_map_4 = [4, 6, 5, 7]
    
    sub_problems = [
        (u_map_1, v_map_1, w_map_1),
        (u_map_2, v_map_2, w_map_2),
        (u_map_3, v_map_3, w_map_3),
        (u_map_4, v_map_4, w_map_4)
    ]
    
    col_offset = 0
    for u_map, v_map, w_map in sub_problems:
        for m_idx in range(7):
            # Fill U
            for local_x_idx in range(4):
                coeff = u_coeffs[m_idx][local_x_idx]
                if coeff != 0:
                    global_u_idx = u_map[local_x_idx]
                    U[global_u_idx][col_offset] = coeff
            
            # Fill V
            for local_y_idx in range(4):
                coeff = v_coeffs[m_idx][local_y_idx]
                if coeff != 0:
                    global_v_idx = v_map[local_y_idx]
                    V[global_v_idx][col_offset] = coeff
            
            # Fill W
            for local_z_idx in range(4):
                coeff = w_coeffs[local_z_idx][m_idx]
                if coeff != 0:
                    global_w_idx = w_map[local_z_idx]
                    W[global_w_idx][col_offset] = coeff
            
            col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
