"""Rewritten program using Block Strassen decomposition for rank 28."""

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    # Rank is 4 blocks * 7 Strassen steps = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps 1-7
    # U coefficients for A (a,b,c,d) at indices 0,1,2,3
    # V coefficients for B block (e,f,g,h)
    # W coefficients for C block (C11, C12, C21, C22) at indices 0,1,2,3 relative to block start
    
    # Define the 7 Strassen steps
    # Step: (U_coeffs, V_coeffs, W_coeffs)
    # U: [a, b, c, d]
    # V: [e, f, g, h]
    # W: [C11, C12, C21, C22]
    
    steps = [
        # 1: (a+d)(e+h)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # 2: (c+d)e
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # 3: a(f-g)
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),
        # 4: d(h-g)
        ([0, 0, 0, 1], [0, 0, -1, 1], [1, 0, 1, 0]),
        # 5: (a+b)h
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # 6: (c+d)(f-h)
        ([0, 0, 1, 1], [0, 1, 0, -1], [0, 0, 0, 1]),
        # 7: (a-b)(e+g)
        ([1, -1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]),
    ]
    
    num_blocks = p // 2 # 4 blocks of 2 columns
    
    for b_idx in range(num_blocks):
        # Block b_idx uses columns 2*b_idx and 2*b_idx+1 of B
        # and rows 2*b_idx and 2*b_idx+1 of C (in the p dimension)
        col_offset = b_idx * 2
        
        for s_idx, (u_c, v_c, w_c) in enumerate(steps):
            r = b_idx * 7 + s_idx
            
            # Fill U
            # U indices: i*m + j. i in 0..1, j in 0..1
            # a: 0, b: 1, c: 2, d: 3
            U[0][r] += u_c[0]
            U[1][r] += u_c[1]
            U[2][r] += u_c[2]
            U[3][r] += u_c[3]
            
            # Fill V
            # V indices: j*p + k. j in 0..1, k in 0..7
            # e: B[0, 2*b_idx] -> 0*8 + 2*b_idx
            # f: B[0, 2*b_idx+1] -> 0*8 + 2*b_idx + 1
            # g: B[1, 2*b_idx] -> 1*8 + 2*b_idx
            # h: B[1, 2*b_idx+1] -> 1*8 + 2*b_idx + 1
            V[0 * p + col_offset][r] += v_c[0]
            V[0 * p + col_offset + 1][r] += v_c[1]
            V[1 * p + col_offset][r] += v_c[2]
            V[1 * p + col_offset + 1][r] += v_c[3]
            
            # Fill W
            # W indices: k*n + i. k in 0..7, i in 0..1
            # C11: C[0, 2*b_idx] -> (2*b_idx)*2 + 0
            # C12: C[0, 2*b_idx+1] -> (2*b_idx+1)*2 + 0
            # C21: C[1, 2*b_idx] -> (2*b_idx)*2 + 1
            # C22: C[1, 2*b_idx+1] -> (2*b_idx+1)*2 + 1
            W[col_offset * n + 0][r] += w_c[0]
            W[(col_offset + 1) * n + 0][r] += w_c[1]
            W[col_offset * n + 1][r] += w_c[2]
            W[(col_offset + 1) * n + 1][r] += w_c[3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
