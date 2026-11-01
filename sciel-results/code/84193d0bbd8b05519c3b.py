"""Matrix multiplication 2x4x8 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16

    # Build the true matrix multiplication tensor T of shape (A_dim, B_dim, C_dim)
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # i.e., a = i*m + j, b = j*p + k, c = k*n + i
    # So for each (i, j, k): a = i*m+j, b = j*p+k, c = k*n+i
    # Equivalently: the index j must satisfy a mod m == b // p == (b - k*p)//p... 
    # More directly: T is nonzero at (a,b,c) iff there exists i,j,k with
    #   a = i*m + j, b = j*p + k, c = k*n + i
    # which means j = a % m, i = a // m, k = b % p, and b // p == j, and c // n == k, and c % n == i.
    
    # Actually let's just enumerate:
    T = np.zeros((A_dim, B_dim, C_dim), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # Strategy: try to find a low-rank decomposition using Strassen on 2x2 blocks
    # The rank-56 solution treats the problem as 8 independent 2x2 multiplications.
    # Let's try a different structural approach:
    # 
    # Key insight: A is 2x4, B is 4x8. We can write:
    #   C = A * B where C is 2x8
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split B into two 4x4 blocks: B = [B_left, B_right]
    # Then C = [A*B_left, A*B_right]
    # Each A*B is a 2x4 * 4x4 = 2x4 product.
    #
    # For a 2x4 * 4x4 product, we can use the following approach:
    # View A as a 2x4 matrix. Split into A = [A1, A2] where A1, A2 are 2x2.
    # View B (4x4) as [[B11, B12], [B21, B22]] where each is 2x2.
    # Then A*B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # Each 2x2*2x2 product can be done with rank 7 (Strassen).
    # So 2x4 * 4x4 needs rank 4*7 = 28.
    # Two such blocks: rank 56.
    #
    # Alternative: Can we do better by exploiting the shared A?
    # A*B_left and A*B_right share the same A.
    # 
    # Let's try: use Strassen on the 4x8 B matrix directly.
    # Split B into four 2x4 blocks? No, B is 4x8.
    # Split B into two 2x8 blocks (top half and bottom half of rows).
    # B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 2x8.
    # Then C[i,k] = sum_{j=0}^{1} A[i,j]*B_top[j,k] + sum_{j=2}^{3} A[i,j]*B_bottom[j,k]
    #            = A_left[i,:] * B_top[:,k] + A_right[i,:] * B_bottom[:,k]
    # where A_left is A[:,0:2] (2x2), A_right is A[:,2:4] (2x2).
    #
    # So C = A_left * B_top + A_right * B_bottom
    # where A_left, A_right are 2x2, B_top, B_bottom are 2x8.
    #
    # Now A_left * B_top is a 2x2 * 2x8 product = (2x2)*(2x8) = 2x8.
    # This can be viewed as two 2x2 * 2x4 products, each needing rank 14 (2*7).
    # So A_left * B_top needs rank 28.
    # Similarly A_right * B_bottom needs rank 28.
    # Total: 56. Same as before.
    #
    # Can we do better? Let's think about the tensor structure more carefully.
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1.
    # 
    # Let's try a different decomposition approach: use the fact that n=2.
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # C[1,k] = sum_j A[1,j] * B[j,k]
    #
    # For each k, C[:,k] = A^T * B[:,k] where A^T is 4x2 and B[:,k] is 4x1.
    # This is a 4x2 * 2x1 product... no wait, C[:,k] is 2x1 = A * B[:,k] where A is 2x4, B[:,k] is 4x1.
    #
    # Let's try yet another approach: use the Laderman or other sub-quadratic methods.
    # Actually, for 2x4 * 4x8, the known optimal rank might be less than 56.
    #
    # Let me try to construct a rank-28 decomposition by being smarter about sharing.
    # 
    # Idea: Instead of treating the two halves of B independently, use a single Strassen-like
    # decomposition that processes the full 4x8 B with the 2x4 A.
    #
    # Actually, let's try: view the problem as computing C = A * B where A is 2x4, B is 4x8.
    # We can pad A to 4x4 by adding zero rows, and B is already 4x8.
    # Then C_padded = A_padded * B is 4x8, and we only need the top 2 rows.
    # A_padded * B can be done with rank 4*8 = 32 (schoolbook) or with Strassen.
    # But we only need half the output, so maybe rank 28?
    #
    # Let's try a concrete construction with rank 28:
    # Use Strassen on 2x2 blocks but share computations.
    #
    # Actually, let me try a completely different approach: use the fact that the tensor
    # T(a,b,c) = 1 iff a%m == b//p and b%p == c//n and c%n == a//m.
    #
    # Let's try to find a rank-r decomposition by solving a system of equations.
    # For rank r, we need U (8xr), V (32xr), W (16xr) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    #
    # This is a system of 8*32*16 = 4096 equations in 8r + 32r + 16r = 56r unknowns.
    # For r=14: 784 unknowns, 4096 equations. Likely overdetermined.
    # For r=28: 1568 unknowns, 4096 equations. Still overdetermined.
    # For r=56: 3136 unknowns, 4096 equations. Underdetermined.
    #
    # Let's try to construct a rank-28 solution using a structured approach.
    #
    # Approach: Use Strassen's algorithm on the 4x4 * 4x4 sub-problems, but exploit
    # the fact that A is only 2x4 (half of a 4x4).
    #
    # Let me try: split the 4x8 B into four 2x2 blocks in a 2x4 grid:
    # B = [[B00, B01, B02, B03],
    #      [B10, B11, B12, B13],
    #      [B20, B21, B22, B23],
    #      [B30, B31, B32, B33]]
    #
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13]]
    #
    # C[0,k] = a00*B[0,k] + a01*B[1,k] + a02*B[2,k] + a03*B[3,k]
    # C[1,k] = a10*B[0,k] + a11*B[1,k] + a12*B[2,k] + a13*B[3,k]
    #
    # Let's define intermediate products using Strassen-like combinations:
    # M1 = (a00+a13)*(B[0,0]+B[3,0])  -- not quite, let me think more carefully.
    #
    # Actually, let's just try the rank-56 solution but verify it works, and also
    # try to see if we can get rank 28 by a different block decomposition.
    #
    # For now, let's implement the rank-56 Strassen solution carefully.
    
    # Strassen factors for 2x2x2
    u_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=int)
    
    v_s = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ], dtype=int)
    
    w_s = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=int)
    
    # Verify Strassen works for 2x2
    T2 = np.zeros((4, 4, 4), dtype=int)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2[i*2+j, j*2+k, k*2+i] = 1
    
    T2_rec = np.einsum('ar,br,cr->abc', u_s.astype(float), v_s.astype(float), w_s.astype(float))
    if not np.allclose(T2, T2_rec, atol=1e-10):
        # Fallback to schoolbook
        rank = 64
        U_fb = np.zeros((A_dim, rank), dtype=int)
        V_fb = np.zeros((B_dim, rank), dtype=int)
        W_fb = np.zeros((C_dim, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Build rank-56 decomposition
    rank = 56
    U = np.zeros((A_dim, rank), dtype=int)
    V = np.zeros((B_dim, rank), dtype=int)
    W = np.zeros((C_dim, rank), dtype=int)
    
    # A is 2x4: indices
    # Row 0: 0,1,2,3
    # Row 1: 4,5,6,7
    # A_L (cols 0,1): global [0,1,4,5] -> local [0,1,2,3]
    # A_R (cols 2,3): global [2,3,6,7] -> local [0,1,2,3]
    
    a_L = [0, 1, 4, 5]
    a_R = [2, 3, 6, 7]
    
    # B is 4x8: indices
    # Row 0:  0-7
    # Row 1:  8-15
    # Row 2: 16-23
    # Row 3: 24-31
    
    # For B columns 0-3 (first 4 columns):
    # B_TL (rows 0,1, cols 0,1): [0,1,8,9]
    # B_TR (rows 0,1, cols 2,3): [2,3,10,11]
    # B_BL (rows 2,3, cols 0,1): [16,17,24,25]
    # B_BR (rows 2,3, cols 2,3): [18,19,26,27]
    
    # For B columns 4-7 (last 4 columns):
    # B_TL2 (rows 0,1, cols 4,5): [4,5,12,13]
    # B_TR2 (rows 0,1, cols 6,7): [6,7,14,15]
    # B_BL2 (rows 2,3, cols 4,5): [20,21,28,29]
    # B_BR2 (rows 2,3, cols 6,7): [22,23,30,31]
    
    # C is 2x8: indices
    # Row 0: 0-7
    # Row 1: 8-15
    
    # C blocks (2x2 each):
    # C_01 (cols 0,1): [0,1,8,9]
    # C_23 (cols 2,3): [2,3,10,11]
    # C_45 (cols 4,5): [4,5,12,13]
    # C_67 (cols 6,7): [6,7,14,15]
    
    c_01 = [0, 1, 8, 9]
    c_23 = [2, 3, 10, 11]
    c_45 = [4, 5, 12, 13]
    c_67 = [6, 7, 14, 15]
    
    # 8 Strassen blocks, each of rank 7
    blocks = [
        # (A_indices, B_indices, C_indices)
        (a_L, [0, 1, 8, 9], c_01),       # A_L * B_TL -> C_01
        (a_R, [16, 17, 24, 25], c_01),    # A_R * B_BL -> C_01
        (a_L, [2, 3, 10, 11], c_23),      # A_L * B_TR -> C_23
        (a_R, [18, 19, 26, 27], c_23),    # A_R * B_BR -> C_23
        (a_L, [4, 5, 12, 13], c_45),      # A_L * B_TL2 -> C_45
        (a_R, [20, 21, 28, 29], c_45),    # A_R * B_BL2 -> C_45
        (a_L, [6, 7, 14, 15], c_67),      # A_L * B_TR2 -> C_67
        (a_R, [22, 23, 30, 31], c_67),    # A_R * B_BR2 -> C_67
    ]
    
    col = 0
    for a_idx, b_idx, c_idx in blocks:
        for s in range(7):
            for li, gi in enumerate(a_idx):
                U[gi, col] = u_s[s, li]
            for lj, gj in enumerate(b_idx):
                V[gj, col] = v_s[s, lj]
            for lk, gk in enumerate(c_idx):
                W[gk, col] = w_s[s, lk]
            col += 1
    
    # Verify
    T_rec = np.einsum('ar,br,cr->abc', U.astype(float), V.astype(float), W.astype(float))
    if not np.allclose(T, T_rec, atol=1e-10):
        # Fallback
        rank_fb = 64
        U_fb = np.zeros((A_dim, rank_fb), dtype=int)
        V_fb = np.zeros((B_dim, rank_fb), dtype=int)
        W_fb = np.zeros((C_dim, rank_fb), dtype=int)
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col_fb] = 1
                    V_fb[j*p+k, col_fb] = 1
                    W_fb[k*n+i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
