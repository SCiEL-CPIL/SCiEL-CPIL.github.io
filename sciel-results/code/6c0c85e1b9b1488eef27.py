import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a rank-49 decomposition for the 4x4x4 part, duplicated for the two halves of the 8 columns.
    Total rank = 98.
    """
    
    # 1. Construct rank-49 factors for 4x4x4 using double Strassen
    
    # Inner Strassen factors for 2x2x2
    # U_in, V_in, W_in are 4x7
    U_in = np.zeros((4, 7), dtype=int)
    V_in = np.zeros((4, 7), dtype=int)
    W_in = np.zeros((4, 7), dtype=int)
    
    # Strassen formulas for 2x2x2
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Indices for 2x2: 0:00, 1:01, 2:10, 3:11
    
    # t=0
    # A' = A00 + A11 -> coeffs [1, 0, 0, 1]
    # B' = B00 + B11 -> coeffs [1, 0, 0, 1]
    U_in[:, 0] = [1, 0, 0, 1] # Wait, U_in[i, t] is coeff of A_i in M_t?
    # No, U_in is the factor matrix.
    # M_t = sum_s U_in[i, s] V_in[j, s] ...
    # The standard Strassen factors are:
    # U_in[i, t] is the coefficient of A_i in the A-part of the t-th multiplication.
    
    # Let's populate U_in, V_in, W_in directly from the definitions.
    # M_t = A'_t B'_t
    # A'_t = sum_i u_{t,i} A_i
    # B'_t = sum_j v_{t,j} B_j
    
    # t=0: A'=A00+A11, B'=B00+B11
    U_in[0, 0] = 1; U_in[3, 0] = 1
    V_in[0, 0] = 1; V_in[3, 0] = 1
    # W_in for t=0: C00+=1, C01+=0, C10+=0, C11+=1?
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    W_in[0, 0] = 1 # C00
    W_in[3, 0] = 1 # C11
    
    # t=1: A'=A10+A11, B'=B00
    U_in[2, 1] = 1; U_in[3, 1] = 1
    V_in[0, 1] = 1
    W_in[2, 1] = 1 # C10
    W_in[3, 1] = -1 # C11
    
    # t=2: A'=A00, B'=B01-B11
    U_in[0, 2] = 1
    V_in[1, 2] = 1; V_in[3, 2] = -1
    W_in[1, 2] = 1 # C01
    W_in[3, 2] = 1 # C11
    
    # t=3: A'=A11, B'=B10-B00
    U_in[3, 3] = 1
    V_in[2, 3] = 1; V_in[0, 3] = -1
    W_in[0, 3] = 1 # C00
    W_in[2, 3] = 1 # C10
    
    # t=4: A'=A00+A01, B'=B11
    U_in[0, 4] = 1; U_in[1, 4] = 1
    V_in[3, 4] = 1
    W_in[0, 4] = -1 # C00
    W_in[1, 4] = 1 # C01
    
    # t=5: A'=A10-A00, B'=B00+B01
    U_in[2, 5] = 1; U_in[0, 5] = -1
    V_in[0, 5] = 1; V_in[1, 5] = 1
    W_in[3, 5] = 1 # C11
    
    # t=6: A'=A01-A11, B'=B10+B11
    U_in[1, 6] = 1; U_in[3, 6] = -1
    V_in[2, 6] = 1; V_in[3, 6] = 1
    W_in[0, 6] = 1 # C00
    W_in[1, 6] = 0 # C01? No, C01 = M2+M4. M6 is not in C01.
    # Check C formulas again.
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # So W_in[1, 6] should be 0.
    # W_in[0, 6] = 1 (from C00)
    # W_in[2, 6] = 0
    # W_in[3, 6] = 0
    
    W_in[0, 6] = 1
    W_in[1, 6] = 0
    W_in[2, 6] = 0
    W_in[3, 6] = 0
    
    # Now construct 4x4x4 factors
    U_444 = np.zeros((16, 49), dtype=int)
    V_444 = np.zeros((16, 49), dtype=int)
    W_444 = np.zeros((16, 49), dtype=int)
    
    # Outer Strassen coefficients
    # K_A[t][i][j] where t in 0..6, i in 0..1, j in 0..1
    # Block indices: 00, 01, 10, 11
    
    K_A = [
        [[1, 0], [0, 1]], # t=0: A00 + A11
        [[0, 0], [1, 1]], # t=1: A10 + A11
        [[1, 0], [0, 0]], # t=2: A00
        [[0, 0], [0, 1]], # t=3: A11
        [[1, 1], [0, 0]], # t=4: A00 + A01
        [[-1, 0], [1, 0]],# t=5: A10 - A00
        [[0, 1], [0, -1]] # t=6: A01 - A11
    ]
    
    K_B = [
        [[1, 0], [0, 1]], # t=0: B00 + B11
        [[1, 0], [0, 0]], # t=1: B00
        [[0, 1], [0, -1]],# t=2: B01 - B11
        [[-1, 0], [1, 0]],# t=3: B10 - B00
        [[0, 0], [0, 1]], # t=4: B11
        [[1, 1], [0, 0]], # t=5: B00 + B01
        [[0, 0], [1, 1]]  # t=6: B10 + B11
    ]
    
    for t in range(7):
        for s in range(7):
            g = t * 7 + s
            # For each block (bi, bj) in 0..1
            for bi in range(2):
                for bj in range(2):
                    # Coefficients for this block in A'_t and B'_t
                    coeff_A = K_A[t][bi][bj]
                    coeff_B = K_B[t][bi][bj]
                    
                    if coeff_A != 0:
                        # Map block (bi, bj) to global indices
                        # Global row = bi*2 + r, Global col = bj*2 + c
                        # Inner index = r*2 + c
                        for r in range(2):
                            for c in range(2):
                                global_idx_A = (bi*2 + r) * 4 + (bj*2 + c)
                                inner_idx_A = r * 2 + c
                                U_444[global_idx_A, g] += coeff_A * U_in[inner_idx_A, s]
                                
                    if coeff_B != 0:
                        for r in range(2):
                            for c in range(2):
                                global_idx_B = (bi*2 + r) * 4 + (bj*2 + c)
                                inner_idx_B = r * 2 + c
                                V_444[global_idx_B, g] += coeff_B * V_in[inner_idx_B, s]
                                
                    # W_444 is independent of the block structure in the same way?
                    # W_444 maps the result M_t (which is a 2x2 matrix) to C.
                    # The result M_t is a 2x2 matrix.
                    # Its elements correspond to C elements.
                    # The inner W_in[inner_idx_C, s] gives the coefficient for the element of M_t.
                    # The element of M_t at (r, c) corresponds to C element at (bi*2+r, bj*2+c)?
                    # No, M_t is the result of multiplying A'_t and B'_t.
                    # A'_t and B'_t are 2x2 matrices formed from blocks.
                    # The product M_t is a 2x2 matrix.
                    # The (r, c) element of M_t contributes to the (r, c) element of C?
                    # No, the blocks of C are formed by linear combinations of the M_t blocks.
                    # C_{block(i,j)} = sum_t w_{t, i, j} M_t
                    # Where M_t is a 2x2 matrix.
                    # So the (r, c) element of M_t contributes to C_{block(i,j)}[r, c].
                    # Global index for C: (i*2+r)*4 + (j*2+c).
                    
                    # We need the outer Strassen coefficients for C.
                    # C00 = M0 + M3 - M4 + M6
                    # C01 = M2 + M4
                    # C10 = M1 + M3
                    # C11 = M0 - M1 + M2 + M5
                    
                    # So for each t, we add to specific blocks of C.
                    # Let's define K_C[t][i][j]
                    
    # Define K_C
    K_C = [
        [[1, 0], [0, 1]], # t=0: M0 -> C00, C11
        [[0, 0], [1, -1]],# t=1: M1 -> C10, -C11
        [[0, 1], [0, 1]], # t=2: M2 -> C01, C11
        [[1, 0], [1, 0]], # t=3: M3 -> C00, C10
        [[-1, 1], [0, 0]],# t=4: M4 -> -C00, C01
        [[0, 0], [0, 1]], # t=5: M5 -> C11
        [[1, 0], [0, 0]]  # t=6: M6 -> C00
    ]
    
    # Redo W_444 construction
    W_444 = np.zeros((16, 49), dtype=int)
    for t in range(7):
        for s in range(7):
            g = t * 7 + s
            for ci in range(2):
                for cj in range(2):
                    coeff_C = K_C[t][ci][cj]
                    if coeff_C != 0:
                        for r in range(2):
                            for c in range(2):
                                global_idx_C = (ci*2 + r) * 4 + (cj*2 + c)
                                inner_idx_C = r * 2 + c
                                W_444[global_idx_C, g] += coeff_C * W_in[inner_idx_C, s]
    
    # 2. Assemble 4x4x8 factors
    rank = 98
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((32, rank), dtype=int)
    W = np.zeros((32, rank), dtype=int)
    
    # First half (cols 0-3 of B, rows 0-3 of C)
    U[:, 0:49] = U_444
    V[0:16, 0:49] = V_444
    W[0:16, 0:49] = W_444
    
    # Second half (cols 4-7 of B, rows 4-7 of C)
    U[:, 49:98] = U_444
    
    # V indices: j * 8 + k
    # For k in 0..3 (local), global k is 4+k
    # V_444 index is j * 4 + k_local
    for j in range(4):
        for k_local in range(4):
            v_idx_444 = j * 4 + k_local
            v_idx_448 = j * 8 + (4 + k_local)
            V[v_idx_448, 49:98] = V_444[v_idx_444, :]
            
    # W indices: k * 4 + i
    # For k in 0..3 (local), global k is 4+k
    # W_444 index is k_local * 4 + i
    for k_local in range(4):
        for i in range(4):
            w_idx_444 = k_local * 4 + i
            w_idx_448 = (4 + k_local) * 4 + i
            W[w_idx_448, 49:98] = W_444[w_idx_444, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
