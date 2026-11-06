"""Matrix multiplication 6x6x6 via Schoolbook 3x3 blocks of Strassen 2x2.
Total rank: 9 * 7 = 63.
"""

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    start_time = time.time()
    
    # Block structure: 3x3 grid of 2x2 blocks
    # Outer indices I, J, K in {0, 1, 2}
    # Inner indices i, j, k in {0, 1}
    
    # Strassen 2x2 terms
    # Each term is (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: list of (local_row, local_col, coeff)
    # b_coeffs: list of (local_row, local_col, coeff)
    # c_coeffs: list of (local_row, local_col, coeff)
    
    strassen_2x2 = [
        # P1 = (A00+A11)(B00+B11) -> C00+=P1, C11+=P1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # P2 = (A10+A11)B00 -> C10+=P2, C11-=P2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # P3 = A00(B01-B11) -> C01+=P3, C11+=P3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # P4 = A11(B10-B00) -> C00+=P4, C10+=P4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # P5 = (A00+A01)B11 -> C00-=P5, C01+=P5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # P6 = (A10-A00)(B00+B01) -> C11+=P6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # P7 = (A01-A11)(B10+B11) -> C00+=P7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    rank = 9 * 7  # 63
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    a_terms, b_terms, c_terms = strassen_2x2[s]
                    
                    # Fill U (A contributions)
                    for (li, lj, coeff) in a_terms:
                        # Local (li, lj) in block A[I][K]
                        # Global row = 2*I + li, Global col = 2*K + lj
                        u_row = 2 * I + li
                        u_col = 2 * K + lj
                        u_idx = u_row * m + u_col
                        U[u_idx][col] += coeff
                        
                    # Fill V (B contributions)
                    for (li, lj, coeff) in b_terms:
                        # Local (li, lj) in block B[K][J]
                        # Global row = 2*K + li, Global col = 2*J + lj
                        v_row = 2 * K + li
                        v_col = 2 * J + lj
                        v_idx = v_row * p + v_col
                        V[v_idx][col] += coeff
                        
                    # Fill W (C contributions)
                    for (li, lj, coeff) in c_terms:
                        # Local (li, lj) in block C[I][J]
                        # Global row = 2*I + li, Global col = 2*J + lj
                        # W index = global_col * n + global_row
                        w_row = 2 * I + li
                        w_col = 2 * J + lj
                        w_idx = w_col * n + w_row
                        W[w_idx][col] += coeff
                        
                    col += 1
                    if time.time() - start_time > 100:
                        break
                if time.time() - start_time > 100:
                    break
            if time.time() - start_time > 100:
                break
        if time.time() - start_time > 100:
            break
            
    # Verification
    np.random.seed(42)
    A_test = np.random.randn(n, m).astype(float)
    B_test = np.random.randn(m, p).astype(float)
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_flat = U_arr[:, r]
        v_flat = V_arr[:, r]
        w_flat = W_arr[:, r]
        
        # s_A = dot(u_flat, A_flat)
        # s_B = dot(v_flat, B_flat)
        # C += s_A * s_B * w_flat.reshape(p, n).T
        
        s_A = np.dot(u_flat, A_test.flatten())
        s_B = np.dot(v_flat, B_test.flatten())
        
        if s_A == 0 or s_B == 0:
            continue
            
        # w_flat index is j*n + i. 
        # Reshape to (p, n) -> w_mat[j, i] = w_flat[j*n + i]
        # We need C[i, j] += s_A * s_B * w_mat[j, i]
        # So C += s_A * s_B * w_mat.T
        
        w_mat = w_flat.reshape(p, n)
        C_computed += s_A * s_B * w_mat.T
    
    max_err = np.max(np.abs(C_computed - C_expected))
    if max_err > 1e-9:
        # Fallback to schoolbook 6x6 if BCLR/Strassen logic is flawed (should not happen)
        # Rank 216
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_fb[i * m + k][idx] = 1
                    V_fb[k * p + j][idx] = 1
                    W_fb[j * n + i][idx] = 1
                    idx += 1
                    
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
