"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    # Strategy: Use Strassen's algorithm on 2x2x2 blocks.
    # Partition A (4x4) into 2x2 blocks: A_00, A_01, A_10, A_11
    # Partition B (4x6) into 2x3 blocks of 2x2: B_00, B_01, B_02 (top 2 rows)
    #                                          B_10, B_11, B_12 (bottom 2 rows)
    # C (4x6) = [C_00 C_01 C_02]
    #           [C_10 C_11 C_12]
    # where C_ik = A_i0 * B_0k + A_i1 * B_1k
    
    # Each 2x2 * 2x2 product uses 7 Strassen multiplications.
    # For each output block C_ik (i in {0,1}, k in {0,1,2}), we have two terms:
    #   A_i0 * B_0k and A_i1 * B_1k
    # Each term is a 2x2x2 tensor, decomposable with rank 7.
    # Total rank = 2 (i) * 3 (k) * 2 (j) * 7 (Strassen) = 84.
    
    # However, we can potentially do better by sharing computations.
    # But for now, let's implement the rank-84 solution correctly first.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for i in range(2):  # A row block index (0 or 1)
        for k in range(3):  # C col block index (0, 1, or 2)
            for j in range(2):  # Sum over j: A_ij * B_jk
                # Block A_ij: rows [2*i, 2*i+1], cols [2*j, 2*j+1]
                # Block B_jk: rows [2*j, 2*j+1], cols [2*k, 2*k+1]
                
                # A_ij elements:
                a_idx = 2*i * m + 2*j
                b_idx = 2*i * m + 2*j + 1
                c_idx = (2*i+1) * m + 2*j
                d_idx = (2*i+1) * m + 2*j + 1
                
                # B_jk elements:
                e_idx = 2*j * p + 2*k
                f_idx = 2*j * p + 2*k + 1
                g_idx = (2*j+1) * p + 2*k
                h_idx = (2*j+1) * p + 2*k + 1
                
                # C_ik block: rows [2*i, 2*i+1], cols [2*k, 2*k+1]
                # W index: W[col*n + row, r]
                c11_idx = (2*k) * n + 2*i
                c12_idx = (2*k+1) * n + 2*i
                c21_idx = (2*k) * n + 2*i + 1
                c22_idx = (2*k+1) * n + 2*i + 1
                
                # Strassen 7 multiplications
                # M1 = (a+d)(e+h)
                U[a_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M2 = (c+d)e
                U[c_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
                col += 1
                
                # M3 = a(f-h)
                U[a_idx][col] = 1
                V[f_idx][col] = 1
                V[h_idx][col] = -1
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M4 = d(g-e)
                U[d_idx][col] = 1
                V[g_idx][col] = 1
                V[e_idx][col] = -1
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
                col += 1
                
                # M5 = (a+b)h
                U[a_idx][col] = 1
                U[b_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
                col += 1
                
                # M6 = (c-a)(e+f)
                U[c_idx][col] = 1
                U[a_idx][col] = -1
                V[e_idx][col] = 1
                V[f_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M7 = (b-d)(g+h)
                U[b_idx][col] = 1
                U[d_idx][col] = -1
                V[g_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_r U[:,r] * V[:,r] * W[:,r] should equal the matmul tensor
    # Tensor T[i,j,k] = 1 if we're computing C[k,i] = sum_j A[i,j]*B[j,k]
    # In W convention: W[k*n+i, r]
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute decomposed tensor
    T_decomposed = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    
    # Verify
    if not np.allclose(T_decomposed, T_expected):
        # If verification fails, fall back to naive rank 96
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
