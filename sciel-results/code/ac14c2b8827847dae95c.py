import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4x4 multiplication (columns 0-3 of B)
    # Rank 49 using Strassen
    # Part 2: 4x4x1 multiplication (column 4 of B)
    # Rank 16 using Schoolbook
    
    total_rank = 49 + 16
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 Strassen ---
    # Define 7 Strassen terms for 2x2x2
    # Each term is (U_2x2, V_2x2, W_2x2)
    # U, V, W are 2x2 matrices
    
    terms_2x2 = [
        # M1
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        # M2
        ([[0, 0], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [1, -1]]),
        # M3
        ([[1, 0], [0, 0]], [[0, 1], [0, -1]], [[0, 1], [0, 1]]),
        # M4
        ([[0, 0], [0, 1]], [[0, 0], [1, -1]], [[1, 0], [1, 0]]),
        # M5
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[-1, 1], [0, 0]]),
        # M6
        ([[0, 0], [1, -1]], [[1, 1], [0, 0]], [[0, 0], [0, 1]]),
        # M7
        ([[0, 1], [0, -1]], [[0, 0], [1, 1]], [[1, 0], [0, 0]])
    ]
    
    # Generate 49 terms for 4x4x4
    col_offset = 0
    for o in range(7):
        U_o, V_o, W_o = terms_2x2[o]
        for i in range(7):
            U_i, V_i, W_i = terms_2x2[i]
            
            # Kronecker product for U, V, W
            # U_4x4 = U_o (2x2) kron U_i (2x2) -> 4x4
            # Indices: (p, r) where p in {0,1} is block row, r in {0,1} is sub-row
            # U_4x4[p*2+r, q*2+s] = U_o[p,q] * U_i[r,s]
            
            for p in range(2):
                for q in range(2):
                    u_o_val = U_o[p][q]
                    if u_o_val == 0:
                        continue
                    for r in range(2):
                        for s in range(2):
                            u_i_val = U_i[r][s]
                            if u_i_val == 0:
                                continue
                            
                            val = u_o_val * u_i_val
                            if val == 0:
                                continue
                            
                            # Map 4x4 index to flat index
                            # U index: i * m + j = i * 4 + j
                            # Here i=p, j=q? No.
                            # A is 4x4. Block A_pq is 2x2.
                            # Element A[p*2+r, q*2+s].
                            # Flat index: (p*2+r)*4 + (q*2+s)
                            u_idx = (p * 2 + r) * 4 + (q * 2 + s)
                            U[u_idx][col_offset] += val
                            
                            # V index: j * p + k = j * 4 + k
                            # B is 4x4 (first 4 cols). Block B_pq is 2x2.
                            # Element B[p*2+r, q*2+s].
                            # Flat index: (p*2+r)*4 + (q*2+s)
                            v_idx = (p * 2 + r) * 4 + (q * 2 + s)
                            # Note: V_o and V_i correspond to B blocks.
                            # V_o[p][q] is coeff for B_pq.
                            # V_i[r][s] is coeff for sub-block.
                            # So V_val = V_o[p][q] * V_i[r][s]
                            v_val = V_o[p][q] * V_i[r][s]
                            if v_val != 0:
                                V[v_idx][col_offset] += v_val
                                
                            # W index: k * n + i = k * 4 + i
                            # C is 4x4. Block C_pq is 2x2.
                            # Element C[p*2+r, q*2+s].
                            # Flat index: (p*2+r)*4 + (q*2+s)
                            w_idx = (p * 2 + r) * 4 + (q * 2 + s)
                            w_val = W_o[p][q] * W_i[r][s]
                            if w_val != 0:
                                W[w_idx][col_offset] += w_val
            
            col_offset += 1
            
    # --- Part 2: 4x4x1 Schoolbook ---
    # A is 4x4, B_col is 4x1.
    # C_col is 4x1.
    # C[i, 0] = sum_j A[i, j] * B[j, 0]
    # Rank 16.
    for i in range(4):
        for j in range(4):
            # U index: i * 4 + j
            U[i * 4 + j][col_offset] = 1
            # V index: j * 5 + 4 (since k=4 is the 5th column)
            V[j * 5 + 4][col_offset] = 1
            # W index: 4 * 4 + i (since k=4)
            W[4 * 4 + i][col_offset] = 1
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    # Simple verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Test with random matrices
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 5)
    C_true = A @ B
    
    # Compute C using tensor decomposition
    # C[k, i] = sum_c W[k*4+i, c] * (sum_j U[i*4+j, c] * A[i, j]) * (sum_j V[j*5+k, c] * B[j, k])
    # This is O(N^3 * Rank).
    # Better: 
    # M_A = U.T @ A_flat? No.
    # U is 16x65. A is 4x4.
    # Let's just do the direct sum for verification.
    
    C_comp = np.zeros((5, 4)) # W index is k*4+i, so C is 5x4? No, C is 4x5.
    # W[k*n+i, c]. n=4.
    # C[i, k] = sum_c W[k*4+i, c] * U_A[i, c] * V_B[k, c]
    # U_A[i, c] = sum_j U[i*4+j, c] * A[i, j]
    # V_B[k, c] = sum_j V[j*5+k, c] * B[j, k]
    
    U_A = np.zeros((4, 65))
    V_B = np.zeros((5, 65))
    for i in range(4):
        for c in range(65):
            U_A[i, c] = sum(U[i*4+j, c] * A[i, j] for j in range(4))
    for k in range(5):
        for c in range(65):
            V_B[k, c] = sum(V[j*5+k, c] * B[j, k] for j in range(4))
            
    for i in range(4):
        for k in range(5):
            s = 0
            for c in range(65):
                s += W[k*4+i, c] * U_A[i, c] * V_B[k, c]
            C_comp[k, i] = s
            
    C_comp = C_comp.T # Now 4x5
    
    if np.allclose(C_true, C_comp):
        print("Verification passed")
    else:
        print("Verification failed")
        print(C_true)
        print(C_comp)
