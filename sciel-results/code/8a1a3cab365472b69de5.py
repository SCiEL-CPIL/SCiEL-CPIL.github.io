import time
import numpy as np

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    def build_rank25():
        """
        Build a rank-25 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
        
        Strategy:
        - Strassen rank-7 for columns (0,1)
        - Strassen rank-7 for columns (2,3) 
        - Strassen rank-7 for columns (4,5)
        - Schoolbook rank-4 for column 6
        Total rank: 7 + 7 + 7 + 4 = 25
        """
        rank = 25
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        def add_strassen_block(col_start, k_start):
            """
            Add Strassen rank-7 terms for a 2x2 block of B (columns k_start, k_start+1).
            
            The 7 Strassen products are:
            M1 = (A00+A11)*(B0,ks + B1,ks+1)
            M2 = (A10+A11)*B0,ks
            M3 = A00*(B0,ks+1 - B1,ks+1)
            M4 = A11*(B1,ks - B0,ks)
            M5 = (A00+A01)*B1,ks+1
            M6 = (A10-A00)*(B0,ks+B0,ks+1)
            M7 = (A01-A11)*(B1,ks+B1,ks+1)
            
            Result: C0,ks = M1 - M5 + M7
                    C1,ks = M2 + M4 - M7
                    C0,ks+1 = M3 + M5 - M1
                    C1,ks+1 = M2 + M4 + M6 - M1
            """
            # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
            U[0, col_start+0] += 1.0
            U[3, col_start+0] += 1.0
            V[7*0 + k_start, col_start+0] += 1.0
            V[7*1 + (k_start+1), col_start+0] += 1.0
            W[2*k_start + 0, col_start+0] += 1.0      # C0,ks += M1
            W[2*(k_start+1) + 1, col_start+0] += 1.0  # C1,ks+1 += M1

            # M2 = (A10 + A11) * B0,ks
            U[2, col_start+1] += 1.0
            U[3, col_start+1] += 1.0
            V[7*0 + k_start, col_start+1] += 1.0
            W[2*k_start + 1, col_start+1] += 1.0      # C1,ks += M2
            W[2*(k_start+1) + 1, col_start+1] -= 1.0  # C1,ks+1 -= M2

            # M3 = A00 * (B0,ks+1 - B1,ks+1)
            U[0, col_start+2] += 1.0
            V[7*0 + (k_start+1), col_start+2] += 1.0
            V[7*1 + (k_start+1), col_start+2] -= 1.0
            W[2*(k_start+1) + 0, col_start+2] += 1.0  # C0,ks+1 += M3
            W[2*(k_start+1) + 1, col_start+2] += 1.0  # C1,ks+1 += M3

            # M4 = A11 * (B1,ks - B0,ks)
            U[3, col_start+3] += 1.0
            V[7*1 + k_start, col_start+3] += 1.0
            V[7*0 + k_start, col_start+3] -= 1.0
            W[2*k_start + 0, col_start+3] += 1.0      # C0,ks += M4
            W[2*k_start + 1, col_start+3] += 1.0      # C1,ks += M4

            # M5 = (A00 + A01) * B1,ks+1
            U[0, col_start+4] += 1.0
            U[1, col_start+4] += 1.0
            V[7*1 + (k_start+1), col_start+4] += 1.0
            W[2*k_start + 0, col_start+4] -= 1.0      # C0,ks -= M5
            W[2*(k_start+1) + 0, col_start+4] += 1.0  # C0,ks+1 += M5

            # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
            U[2, col_start+5] += 1.0
            U[0, col_start+5] -= 1.0
            V[7*0 + k_start, col_start+5] += 1.0
            V[7*0 + (k_start+1), col_start+5] += 1.0
            W[2*(k_start+1) + 1, col_start+5] += 1.0  # C1,ks+1 += M6

            # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
            U[1, col_start+6] += 1.0
            U[3, col_start+6] -= 1.0
            V[7*1 + k_start, col_start+6] += 1.0
            V[7*1 + (k_start+1), col_start+6] += 1.0
            W[2*k_start + 0, col_start+6] += 1.0      # C0,ks += M7

        # Apply Strassen to pairs (0,1), (2,3), (4,5)
        add_strassen_block(0, 0)
        add_strassen_block(7, 2)
        add_strassen_block(14, 4)
        
        # Apply schoolbook to column 6
        col_start = 21
        k = 6
        
        # C0,6 = A00*B0,6 + A01*B1,6
        U[0, col_start+0] += 1.0
        V[7*0 + k, col_start+0] += 1.0
        W[2*k + 0, col_start+0] += 1.0
        
        U[1, col_start+1] += 1.0
        V[7*1 + k, col_start+1] += 1.0
        W[2*k + 0, col_start+1] += 1.0
        
        # C1,6 = A10*B0,6 + A11*B1,6
        U[2, col_start+2] += 1.0
        V[7*0 + k, col_start+2] += 1.0
        W[2*k + 1, col_start+2] += 1.0
        
        U[3, col_start+3] += 1.0
        V[7*1 + k, col_start+3] += 1.0
        W[2*k + 1, col_start+3] += 1.0

        return {"U": U, "V": V, "W": W}

    def verify_decomp_fast(U, V, W, n, m, p):
        """
        Fast verification using vectorized tensor contraction.
        Checks that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for all valid (i,j,k) where the tensor entry is 1.
        """
        rank = U.shape[1]
        
        # Vectorized check of all 28 tensor entries
        # For each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    # Vectorized: sum_r U[a,r]*V[b,r]*W[c,r] = sum_r (U[a,r]*V[b,r])*W[c,r]
                    s = np.dot(U[a_idx, :] * V[b_idx, :], W[c_idx, :])
                    if abs(s - 1.0) > 1e-10:
                        return False
        
        # Quick random sanity check with 5 trials
        import random
        for _ in range(5):
            A = np.array([[random.randint(-10, 10) for _ in range(m)] for _ in range(n)], dtype=float)
            B = np.array([[random.randint(-10, 10) for _ in range(p)] for _ in range(m)], dtype=float)
            C_true = A @ B
            
            # Compute C_calc using the decomposition
            C_calc = np.zeros((n, p))
            for r in range(rank):
                u_val = np.dot(U[:, r], A.flatten())
                v_val = np.dot(V[:, r], B.flatten())
                prod = u_val * v_val
                # W[:, r] has shape (p*n,), index k*n+i corresponds to C[i,k]
                w_vec = W[:, r].reshape(p, n).T  # Now shape (n, p), w_vec[i,k] = W[k*n+i, r]
                C_calc += w_vec * prod
            
            if not np.allclose(C_true, C_calc, atol=1e-9):
                return False
        
        return True

    # Build the rank-25 decomposition
    result = build_rank25()
    
    # Verify with fast method
    if verify_decomp_fast(result["U"], result["V"], result["W"], n, m, p):
        return {"U": result["U"].tolist(), "V": result["V"].tolist(), "W": result["W"].tolist()}

    # Fallback: schoolbook rank 28
    rank_sb = 28
    U_sb = np.zeros((n*m, rank_sb), dtype=float)
    V_sb = np.zeros((m*p, rank_sb), dtype=float)
    W_sb = np.zeros((p*n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
