import numpy as np
import time

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Try to construct Rank 38 using Laderman for 3x3x3 blocks
    try:
        if time.time() > deadline:
            return get_schoolbook()
        
        # Laderman Rank 19 for 3x3x3
        # We define U and V for the 19 products.
        # U shape (9, 19), V shape (9, 19)
        U_19 = np.zeros((9, 19))
        V_19 = np.zeros((9, 19))
        
        # Product 1: (a00+a10)(b00+b10)
        U_19[0, 0] = 1; U_19[3, 0] = 1
        V_19[0, 0] = 1; V_19[3, 0] = 1
        
        # Product 2: (a01+a11)(b01+b11)
        U_19[1, 1] = 1; U_19[4, 1] = 1
        V_19[1, 1] = 1; V_19[4, 1] = 1
        
        # Product 3: (a02+a12)(b02+b12)
        U_19[2, 2] = 1; U_19[5, 2] = 1
        V_19[2, 2] = 1; V_19[5, 2] = 1
        
        # Product 4: (a00+a20)(b00+b20)
        U_19[0, 3] = 1; U_19[6, 3] = 1
        V_19[0, 3] = 1; V_19[6, 3] = 1
        
        # Product 5: (a01+a21)(b01+b21)
        U_19[1, 4] = 1; U_19[7, 4] = 1
        V_19[1, 4] = 1; V_19[7, 4] = 1
        
        # Product 6: (a02+a22)(b02+b22)
        U_19[2, 5] = 1; U_19[8, 5] = 1
        V_19[2, 5] = 1; V_19[8, 5] = 1
        
        # Product 7: (a10+a20)(b10+b20)
        U_19[3, 6] = 1; U_19[6, 6] = 1
        V_19[3, 6] = 1; V_19[6, 6] = 1
        
        # Product 8: (a11+a21)(b11+b21)
        U_19[4, 7] = 1; U_19[7, 7] = 1
        V_19[4, 7] = 1; V_19[7, 7] = 1
        
        # Product 9: (a12+a22)(b12+b22)
        U_19[5, 8] = 1; U_19[8, 8] = 1
        V_19[5, 8] = 1; V_19[8, 8] = 1
        
        # Product 10: (a00-a10+a20)(b00-b10+b20)
        U_19[0, 9] = 1; U_19[3, 9] = -1; U_19[6, 9] = 1
        V_19[0, 9] = 1; V_19[3, 9] = -1; V_19[6, 9] = 1
        
        # Product 11: (a01-a11+a21)(b01-b11+b21)
        U_19[1, 10] = 1; U_19[4, 10] = -1; U_19[7, 10] = 1
        V_19[1, 10] = 1; V_19[4, 10] = -1; V_19[7, 10] = 1
        
        # Product 12: (a02-a12+a22)(b02-b12+b22)
        U_19[2, 11] = 1; U_19[5, 11] = -1; U_19[8, 11] = 1
        V_19[2, 11] = 1; V_19[5, 11] = -1; V_19[8, 11] = 1
        
        # Product 13: (a00+a01+a02)(b00+b01+b02)
        U_19[0, 12] = 1; U_19[1, 12] = 1; U_19[2, 12] = 1
        V_19[0, 12] = 1; V_19[1, 12] = 1; V_19[2, 12] = 1
        
        # Product 14: (a10+a11+a12)(b10+b11+b12)
        U_19[3, 13] = 1; U_19[4, 13] = 1; U_19[5, 13] = 1
        V_19[3, 13] = 1; V_19[4, 13] = 1; V_19[5, 13] = 1
        
        # Product 15: (a20+a21+a22)(b20+b21+b22)
        U_19[6, 14] = 1; U_19[7, 14] = 1; U_19[8, 14] = 1
        V_19[6, 14] = 1; V_19[7, 14] = 1; V_19[8, 14] = 1
        
        # Product 16: (a00-a01+a02)(b00-b01+b02)
        U_19[0, 15] = 1; U_19[1, 15] = -1; U_19[2, 15] = 1
        V_19[0, 15] = 1; V_19[1, 15] = -1; V_19[2, 15] = 1
        
        # Product 17: (a10-a11+a12)(b10-b11+b12)
        U_19[3, 16] = 1; U_19[4, 16] = -1; U_19[5, 16] = 1
        V_19[3, 16] = 1; V_19[4, 16] = -1; V_19[5, 16] = 1
        
        # Product 18: (a20-a21+a22)(b20-b21+b22)
        U_19[6, 17] = 1; U_19[7, 17] = -1; U_19[8, 17] = 1
        V_19[6, 17] = 1; V_19[7, 17] = -1; V_19[8, 17] = 1
        
        # Product 19: (a00+a11+a22)(b00+b11+b22) -- This is a guess for the 19th. 
        # Actually, Laderman's 19th is often (a00+a11+a22)(b00+b11+b22) or similar.
        # Let's use this and solve for W. If it fails, we fall back.
        U_19[0, 18] = 1; U_19[4, 18] = 1; U_19[8, 18] = 1
        V_19[0, 18] = 1; V_19[4, 18] = 1; V_19[8, 18] = 1
        
        # Now construct U, V for the full 3x3x6 problem (Rank 38)
        # U shape (9, 38), V shape (18, 38), W shape (18, 38)
        U = np.zeros((9, 38))
        V = np.zeros((18, 38))
        W = np.zeros((18, 38))
        
        # Block 1: k in 0..2
        # U is shared, so U[:, 0:19] = U_19
        U[:, 0:19] = U_19
        # V for block 1: V_19 corresponds to B indices (j, k) for k in 0..2
        # B index is j*6 + k.
        # V_19 index is j*3 + k.
        # So V[j*6+k, ell] = V_19[j*3+k, ell]
        for j in range(3):
            for k in range(3):
                V[j*6+k, 0:19] = V_19[j*3+k, 0:19]
                
        # Block 2: k in 3..5
        # U is shared, so U[:, 19:38] = U_19
        U[:, 19:38] = U_19
        # V for block 2: V_19 corresponds to B indices (j, k) for k in 0..2
        # But we map to k' = k+3.
        # B index is j*6 + k'.
        # V_19 index is j*3 + k.
        # So V[j*6+k', ell+19] = V_19[j*3+k, ell]
        for j in range(3):
            for k in range(3):
                V[j*6+(k+3), 19:38] = V_19[j*3+k, 0:19]
                
        # Now solve for W.
        # T[a, b, c] = sum_ell U[a, ell] V[b, ell] W[c, ell]
        # For each c (0..17), we have:
        # T[:, :, c] = sum_ell W[c, ell] (U[:, ell] * V[:, ell])
        # Let M_c be a matrix of shape (9*18, 38) where M_c[(a,b), ell] = U[a, ell] * V[b, ell]
        # Then M_c @ W[c, :]^T = T[:, :, c].reshape(-1)
        
        # Build T
        T = np.zeros((9, 18, 18))
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    a = i*3 + j
                    b = j*6 + k
                    c = k*3 + i
                    T[a, b, c] = 1.0
        
        W_sol = np.zeros((18, 38))
        for c in range(18):
            # Build M_c
            M_c = np.zeros((9*18, 38))
            for a in range(9):
                for b in range(18):
                    M_c[a*18+b, :] = U[a, :] * V[b, :]
            
            # Solve M_c @ W_c^T = T_c
            T_c = T[:, :, c].reshape(-1)
            # Use lstsq
            try:
                W_c, residuals, rank, sv = np.linalg.lstsq(M_c, T_c, rcond=None)
                W_sol[c, :] = W_c
            except:
                # If lstsq fails, return schoolbook
                return get_schoolbook()
                
        # Verify W_sol
        # Check if entries are close to integers
        max_diff = np.max(np.abs(W_sol - np.round(W_sol)))
        if max_diff > 1e-9:
            # Not integer, might be invalid or my Laderman coeffs are wrong
            # Try to round and verify
            W_int = np.round(W_sol)
            err = 0.0
            for c in range(18):
                # Reconstruct T_c
                recon = np.zeros(9*18)
                for ell in range(38):
                    recon += W_int[c, ell] * (U[:, ell] * V[:, ell]).reshape(-1)
                err += np.sum(np.abs(recon - T[:, :, c].reshape(-1)))
            if err > 1e-6:
                return get_schoolbook()
            W_sol = W_int
            
        # Final verification with random matrices
        A_rand = np.random.randint(-10, 10, (3, 3))
        B_rand = np.random.randint(-10, 10, (3, 6))
        C_expected = A_rand @ B_rand
        
        # Compute C using decomposition
        # C[k, i] = sum_ell W[k*3+i, ell] * (sum_j U[i*3+j, ell] * A[i,j]) * (sum_j V[j*6+k, ell] * B[j,k])
        # Wait, the decomposition is:
        # C_ki = sum_ell W_ci * (sum_j U_aj * A_ij) * (sum_j V_bk * B_jk)
        # Where a = i*3+j, b = j*6+k.
        # This is not quite right. The bilinear form is:
        # C_ci = sum_ell W_ci * (sum_j U_ai * A_ij) * (sum_j V_bk * B_jk)
        # No, the standard form is:
        # C_ci = sum_ell W_ci * P_ell
        # P_ell = (sum_j U_ai * A_ij) * (sum_j V_bk * B_jk) ?
        # No, P_ell = (sum_j U_ai * A_ij) * (sum_j V_bk * B_jk) is not correct.
        # P_ell = (sum_j U_ai * A_ij) * (sum_j V_bk * B_jk)
        # Wait, U is indexed by (i,j). So U_ai is coeff of A_ij.
        # V is indexed by (j,k). So V_bk is coeff of B_jk.
        # So P_ell = (sum_j U_ai * A_ij) * (sum_j V_bk * B_jk)
        # This is a product of two linear forms.
        
        C_computed = np.zeros((3, 6))
        for i in range(3):
            for k in range(6):
                c_idx = k*3 + i
                sum_val = 0.0
                for ell in range(38):
                    # Compute linear form for A
                    la = 0.0
                    for j in range(3):
                        a_idx = i*3 + j
                        la += U[a_idx, ell] * A_rand[i, j]
                    # Compute linear form for B
                    lb = 0.0
                    for j in range(3):
                        b_idx = j*6 + k
                        lb += V[b_idx, ell] * B_rand[j, k]
                    sum_val += W_sol[c_idx, ell] * la * lb
                C_computed[i, k] = sum_val
                
        if not np.allclose(C_computed, C_expected, atol=1e-6):
            return get_schoolbook()
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W_sol.tolist()}
        
    except Exception:
        return get_schoolbook()

# EVOLVE-BLOCK-END
