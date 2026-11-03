import time
import sys
import numpy as np
import scipy.linalg

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=4, p=8
    
    # Strassen decomposition for 2x2 matrix multiplication (rank 7)
    # Local indices:
    # A: 2x2 -> flat i*2+j
    # B: 2x2 -> flat j*2+k
    # C: 2x2 -> flat i*2+k (local), mapped to W convention k*n+i globally
    
    # Strassen components:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # r=0 (M1): (a00+a11)(b00+b11)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1  # C00
    W_local[3][0] = 1  # C11
    
    # r=1 (M2): (a10+a11)b00
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1   # C10
    W_local[3][1] = -1  # C11
    
    # r=2 (M3): a00(b01-b11)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1   # C01
    W_local[3][2] = 1   # C11
    
    # r=3 (M4): a11(b10-b00)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1   # C00
    W_local[2][3] = 1   # C10
    
    # r=4 (M5): (a00+a01)b11
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1  # C00
    W_local[1][4] = 1   # C01
    
    # r=5 (M6): (a10-a00)(b00+b01)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1   # C11
    
    # r=6 (M7): (a01-a11)(b10+b11)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1   # C00

    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Decompose 2x4 x 4x8 into 8 blocks of 2x2 x 2x2
    # For each column-pair j (0..3) and row-pair s (0..1):
    #   A block: rows 0,1; cols 2s, 2s+1
    #   B block: rows 2s, 2s+1; cols 2j, 2j+1
    #   C block: rows 0,1; cols 2j, 2j+1
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # Map local A indices (i_loc*2+j_loc) to global (i_g*4+j_g)
            a_map = [
                0 * 4 + (2 * s + 0),   # i=0, j=2s
                0 * 4 + (2 * s + 1),   # i=0, j=2s+1
                1 * 4 + (2 * s + 0),   # i=1, j=2s
                1 * 4 + (2 * s + 1),   # i=1, j=2s+1
            ]
            
            # Map local B indices (j_loc*2+k_loc) to global (j_g*8+k_g)
            b_map = [
                (2 * s + 0) * 8 + (2 * j + 0),   # j_g=2s, k_g=2j
                (2 * s + 0) * 8 + (2 * j + 1),   # j_g=2s, k_g=2j+1
                (2 * s + 1) * 8 + (2 * j + 0),   # j_g=2s+1, k_g=2j
                (2 * s + 1) * 8 + (2 * j + 1),   # j_g=2s+1, k_g=2j+1
            ]
            
            # Map local C indices (i_loc*2+k_loc) to W convention (k_g*n + i_g)
            c_map = [
                (2 * j + 0) * 2 + 0,   # i_g=0, k_g=2j
                (2 * j + 1) * 2 + 0,   # i_g=0, k_g=2j+1
                (2 * j + 0) * 2 + 1,   # i_g=1, k_g=2j
                (2 * j + 1) * 2 + 1,   # i_g=1, k_g=2j+1
            ]
            
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W[c_map[lc]][r] += val
                        
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the tensor as a matrix: each rank-1 term r contributes
    # T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # We want to find vectors c in R^56 such that sum_r c_r * T_r = 0
    # This is equivalent to finding the null space of the "tensor matrix"
    
    # Flatten each rank-1 term into a vector of length n*m*p = 64
    # T_r[i,j,k] = U[i,r] * V[j,r] * W[k,r]
    
    M = np.zeros((n * m * p, total_rank))
    for r in range(total_rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[i * m * p + j * p + k, r] = U_np[i, r] * V_np[j, r] * W_np[k, r]
    
    # Find null space of M (64 x 56). If rank(M) < 56, there are dependencies.
    # Use SVD to find the null space
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=True)
        # Null space is spanned by columns of Vt_svd.T corresponding to small singular values
        tol = 1e-10
        rank_m = np.sum(s_vals > tol)
        
        if rank_m < total_rank:
            # There are (total_rank - rank_m) linear dependencies
            # We want to keep rank_m independent terms.
            # The first rank_m columns of Vt_svd.T form a basis for the column space of M.
            # But we need to map back to the original indices.
            # The columns of M are the rank-1 tensors.
            # If they are dependent, we can drop some.
            # We can use QR with pivoting to find a subset of columns that spans the space.
            
            Q, R, P = scipy.linalg.qr(M, pivoting=True)
            diag_R = np.abs(np.diag(R))
            
            rank_m_qr = 0
            for i in range(len(diag_R)):
                if diag_R[i] > tol:
                    rank_m_qr = i + 1
                else:
                    break
            
            if rank_m_qr < total_rank:
                keep_indices = P[:rank_m_qr]
                
                U_new = U_np[:, keep_indices]
                V_new = V_np[:, keep_indices]
                W_new = W_np[:, keep_indices]
                
                new_rank = len(keep_indices)
                
                # Verify correctness
                tensor_recon = np.zeros((n, m, p))
                for r in range(new_rank):
                    u_r = U_new[:, r]
                    v_r = V_new[:, r]
                    w_r = W_new[:, r]
                    tensor_recon += np.einsum('i,j,k->ijk', u_r, v_r, w_r)
                
                max_err = np.max(np.abs(tensor_recon - 1.0))
                
                if max_err < 1e-8:
                    U = U_new.tolist()
                    V = V_new.tolist()
                    W = W_new.tolist()
    except Exception:
        pass

    if time.time() - start_time > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
