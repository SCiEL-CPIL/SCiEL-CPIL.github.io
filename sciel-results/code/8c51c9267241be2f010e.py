import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 135

    # Helper to verify a decomposition
    def verify(U, V, W):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        # Check tensor identity: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Helper to snap to half-integers
    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    # Helper to get Strassen 2x2 x 2x2 decomposition
    def get_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: (a+d)(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1; W[1, 0] = -1; W[2, 0] = -1
        
        # Term 2: (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # Term 3: a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # Term 4: d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1
        
        # Term 5: (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1
        
        # Term 6: (c-d)(f+g)
        U[2, 5] = 1; U[3, 5] = -1
        V[1, 5] = 1; V[2, 5] = 1
        W[2, 5] = 1
        
        # Term 7: (b-d)(e+g)
        U[1, 6] = 1; U[3, 6] = -1
        V[0, 6] = 1; V[2, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Helper to get Schoolbook 2x2 x 2x1 decomposition
    def get_school_221():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # Term 1: A[0,0]B[0] -> C[0]
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1
        
        # Term 2: A[0,1]B[1] -> C[0]
        U[1, 1] = 1
        V[1, 1] = 1
        W[0, 1] = 1
        
        # Term 3: A[1,0]B[0] -> C[1]
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1
        
        # Term 4: A[1,1]B[1] -> C[1]
        U[3, 3] = 1
        V[1, 3] = 1
        W[1, 3] = 1
        
        return U, V, W

    # Combine decompositions
    def combine_tensors(decomps):
        total_rank = sum(d[0].shape[1] for d in decomps)
        U_global = np.zeros((8, total_rank))
        V_global = np.zeros((28, total_rank))
        W_global = np.zeros((14, total_rank))
        
        rank_offset = 0
        for U_d, V_d, W_d, i_off, j_off, k_off, m_l, p_l, n_l in decomps:
            r_d = U_d.shape[1]
            
            for u_idx in range(U_d.shape[0]):
                i_l = u_idx // m_l
                j_l = u_idx % m_l
                u_idx_g = (i_off + i_l) * m + (j_off + j_l)
                U_global[u_idx_g, rank_offset:rank_offset+r_d] += U_d[u_idx, :]
                
            for v_idx in range(V_d.shape[0]):
                j_l = v_idx // p_l
                k_l = v_idx % p_l
                v_idx_g = (j_off + j_l) * p + (k_off + k_l)
                V_global[v_idx_g, rank_offset:rank_offset+r_d] += V_d[v_idx, :]
                
            for w_idx in range(W_d.shape[0]):
                k_l = w_idx // n_l
                i_l = w_idx % n_l
                w_idx_g = (k_off + k_l) * n + (i_off + i_l)
                W_global[w_idx_g, rank_offset:rank_offset+r_d] += W_d[w_idx, :]
                
            rank_offset += r_d
            
        return U_global, V_global, W_global

    # Construct the specific decompositions
    U_s, V_s, W_s = get_strassen_222()
    U_sc, V_sc, W_sc = get_school_221()
    
    decomps = []
    
    # A1 (rows 0-1, cols 0-1) * B1a (rows 0-1, cols 0-1)
    decomps.append((U_s, V_s, W_s, 0, 0, 0, 2, 2, 2))
    # A1 * B1b (rows 0-1, cols 2-3)
    decomps.append((U_s, V_s, W_s, 0, 0, 2, 2, 2, 2))
    # A1 * B1c (rows 0-1, cols 4-5)
    decomps.append((U_s, V_s, W_s, 0, 0, 4, 2, 2, 2))
    # A1 * B1d (rows 0-1, cols 6-6)
    decomps.append((U_sc, V_sc, W_sc, 0, 0, 6, 2, 1, 2))
    
    # A2 (rows 0-1, cols 2-3) * B2a (rows 2-3, cols 0-1)
    decomps.append((U_s, V_s, W_s, 0, 2, 0, 2, 2, 2))
    # A2 * B2b (rows 2-3, cols 2-3)
    decomps.append((U_s, V_s, W_s, 0, 2, 2, 2, 2, 2))
    # A2 * B2c (rows 2-3, cols 4-5)
    decomps.append((U_s, V_s, W_s, 0, 2, 4, 2, 2, 2))
    # A2 * B2d (rows 2-3, cols 6-6)
    decomps.append((U_sc, V_sc, W_sc, 0, 2, 6, 2, 1, 2))
    
    U_final, V_final, W_final = combine_tensors(decomps)
    
    # Verify
    if verify(U_final, V_final, W_final):
        return {
            "U": U_final.tolist(),
            "V": V_final.tolist(),
            "W": W_final.tolist()
        }
    else:
        # Fallback to schoolbook
        rank_schoolbook = n * m * p
        U_school = [[0] * rank_schoolbook for _ in range(n * m)]
        V_school = [[0] * rank_schoolbook for _ in range(m * p)]
        W_school = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
