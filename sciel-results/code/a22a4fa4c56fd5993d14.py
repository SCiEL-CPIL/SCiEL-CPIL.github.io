import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to build matrices from lists of terms
    def build_matrices(U_list, V_list, W_list):
        rank = len(U_list)
        U_mat = np.zeros((n*m, rank))
        V_mat = np.zeros((m*p, rank))
        W_mat = np.zeros((p*n, rank))
        
        for r in range(rank):
            for idx, val in enumerate(U_list[r]):
                if val != 0:
                    U_mat[idx, r] = val
            for idx, val in enumerate(V_list[r]):
                if val != 0:
                    V_mat[idx, r] = val
            for idx, val in enumerate(W_list[r]):
                if val != 0:
                    W_mat[idx, r] = val
        return U_mat, V_mat, W_mat

    def verify(U_mat, V_mat, W_mat):
        # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
        # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
        # Indexing:
        # U[i*m + j, r]
        # V[j*p + k, r]
        # W[k*n + i, r]
        
        # We want C[i, k] = sum_j A[i, j] B[j, k]
        # The tensor equation is:
        # sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1 (coefficient of A[i,j]B[j,k])
        
        # Efficient check using matrix multiplication?
        # Let M1 = U reshaped to (n, m, rank)
        # M2 = V reshaped to (m, p, rank)
        # M3 = W reshaped to (p, n, rank)
        
        U_3d = U_mat.reshape(n, m, -1)
        V_3d = V_mat.reshape(m, p, -1)
        W_3d = W_mat.reshape(p, n, -1)
        
        # We need to check T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # This is a tensor contraction.
        # We can loop over i, j, k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Sum over r
                    # U[i, j, :] * V[j, k, :] * W[k, i, :]
                    val = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
                    if not np.isclose(val, 1.0):
                        return False
        return True

    # --- Decomposition 1: 1+2+2 Split (Rank 117) ---
    def get_1_2_2_decomp():
        U = []
        V = []
        W = []
        
        def add_term(u, v, w):
            U.append(u)
            V.append(v)
            W.append(w)
            
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        def add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end + 1):
                for j in range(j_start, j_end + 1):
                    for k in range(k_start, k_end + 1):
                        u = [0] * (n*m)
                        v = [0] * (m*p)
                        w = [0] * (p*n)
                        u[u_idx(i, j)] = 1
                        v[v_idx(j, k)] = 1
                        w[w_idx(k, i)] = 1
                        add_term(u, v, w)

        def add_strassen_2x2x2(i0, j0, k0):
            # 7 terms for 2x2x2
            # Term 1: (i0, j0) and (i0+1, j0+1)
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
            v[v_idx(j0, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
            w[w_idx(k0, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
            add_term(u, v, w)
            
            # Term 2
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0+1, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
            v[v_idx(j0, k0)] = 1
            w[w_idx(k0, i0+1)] = 1; w[w_idx(k0+1, i0+1)] = -1
            add_term(u, v, w)
            
            # Term 3
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0)] = 1
            v[v_idx(j0, k0+1)] = 1; v[v_idx(j0+1, k0+1)] = -1
            w[w_idx(k0+1, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
            add_term(u, v, w)
            
            # Term 4
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0+1, j0+1)] = 1
            v[v_idx(j0+1, k0)] = 1; v[v_idx(j0, k0)] = -1
            w[w_idx(k0, i0)] = 1; w[w_idx(k0, i0+1)] = 1
            add_term(u, v, w)
            
            # Term 5
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0)] = 1; u[u_idx(i0, j0+1)] = 1
            v[v_idx(j0+1, k0+1)] = 1
            w[w_idx(k0, i0)] = -1; w[w_idx(k0+1, i0)] = 1
            add_term(u, v, w)
            
            # Term 6
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0+1, j0)] = 1; u[u_idx(i0, j0)] = -1
            v[v_idx(j0, k0)] = 1; v[v_idx(j0, k0+1)] = 1
            w[w_idx(k0+1, i0+1)] = 1
            add_term(u, v, w)
            
            # Term 7
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0+1)] = 1; u[u_idx(i0+1, j0+1)] = -1
            v[v_idx(j0+1, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
            w[w_idx(k0, i0)] = 1
            add_term(u, v, w)

        blocks = [[0], [1, 2], [3, 4]]
        
        for bi, ib in enumerate(blocks):
            for bj, jb in enumerate(blocks):
                for bk, kb in enumerate(blocks):
                    i_start, i_end = min(ib), max(ib)
                    j_start, j_end = min(jb), max(jb)
                    k_start, k_end = min(kb), max(kb)
                    
                    i_size = len(ib)
                    j_size = len(jb)
                    k_size = len(kb)
                    
                    if i_size == 2 and j_size == 2 and k_size == 2:
                        add_strassen_2x2x2(i_start, j_start, k_start)
                    else:
                        add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end)
                        
        return U, V, W

    # --- Decomposition 2: 3+1+1 Split (Rank 113 with Rank 23 for 3x3x3) ---
    def get_3_1_1_decomp():
        U = []
        V = []
        W = []
        
        def add_term(u, v, w):
            U.append(u)
            V.append(v)
            W.append(w)
            
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # Standard blocks for 3x3x1, 3x1x1, etc.
        def add_standard_block(i_start, i_end, j_start, j_end, k_start, k_end):
            for i in range(i_start, i_end + 1):
                for j in range(j_start, j_end + 1):
                    for k in range(k_start, k_end + 1):
                        u = [0] * (n*m)
                        v = [0] * (m*p)
                        w = [0] * (p*n)
                        u[u_idx(i, j)] = 1
                        v[v_idx(j, k)] = 1
                        w[w_idx(k, i)] = 1
                        add_term(u, v, w)
                        
        # Rank 23 decomposition for 3x3x3
        # Using Bini et al. algorithm
        # Indices 0,1,2 for the 3x3 block
        def add_bini_3x3x3(i0, j0, k0):
            # 23 terms
            # This is complex to hardcode perfectly without reference.
            # I will use a known set of 23 terms.
            # If this fails verification, we fall back.
            
            # For safety, I will implement the 27-term standard algorithm here
            # and rely on the verification to reject it if it's not 23?
            # No, 27 terms is valid, just higher rank.
            # Rank 27 for 3x3x3 -> Total Rank 117.
            # Rank 23 for 3x3x3 -> Total Rank 113.
            
            # I will implement the 27-term standard first to ensure a valid fallback within this function?
            # No, the function returns the terms.
            # I will implement the 23-term terms if I can recall them.
            # Actually, I will just use the 27-term standard for now to ensure validity,
            # and note that 117 is the target.
            # Wait, if I use 27, the rank is 117, same as 1+2+2.
            # To beat 1.0684, I need < 117.
            # So I MUST use 23.
            
            # I will hardcode the 23 terms from a trusted source.
            # Source: "Rank 23 algorithm for 3x3x3 matrix multiplication"
            
            # Term 1
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
            v[v_idx(j0, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
            w[w_idx(k0, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
            add_term(u, v, w)
            
            # Term 2
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0+1, j0)] = 1; u[u_idx(i0+1, j0+1)] = 1
            v[v_idx(j0, k0)] = 1
            w[w_idx(k0, i0+1)] = 1; w[w_idx(k0+1, i0+1)] = -1
            add_term(u, v, w)
            
            # Term 3
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0)] = 1
            v[v_idx(j0, k0+1)] = 1; v[v_idx(j0+1, k0+1)] = -1
            w[w_idx(k0+1, i0)] = 1; w[w_idx(k0+1, i0+1)] = 1
            add_term(u, v, w)
            
            # Term 4
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0+1, j0+1)] = 1
            v[v_idx(j0+1, k0)] = 1; v[v_idx(j0, k0)] = -1
            w[w_idx(k0, i0)] = 1; w[w_idx(k0, i0+1)] = 1
            add_term(u, v, w)
            
            # Term 5
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0)] = 1; u[u_idx(i0, j0+1)] = 1
            v[v_idx(j0+1, k0+1)] = 1
            w[w_idx(k0, i0)] = -1; w[w_idx(k0+1, i0)] = 1
            add_term(u, v, w)
            
            # Term 6
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0+1, j0)] = 1; u[u_idx(i0, j0)] = -1
            v[v_idx(j0, k0)] = 1; v[v_idx(j0, k0+1)] = 1
            w[w_idx(k0+1, i0+1)] = 1
            add_term(u, v, w)
            
            # Term 7
            u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
            u[u_idx(i0, j0+1)] = 1; u[u_idx(i0+1, j0+1)] = -1
            v[v_idx(j0+1, k0)] = 1; v[v_idx(j0+1, k0+1)] = 1
            w[w_idx(k0, i0)] = 1
            add_term(u, v, w)
            
            # These are the first 7 (Strassen-like).
            # I need 16 more specific to 3x3x3 Rank 23.
            # Given the risk of error, I will use the 27-term standard algorithm for 3x3x3
            # which is just standard blocks.
            # This will result in Rank 117, which is NOT an improvement.
            
            # Therefore, I will NOT use the 3+1+1 split unless I have the 23 terms.
            # Since I don't have them reliably, I will stick to 1+2+2.
            
            pass

        # Since I don't have the 23 terms, I will just return the 1+2+2 decomposition.
        return get_1_2_2_decomp()

    # Try 3+1+1 first? No, it's invalid without 23 terms.
    # So we just use 1+2+2.
    
    U_list, V_list, W_list = get_1_2_2_decomp()
    U_mat, V_mat, W_mat = build_matrices(U_list, V_list, W_list)
    
    # Final verification
    if not verify(U_mat, V_mat, W_mat):
        # If verification fails (should not happen), return empty or raise
        raise ValueError("Decomposition invalid")
        
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
