import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    n, m, p = 2, 3, 9
    
    # Build the rank-50 decomposition
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    k8 = 8
    v_e = k8
    v_g = p + k8
    
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0
    
    T_flat = T.flatten()
    
    # Compute the full decomposition tensor efficiently
    # Decomp[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    decomp_full = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    # Verify the full decomposition is correct
    if not np.allclose(decomp_full, T, atol=1e-10):
        # Fallback: return rank-50
        return {"U": U, "V": V, "W": W}
    
    # Check for linear dependencies using SVD
    try:
        # Build the coefficient matrix: each column r is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # This is a 2916 x 50 matrix
        n_entries = n * m * m * p * p * n
        mat = np.zeros((n_entries, rank))
        
        for r in range(rank):
            u_r = U_np[:, r]
            v_r = V_np[:, r]
            w_r = W_np[:, r]
            vec = np.einsum('i,j,k->ijk', u_r, v_r, w_r).flatten()
            mat[:, r] = vec
        
        # Compute SVD
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        
        # Find the numerical rank
        tol = s[0] * rank * 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There are linear dependencies
            # The null space is spanned by the last (rank - num_rank) rows of Vt
            null_dim = rank - num_rank
            best_residual = float('inf')
            best_drop_col = None
            best_U = None
            best_V = None
            best_W = None
            
            for null_idx in range(null_dim):
                if time.time() > deadline - 10:
                    break
                
                null_vec = Vt[-(null_idx+1)]
                abs_null = np.abs(null_vec)
                # Only consider columns with significant weight in null space
                threshold = np.max(abs_null) * 0.05
                candidates = np.where(abs_null > threshold)[0]
                
                for drop_col in candidates:
                    if time.time() > deadline - 5:
                        break
                    
                    # Remove that column
                    mask = np.ones(rank, dtype=bool)
                    mask[drop_col] = False
                    U_new = U_np[:, mask]
                    V_new = V_np[:, mask]
                    W_new = W_np[:, mask]
                    
                    # Verify efficiently: recompute the full tensor
                    decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                    residual = np.sum((decomp_new - T)**2)
                    
                    if residual < best_residual:
                        best_residual = residual
                        best_drop_col = drop_col
                        best_U = U_new
                        best_V = V_new
                        best_W = W_new
                    
                    if residual < 1e-8:
                        break
                
                if best_residual < 1e-8:
                    break
            
            if best_residual < 1e-8 and best_drop_col is not None:
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
    except Exception as e:
        pass
    
    # Fall back to rank-50
    return {"U": U, "V": V, "W": W}
