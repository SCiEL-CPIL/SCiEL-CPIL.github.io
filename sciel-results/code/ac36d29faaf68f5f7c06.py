import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # The problem decomposes into two 2x3x3 multiplications
    # Rank of 2x3x3 is at most 15 (known result)
    # We attempt to find a rank 15 decomposition for 2x3x3
    # If successful, total rank is 30.
    # If not, we fall back to schoolbook rank 36.
    
    def find_rank_decomp(n, m, p, R, max_iters=100, time_limit=5):
        # Target tensor M_{n,m,p}
        # We want sum_r u_r v_r w_r = M
        # Flatten U, V, W
        # U shape (n*m, R), V (m*p, R), W (p*n, R)
        
        # Initialize with schoolbook if R >= n*m*p, else random
        if R >= n*m*p:
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            return U, V, W
        else:
            # Random initialization
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            def residuals(params):
                u_flat, v_flat, w_flat = params
                u = u_flat.reshape(n*m, R)
                v = v_flat.reshape(m*p, R)
                w = w_flat.reshape(p*n, R)
                
                # Compute sum_r u_r v_r w_r
                # This is a tensor of shape (n*m, m*p, p*n)
                # We want it to match the target tensor
                # Target tensor T has T[i*m+j, j*p+k, k*n+i] = 1
                # All other entries 0?
                # No, the decomposition is for the multiplication map.
                # The condition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k?
                # No, that's not right.
                # The condition is that the bilinear form matches.
                # Let's check the seed code logic.
                # Seed code sets U, V, W such that for each (i,j,k), there is a rank 1 term.
                # The "target" tensor is the sum of these rank 1 terms.
                # So we are trying to approximate the tensor M = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}
                # by sum_{r=1}^R u_r \otimes v_r \otimes w_r.
                
                # Construct target tensor T (flattened)
                # T is sparse.
                # We compute the product tensor P = sum_r u_r v_r w_r
                # P[a, b, c] = sum_r u[a, r] v[b, r] w[c, r]
                # We want P[a, b, c] = 1 if (a,b,c) corresponds to (i,j,k) i.e. a=i*m+j, b=j*p+k, c=k*n+i
                # and 0 otherwise.
                
                # Efficient computation of P is O(R * n*m * m*p * p*n) which is too slow.
                # We can compute the residuals only on the non-zero entries of the target.
                # The target has n*m*p non-zero entries.
                # For each (i,j,k), we want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1.
                # This gives n*m*p equations.
                # Variables: R * (n*m + m*p + p*n).
                
                idx_i = np.repeat(np.arange(n), m*p)
                idx_j = np.tile(np.repeat(np.arange(m), p), n)
                idx_k = np.tile(np.arange(p), n*m)
                
                u_idx = idx_i * m + idx_j
                v_idx = idx_j * p + idx_k
                w_idx = idx_k * n + idx_i
                
                # Extract relevant rows
                u_sel = u[u_idx, :]
                v_sel = v[v_idx, :]
                w_sel = w[w_idx, :]
                
                # sum_r u_sel[r] v_sel[r] w_sel[r]
                # This is element-wise product sum over r
                prod = u_sel * v_sel * w_sel
                s = np.sum(prod, axis=1)
                
                # Residuals: s - 1
                return s - 1

            start_time = time.time()
            best_residual = np.inf
            best_params = None
            
            for _ in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                # Random restart
                U_init = np.random.randn(n*m, R)
                V_init = np.random.randn(m*p, R)
                W_init = np.random.randn(p*n, R)
                # Normalize to avoid scaling issues
                U_init /= np.linalg.norm(U_init)
                V_init /= np.linalg.norm(V_init)
                W_init /= np.linalg.norm(W_init)
                
                params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                try:
                    res = least_squares(residuals, params_init, method='lm', max_nfev=1000)
                    if res.cost < best_residual:
                        best_residual = res.cost
                        best_params = res.x
                except Exception:
                    pass
            
            if best_residual < 1e-8:
                u_flat, v_flat, w_flat = best_params
                U = u_flat.reshape(n*m, R)
                V = v_flat.reshape(m*p, R)
                W = w_flat.reshape(p*n, R)
                return U, V, W
            else:
                return None

    # Try to find rank 15 for 2x3x3
    U_sub, V_sub, W_sub = find_rank_decomp(2, 3, 3, 15)
    
    if U_sub is not None:
        # Duplicate for the two 3x3 blocks of the 3x6 matrix
        # B is 3x6. Split into B1 (3x3) and B2 (3x3).
        # C is 2x6. Split into C1 (2x3) and C2 (2x3).
        # C1 = A B1, C2 = A B2.
        # The decomposition for C1 uses U_sub, V_sub1, W_sub1.
        # The decomposition for C2 uses U_sub, V_sub2, W_sub2.
        # U is shared? No, U is the same for both because A is the same.
        # But in the global decomposition, U has shape (2*3, Rank).
        # V has shape (3*6, Rank).
        # W has shape (6*2, Rank).
        
        Rank = 30
        U = np.zeros((2*3, Rank))
        V = np.zeros((3*6, Rank))
        W = np.zeros((6*2, Rank))
        
        # First 15 columns for block 1
        U[:, 0:15] = U_sub
        # V for block 1: B1 indices are j*3 + k (k=0,1,2)
        # Global V index is j*6 + k_global.
        # For block 1, k_global = k.
        for j in range(3):
            for k in range(3):
                V[j*6 + k, 0:15] = V_sub[j*3 + k, 0:15]
        # W for block 1: C1 indices are k*2 + i.
        # Global W index is k_global*2 + i.
        # For block 1, k_global = k.
        for k in range(3):
            for i in range(2):
                W[k*2 + i, 0:15] = W_sub[k*2 + i, 0:15]
                
        # Next 15 columns for block 2
        U[:, 15:30] = U_sub
        # V for block 2: B2 indices are j*3 + k (k=0,1,2) but global k = 3+k.
        for j in range(3):
            for k in range(3):
                V[j*6 + (3+k), 15:30] = V_sub[j*3 + k, 0:15]
        # W for block 2: C2 indices are k*2 + i.
        # Global W index is k_global*2 + i.
        # For block 2, k_global = 3+k.
        for k in range(3):
            for i in range(2):
                W[(3+k)*2 + i, 15:30] = W_sub[k*2 + i, 0:15]
                
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = 2*3*6
        U = [[0] * rank for _ in range(2*3)]
        V = [[0] * rank for _ in range(3*6)]
        W = [[0] * rank for _ in range(6*2)]
        column = 0
        for i in range(2):
            for j in range(3):
                for k in range(6):
                    U[i*3+j][column] = 1
                    V[j*6+k][column] = 1
                    W[k*2+i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
