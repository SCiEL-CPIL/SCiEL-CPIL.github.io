import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Pre-compute RHS vectors for ALS
    # T is (dim_u, dim_v, dim_w)
    # For U update (fix I): RHS is T[I, :, :].flatten() -> shape (dim_v*dim_w,)
    # For V update (fix J): RHS is T[:, J, :].flatten() -> shape (dim_u*dim_w,)
    # For W update (fix K): RHS is T[:, :, K].flatten() -> shape (dim_u*dim_v,)
    
    RHS_U = np.array([T[i, :, :].flatten() for i in range(dim_u)]) # (dim_u, dim_v*dim_w)
    RHS_V = np.array([T[:, j, :].flatten() for j in range(dim_v)]) # (dim_v, dim_u*dim_w)
    RHS_W = np.array([T[:, :, k].flatten() for k in range(dim_w)]) # (dim_w, dim_u*dim_v)

    def try_rank(r, max_iters=500, num_restarts=200, deadline=None):
        best_result = None
        best_err = float('inf')
        
        for restart in range(num_restarts):
            if deadline and time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(42 + restart)
            U = np.random.randn(dim_u, r) * 0.1
            V = np.random.randn(dim_v, r) * 0.1
            W = np.random.randn(dim_w, r) * 0.1
            
            for it in range(max_iters):
                if deadline and time.time() > deadline:
                    return best_result
                
                # Update U
                # Minimize || T[I,:,:] - U[I,:] * (V[:,:,c] * W[:,:,c]) ||
                # M_U is (dim_v*dim_w, r) with M_U[j*dim_w+k, c] = V[j,c]*W[k,c]
                M_U = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
                try:
                    # Use lstsq for stability and speed on small matrices
                    U_new, _, _, _ = np.linalg.lstsq(M_U, RHS_U.T, rcond=None)
                    U_new = U_new.T
                except:
                    continue
                
                # Update V
                # M_V is (dim_u*dim_w, r) with M_V[i*dim_w+k, c] = U_new[i,c]*W[k,c]
                M_V = np.einsum('ic,kc->ikc', U_new, W).reshape(dim_u * dim_w, r)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M_V, RHS_V.T, rcond=None)
                    V_new = V_new.T
                except:
                    continue
                
                # Update W
                # M_W is (dim_u*dim_v, r) with M_W[i*dim_v+j, c] = U_new[i,c]*V_new[j,c]
                M_W = np.einsum('ic,jc->ijc', U_new, V_new).reshape(dim_u * dim_v, r)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M_W, RHS_W.T, rcond=None)
                    W_new = W_new.T
                except:
                    continue
                
                U, V, W = U_new, V_new, W_new
                
                # Check for convergence/exactness
                # Snap to half-integers
                U_h = np.round(U * 2) / 2
                V_h = np.round(V * 2) / 2
                W_h = np.round(W * 2) / 2
                
                # Check if snap is close
                if np.allclose(U, U_h, atol=1e-5) and np.allclose(V, V_h, atol=1e-5) and np.allclose(W, W_h, atol=1e-5):
                    T_approx = np.einsum('ic,jc,kc->ijk', U_h, V_h, W_h)
                    if np.allclose(T_approx, T, atol=1e-9):
                        return {
                            "U": U_h.tolist(),
                            "V": V_h.tolist(),
                            "W": W_h.tolist()
                        }
                    
                    # Track best error if not exact
                    err = np.linalg.norm(T_approx - T)
                    if err < best_err:
                        best_err = err
                        best_result = {
                            "U": U_h.tolist(),
                            "V": V_h.tolist(),
                            "W": W_h.tolist()
                        }
                        
        return None

    deadline = time.time() + 110 # Leave 10s margin
    
    # Try rank 29
    res = try_rank(29, max_iters=500, num_restarts=200, deadline=deadline)
    if res:
        return res
        
    # Fallback to schoolbook (rank 30)
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
