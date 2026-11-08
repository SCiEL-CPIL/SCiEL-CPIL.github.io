import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Construct the target tensor T
    # T[I, J, K] where I=i*m+j, J=j*p+k, K=k*n+i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    # Pre-compute RHS vectors for ALS
    # For U update (fix I): RHS is T[I, :, :].flatten()
    # For V update (fix J): RHS is T[:, J, :].flatten()
    # For W update (fix K): RHS is T[:, :, K].flatten()
    
    RHS_U = np.array([T[i, :, :].flatten() for i in range(dim_u)]) # (dim_u, dim_v*dim_w)
    RHS_V = np.array([T[:, j, :].flatten() for j in range(dim_v)]) # (dim_v, dim_u*dim_w)
    RHS_W = np.array([T[:, :, k].flatten() for k in range(dim_w)]) # (dim_w, dim_u*dim_v)

    def try_rank(r, max_iters=500, num_restarts=200, deadline=None):
        best_result = None
        
        for restart in range(num_restarts):
            if deadline and time.time() > deadline:
                break
                
            # Use different random seeds for diversity
            np.random.seed(42 + restart * 101)
            
            # Initialize with small random values
            U = np.random.randn(dim_u, r)
            V = np.random.randn(dim_v, r)
            W = np.random.randn(dim_w, r)
            
            # Normalize to prevent divergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(max_iters):
                if deadline and time.time() > deadline:
                    return best_result
                
                # Update U
                # M_U[(j,k), c] = V[j,c] * W[k,c]
                # We need to solve U[i, :] such that sum_c U[i,c] M_U[(j,k), c] = T[i,j,k]
                # This is a linear system: M_U.T @ U[i, :] = T[i, :, :].flatten()
                # Actually, it's: U[i, :] = pinv(M_U) @ T[i, :, :].flatten()
                
                M_U = np.einsum('jc,kc->jkc', V, W).reshape(dim_v * dim_w, r)
                try:
                    # Using lstsq for stability if r is small
                    # But pinv is fine for small r
                    M_U_pinv = np.linalg.pinv(M_U)
                    U_new = (M_U_pinv @ RHS_U.T).T
                except:
                    continue
                
                # Update V
                # M_V[(i,k), c] = U[i,c] * W[k,c]
                M_V = np.einsum('ic,kc->ikc', U_new, W).reshape(dim_u * dim_w, r)
                try:
                    M_V_pinv = np.linalg.pinv(M_V)
                    V_new = (M_V_pinv @ RHS_V.T).T
                except:
                    continue
                
                # Update W
                # M_W[(i,j), c] = U[i,c] * V[j,c]
                M_W = np.einsum('ic,jc->ijc', U_new, V_new).reshape(dim_u * dim_v, r)
                try:
                    M_W_pinv = np.linalg.pinv(M_W)
                    W_new = (M_W_pinv @ RHS_W.T).T
                except:
                    continue
                
                U, V, W = U_new, V_new, W_new
                
                # Check for convergence to a valid integer/half-integer solution
                # We check every 10 iterations to save time
                if it % 10 == 0:
                    # Snap to nearest half-integer
                    U_h = np.round(U * 2) / 2
                    V_h = np.round(V * 2) / 2
                    W_h = np.round(W * 2) / 2
                    
                    # Check if the snap is close to the current solution
                    # This indicates we are in the basin of attraction of an integer solution
                    if np.linalg.norm(U - U_h) < 0.1 and np.linalg.norm(V - V_h) < 0.1 and np.linalg.norm(W - W_h) < 0.1:
                        # Verify exactness
                        T_approx = np.einsum('ic,jc,kc->ijk', U_h, V_h, W_h)
                        if np.allclose(T_approx, T, atol=1e-9):
                            return {
                                "U": U_h.tolist(),
                                "V": V_h.tolist(),
                                "W": W_h.tolist()
                            }
                            
        return None

    deadline = time.time() + 110 # Leave 10s margin for safety

    # Try rank 29
    res = try_rank(29, max_iters=300, num_restarts=500, deadline=deadline)
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
