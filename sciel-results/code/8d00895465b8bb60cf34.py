import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    min_rank = max(n * m, m * p, p * n)
    schoolbook_rank = n * m * p
    
    def get_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    best_solution = None
    best_rank = schoolbook_rank
    
    # Try ranks from min_rank to schoolbook_rank - 1
    # If we find a solution, it will be better than schoolbook
    max_ranks_to_try = min(schoolbook_rank - min_rank, 50)
    
    for rank_offset in range(max_ranks_to_try):
        if time.time() > deadline:
            break
            
        R = min_rank + rank_offset
        
        num_restarts = 2
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + rank_offset * 100 + restart)
            
            U = np.random.randn(n * m, R) * 0.5
            V = np.random.randn(m * p, R) * 0.5
            W = np.random.randn(p * n, R) * 0.5
            
            max_iters = 100
            last_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        W_slice = W[:, i, :] # (p, R)
                        V_slice = V[j, :]    # (p, R)
                        A = V_slice * W_slice # (p, R)
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        U_slice = U[:, j, :] # (n, R)
                        W_slice = W[k, :]    # (n, R)
                        B = U_slice * W_slice # (n, R)
                        v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        U_slice = U[i, :] # (m, R)
                        V_slice = V[:, k] # (m, R)
                        C = U_slice * V_slice # (m, R)
                        w, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                        
                # Error
                U_3d = U.reshape(n, m, R)
                V_3d = V.reshape(m, p, R)
                W_3d = W.reshape(p, n, R)
                T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                err = np.sum((T - 1.0) ** 2)
                
                if err < last_err:
                    last_err = err
                    if err < 1e-10:
                        break
                else:
                    # Optional: break if error increases
                    pass

            if last_err < 1e-6:
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                U_3d = U_int.reshape(n, m, R)
                V_3d = V_int.reshape(m, p, R)
                W_3d = W_int.reshape(p, n, R)
                T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                
                if np.allclose(T, 1.0, atol=1e-9):
                    best_rank = R
                    best_solution = (U_int, V_int, W_int)
                    break

        if best_solution is not None:
            break

    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
