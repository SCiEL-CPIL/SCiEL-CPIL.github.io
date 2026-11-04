import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 24
    
    # Trivial decomposition
    U_trivial = np.zeros((n * m, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((m * p, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((p * n, trivial_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
                
    T = np.ones((n, m, p), dtype=np.float64)
    
    best_result = {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
    best_rank = trivial_rank

    def compute_error(U, V, W):
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        try:
            T_calc = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
            err = np.sum((T_calc - T)**2)
            return err
        except Exception:
            return float('inf')

    def verify_decomposition(U, V, W, tol=1e-10):
        err = compute_error(U, V, W)
        return err < tol

    def discrete_local_search(U, V, W, rank, max_sweeps=5):
        vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        best_err = compute_error(U, V, W)
        
        for sweep in range(max_sweeps):
            if time.time() > deadline - 5.0:
                return U, V, W, best_err
                
            improved = False
            
            # Update U
            for r in range(rank):
                for idx in range(n*m):
                    old = U[idx, r]
                    for new_val in vals:
                        if new_val == old: continue
                        U[idx, r] = new_val
                        err = compute_error(U, V, W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            U[idx, r] = old
                            
            # Update V
            for r in range(rank):
                for idx in range(m*p):
                    old = V[idx, r]
                    for new_val in vals:
                        if new_val == old: continue
                        V[idx, r] = new_val
                        err = compute_error(U, V, W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            V[idx, r] = old
                            
            # Update W
            for r in range(rank):
                for idx in range(p*n):
                    old = W[idx, r]
                    for new_val in vals:
                        if new_val == old: continue
                        W[idx, r] = new_val
                        err = compute_error(U, V, W)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            W[idx, r] = old
                            
            if not improved:
                break
                
        return U, V, W, best_err

    def search_rank(rank, num_trials=10):
        if time.time() > deadline - 10.0:
            return None
            
        np.random.seed(42 + rank)
        vals_init = [-1.0, 0.0, 1.0]
        
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                return None
                
            U = np.random.choice(vals_init, size=(n * m, rank))
            V = np.random.choice(vals_init, size=(m * p, rank))
            W = np.random.choice(vals_init, size=(p * n, rank))
            
            U_final, V_final, W_final, final_err = discrete_local_search(
                U, V, W, rank, max_sweeps=5
            )
            
            if final_err < 1e-10:
                if verify_decomposition(U_final, V_final, W_final):
                    return (U_final, V_final, W_final)
                    
        return None

    # Try ranks from 16 to 23
    for rank in [16, 17, 18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 15.0:
            break
        result = search_rank(rank, num_trials=10)
        if result:
            U, V, W = result
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            best_rank = rank
            break

    if best_rank == trivial_rank:
        return {
            "U": U_trivial.tolist(),
            "V": V_trivial.tolist(),
            "W": W_trivial.tolist()
        }
        
    return best_result
