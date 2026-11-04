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

    def verify_decomposition(U, V, W, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - T[i, j, k]) > tol:
                        return False
        return True

    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (val - 1.0)**2
        return err

    def half_integer_local_search(U, V, W, rank, max_iter=50):
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        best_err = compute_error(U, V, W)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                return U, V, W, best_err
                
            improved = False
            
            # Try changing one element in U, V, or W
            # Change U
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
            # Change V
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
            # Change W
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

    def try_rank_with_structured_init(rank, num_trials=30):
        if time.time() > deadline - 10.0:
            return None
            
        np.random.seed(42 + rank)
        
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                return None
            
            # Structured initialization: start from trivial and remove some columns
            # then perturb
            U = np.zeros((n * m, rank), dtype=np.float64)
            V = np.zeros((m * p, rank), dtype=np.float64)
            W = np.zeros((p * n, rank), dtype=np.float64)
            
            # Initialize with a subset of trivial decomposition
            for r in range(rank):
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
            
            # Add random perturbation
            U += np.random.randn(n * m, rank) * 0.3
            V += np.random.randn(m * p, rank) * 0.3
            W += np.random.randn(p * n, rank) * 0.3
            
            # Snap to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            # Refine with half-integer local search
            U_final, V_final, W_final, final_err = half_integer_local_search(
                U_snapped, V_snapped, W_snapped, rank, max_iter=40
            )
            
            if final_err < 1e-10:
                if verify_decomposition(U_final, V_final, W_final):
                    return (U_final, V_final, W_final)
                    
            # Also try pure random initialization
            U_rand = np.random.randn(n * m, rank) * 0.5
            V_rand = np.random.randn(m * p, rank) * 0.5
            W_rand = np.random.randn(p * n, rank) * 0.5
            
            U_rand_snapped = np.round(U_rand * 2) / 2
            V_rand_snapped = np.round(V_rand * 2) / 2
            W_rand_snapped = np.round(W_rand * 2) / 2
            
            U_final2, V_final2, W_final2, final_err2 = half_integer_local_search(
                U_rand_snapped, V_rand_snapped, W_rand_snapped, rank, max_iter=40
            )
            
            if final_err2 < 1e-10:
                if verify_decomposition(U_final2, V_final2, W_final2):
                    return (U_final2, V_final2, W_final2)
                    
        return None

    # Try ranks from 20 to 23
    for rank in [20, 21, 22, 23]:
        if time.time() > deadline - 15.0:
            break
        result = try_rank_with_structured_init(rank, num_trials=25)
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
