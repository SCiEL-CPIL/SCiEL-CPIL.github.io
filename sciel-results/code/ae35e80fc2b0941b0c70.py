import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p
    
    # Trivial decomposition fallback
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
                
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    best_result = {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
    best_rank = trivial_rank

    def verify_decomposition(U, V, W, tol=1e-10):
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2)):
                return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def fast_search(rank, num_trials=500):
        if time.time() > deadline - 5.0:
            return None
            
        np.random.seed(42 + rank)
        
        vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for trial in range(num_trials):
            if time.time() > deadline - 5.0:
                return None
                
            # Initialize
            U = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(n*m)])
            V = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(m*p)])
            W = np.array([[np.random.choice(vals) for _ in range(rank)] for _ in range(p*n)])
            
            # Compute initial T and Error
            T = np.zeros((n, m, p), dtype=np.float64)
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        u_val = U[i*m + j, r]
                        if u_val == 0: continue
                        for k in range(p):
                            T[i, j, k] += u_val * V[j*p + k, r] * W[k*n + i, r]
                            
            err = np.sum((T - 1.0)**2)
            
            if err < 1e-10:
                if verify_decomposition(U, V, W):
                    return (U, V, W)

            # Local Search
            improved = True
            # Limit iterations to prevent infinite loops if stuck in a cycle (though greedy usually doesn't cycle)
            max_iterations = 5000
            iteration = 0
            
            while improved and iteration < max_iterations:
                if time.time() > deadline - 5.0:
                    return None
                
                improved = False
                iteration += 1

                # Try changing U
                for r in range(rank):
                    for idx in range(n*m):
                        i = idx // m
                        j = idx % m
                        old_val = U[idx, r]
                        
                        for new_val in vals:
                            if new_val == old_val: continue
                            
                            delta_err = 0.0
                            for k in range(p):
                                a_k = V[j*p + k, r] * W[k*n + i, r]
                                if a_k == 0: continue
                                old_t = T[i, j, k]
                                new_t = old_t + (new_val - old_val) * a_k
                                delta_err += (new_t - 1.0)**2 - (old_t - 1.0)**2
                                
                            if delta_err < -1e-12:
                                for k in range(p):
                                    a_k = V[j*p + k, r] * W[k*n + i, r]
                                    if a_k != 0:
                                        T[i, j, k] += (new_val - old_val) * a_k
                                U[idx, r] = new_val
                                err += delta_err
                                improved = True
                                break

                # Try changing V
                for r in range(rank):
                    for idx in range(m*p):
                        j = idx // p
                        k = idx % p
                        old_val = V[idx, r]
                        
                        for new_val in vals:
                            if new_val == old_val: continue
                            
                            delta_err = 0.0
                            for i in range(n):
                                a_i = U[i*m + j, r] * W[k*n + i, r]
                                if a_i == 0: continue
                                old_t = T[i, j, k]
                                new_t = old_t + (new_val - old_val) * a_i
                                delta_err += (new_t - 1.0)**2 - (old_t - 1.0)**2
                                
                            if delta_err < -1e-12:
                                for i in range(n):
                                    a_i = U[i*m + j, r] * W[k*n + i, r]
                                    if a_i != 0:
                                        T[i, j, k] += (new_val - old_val) * a_i
                                V[idx, r] = new_val
                                err += delta_err
                                improved = True
                                break

                # Try changing W
                for r in range(rank):
                    for idx in range(p*n):
                        k = idx // n
                        i = idx % n
                        old_val = W[idx, r]
                        
                        for new_val in vals:
                            if new_val == old_val: continue
                            
                            delta_err = 0.0
                            for j in range(m):
                                a_j = U[i*m + j, r] * V[j*p + k, r]
                                if a_j == 0: continue
                                old_t = T[i, j, k]
                                new_t = old_t + (new_val - old_val) * a_j
                                delta_err += (new_t - 1.0)**2 - (old_t - 1.0)**2
                                
                            if delta_err < -1e-12:
                                for j in range(m):
                                    a_j = U[i*m + j, r] * V[j*p + k, r]
                                    if a_j != 0:
                                        T[i, j, k] += (new_val - old_val) * a_j
                                W[idx, r] = new_val
                                err += delta_err
                                improved = True
                                break

            if err < 1e-10:
                if verify_decomposition(U, V, W):
                    return (U, V, W)
                    
        return None

    # Try ranks
    for rank in [20, 21, 22, 23]:
        if time.time() > deadline - 10.0:
            break
        result = fast_search(rank)
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
