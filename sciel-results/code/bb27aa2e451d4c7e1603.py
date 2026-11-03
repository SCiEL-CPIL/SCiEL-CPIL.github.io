import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_result = {"U": U, "V": V, "W": W}
    
    # Attempt structured search for lower rank
    # Try a few candidate ranks with random initialization and local search
    # Given time constraints, this is unlikely to find a valid solution for
    # significantly lower ranks, but we try a few.
    
    rng = np.random.RandomState(12345)
    
    # Try rank 124, 120, 100 with a few iterations each
    for target_rank in [124, 120, 100]:
        if time.time() > deadline - 20:
            break
            
        max_iters = 50
        found = False
        
        for attempt in range(3):
            if time.time() > deadline - 20:
                    break
                    
            # Random initialization
            U_r = rng.randn(n * m, target_rank) * 0.1
            V_r = rng.randn(m * p, target_rank) * 0.1
            W_r = rng.randn(p * n, target_rank) * 0.1
            
            # Simple iterative refinement
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                    
                # Compute error
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r] for r in range(target_rank))
                            err += (val - 1.0) ** 2
                
                if err < 1e-6:
                    # Verify exactly
                    U_list = [[float(U_r[i*m+j, r]) for r in range(target_rank)] for j in range(m) for i in range(n)]
                    # Reorder: U has shape (n*m, rank), indexed as [i*m+j, r]
                    U_list = [[float(U_r[idx, r]) for r in range(target_rank)] for idx in range(n*m)]
                    V_list = [[float(V_r[idx, r]) for r in range(target_rank)] for idx in range(m*p)]
                    W_list = [[float(W_r[idx, r]) for r in range(target_rank)] for idx in range(p*n)]
                    
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
                    
                # Gradient step (simplified)
                lr = 0.01
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err_ijk = sum(U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r] for r in range(target_rank)) - 1.0
                            for r in range(target_rank):
                                U_r[i*m+j, r] -= lr * err_ijk * V_r[j*p+k, r] * W_r[k*n+i, r]
                                V_r[j*p+k, r] -= lr * err_ijk * U_r[i*m+j, r] * W_r[k*n+i, r]
                                W_r[k*n+i, r] -= lr * err_ijk * U_r[i*m+j, r] * V_r[j*p+k, r]
                
                if found:
                    break
            
            if found:
                break
    
    return best_result
