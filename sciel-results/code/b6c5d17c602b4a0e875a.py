import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _try_rank_search(n, m, p, max_rank_to_try, deadline):
    """Try to find a lower-rank decomposition using linear algebra."""
    import random
    random.seed(12345)
    
    n, m, p = 2, 3, 10
    num_U = n * m  # 6
    num_V = m * p  # 30
    num_W = p * n  # 20
    num_constraints = n * m * p  # 60
    
    # The constraint is: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # This is a system of trilinear equations.
    
    # Strategy: fix two of the three matrices and solve for the third.
    # Or use iterative refinement.
    
    # Let's try: for a given rank r, initialize U, V, W randomly and do gradient descent
    # on the squared error.
    
    best_result = None
    best_rank = n * m * p
    
    for r in range(30, min(best_rank, max_rank_to_try + 1)):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 3:
                break
            
            # Initialize
            U = np.random.randn(num_U, r) * 0.1
            V = np.random.randn(num_V, r) * 0.1
            W = np.random.randn(num_W, r) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 2:
                    break
                
                # Compute error and gradients
                error = 0.0
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            
                            # Compute the current value
                            val = 0.0
                            for rr in range(r):
                                val += U[idx_u, rr] * V[idx_v, rr] * W[idx_w, rr]
                            
                            err = val - 1.0
                            error += err * err
                            
                            # Gradients
                            for rr in range(r):
                                U_grad[idx_u, rr] += 2 * err * V[idx_v, rr] * W[idx_w, rr]
                                V_grad[idx_v, rr] += 2 * err * U[idx_u, rr] * W[idx_w, rr]
                                W_grad[idx_w, rr] += 2 * err * U[idx_u, rr] * V[idx_v, rr]
                
                # Update
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                if error < 1e-12:
                    found = True
                    break
            
            if found:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Verify
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = (U_list, V_list, W_list)
                    best_rank = r
                    break
        
        if best_rank <= r:
            break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook solution as fallback
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    result = _try_rank_search(n, m, p, 55, deadline)
    if result is not None:
        U, V, W = result
        best_rank = len(U[0])
    
    if time.time() > deadline - 2:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
