import numpy as np
import time
import random

def construct():
    n, m, p = 3, 4, 4
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Target: find rank r such that there exist U (n*m, r), V (m*p, r), W (p*n, r)
    # with integer/half-integer entries where:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # The trivial solution has rank n*m*p = 48
    best_rank = n * m * p
    best_solution = None
    
    def make_trivial_solution(rank):
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W
    
    U, V, W = make_trivial_solution(best_rank)
    best_solution = (U, V, W)
    
    # Try to find lower rank solutions
    # Lower bound: max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound up to best_rank - 1
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            random.seed(rank * 1000 + attempt)
            np.random.seed(rank * 1000 + attempt)
            
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Gradient descent to minimize error
            # Error = sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 10:
                    break
                    
                # Compute error and gradients
                error = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            
                            # Compute the sum
                            s = 0.0
                            for c in range(rank):
                                s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                            
                            err = s - 1.0
                            error += err * err
                            
                            # Gradients
                            for c in range(rank):
                                grad_U[u_idx, c] += 2 * err * V[v_idx, c] * W[w_idx, c]
                                grad_V[v_idx, c] += 2 * err * U[u_idx, c] * W[w_idx, c]
                                grad_W[w_idx, c] += 2 * err * U[u_idx, c] * V[v_idx, c]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if error < 1e-12:
                    break
            
            # Check if we found a valid solution
            if error < 1e-10:
                # Round to nearest half-integer
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded solution
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            s = 0.0
                            for c in range(rank):
                                s += U_rounded[u_idx, c] * V_rounded[v_idx, c] * W_rounded[w_idx, c]
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-10:
                    best_rank = rank
                    best_solution = (U_rounded, V_rounded, W_rounded)
                    break  # Found valid solution at this rank, try lower
    
    # Return the best solution found
    if best_solution is not None:
        U, V, W = best_solution
    else:
        U, V, W = make_trivial_solution(best_rank)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
