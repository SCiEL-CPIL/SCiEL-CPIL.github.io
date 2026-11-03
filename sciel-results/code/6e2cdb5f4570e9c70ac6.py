import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    # Build the target tensor for 2x3x6 multiplication
    # T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i

    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))

    # Schoolbook solution: rank = n*m*p = 36
    best_rank = n * m * p
    rank = best_rank
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to find lower rank decompositions
    # We will try ranks 30, 31, 32, 33, 34, 35
    # For each rank, we will try multiple random initializations and use gradient descent
    
    for target_rank in [30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(12345 + target_rank * 100 + restart)
            U_curr = np.random.randn(num_A, target_rank) * 0.5
            V_curr = np.random.randn(num_B, target_rank) * 0.5
            W_curr = np.random.randn(num_C, target_rank) * 0.5
            
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline - 2:
                    break
                    
                # Compute gradients
                grad_U = np.zeros_like(U_curr)
                grad_V = np.zeros_like(V_curr)
                grad_W = np.zeros_like(W_curr)
                residual_sum = 0
                
                # Precompute UV and VW for efficiency? 
                # UV is (num_A, num_B, rank) which is 6x18x30 = 3240 elements.
                # This is small enough to compute directly.
                
                for (a, b, c) in valid_triples:
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    err = val - 1.0
                    residual_sum += err * err
                    
                    # Gradient: d/da = V[b,r]*W[c,r]*err
                    for r in range(target_rank):
                        grad_U[a, r] += err * V_curr[b, r] * W_curr[c, r]
                        grad_V[b, r] += err * U_curr[a, r] * W_curr[c, r]
                        grad_W[c, r] += err * U_curr[a, r] * V_curr[b, r]
                
                # Update
                U_curr -= lr * grad_U
                V_curr -= lr * grad_V
                W_curr -= lr * grad_W
                lr *= 0.99
                
                if residual_sum < 1e-8:
                    found = True
                    break
            
            if found:
                # Verify
                max_err = 0
                for (a, b, c) in valid_triples:
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                    break
        
        if found:
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
