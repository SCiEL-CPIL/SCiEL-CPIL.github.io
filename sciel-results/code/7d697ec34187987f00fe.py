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
    
    # Build the target tensor
    # T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    
    # Schoolbook: rank = n*m*p = 36
    # Each term corresponds to one (i,j,k)
    
    best_rank = n * m * p
    best_U, best_V, best_W = None, None, None
    
    # First, create schoolbook solution
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
    
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    
    # Now try to reduce rank using structured low-rank approaches
    # The matrix multiplication tensor for 2x3x6 has special structure.
    # Let's try to find a lower rank decomposition using algebraic methods.
    
    # For 2x3x6, the tensor can be viewed as:
    # C[k*n+i, j*p+k] = sum_j A[i*m+j, j*p+k] * B[j*p+k, ...]
    # Wait, let me re-examine the indices.
    
    # A is 2x3, B is 3x6, C is 2x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In our tensor representation:
    # a = i*m + j = i*3 + j (row i, column j of A)
    # b = j*p + k = j*6 + k (row j, column k of B)
    # c = k*n + i = k*2 + i (column k, row i of C)
    
    # So T[a,b,c] = 1 iff there exist i,j,k such that:
    # a = 3i + j, b = 6j + k, c = 2k + i
    # This means: i = a//3, j = a%3, j = b//6, k = b%6, k = c//2, i = c%2
    
    # The constraint is: a%3 == b//6 and b%6//2 == c//2 and c%2 == a//3
    # More precisely: j = a%3 = b//6, k = b%6, i = a//3 = c%2, and k//2 = c//2
    
    # Let me enumerate all valid triples
    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))
    
    # Try to find a lower rank decomposition
    # Strategy: use the fact that for 2x3x6, we can potentially use
    # a structured decomposition based on the dimensions.
    
    # For 2x3x6, one approach is to use the "laser method" or
    # structured low-rank decompositions.
    
    # Let's try rank 24, 21, 18, etc.
    
    # Actually, for 2x3x6, the known lower bound is around 18-24.
    # Let me try a random initialization with gradient descent.
    
    if time.time() < deadline:
        # Try several target ranks
        for target_rank in [24, 21, 18, 16, 15, 14, 13, 12]:
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(42 + target_rank)
            U_rand = np.random.randn(num_A, target_rank)
            V_rand = np.random.randn(num_B, target_rank)
            W_rand = np.random.randn(num_C, target_rank)
            
            # Gradient descent to minimize the residual
            # Residual = sum over all (a,b,c) of (sum_r U[a,r]*V[b,r]*W[c,r] - T[a,b,c])^2
            
            # This is a non-convex problem, so we use multiple restarts
            found = False
            for restart in range(5):
                if time.time() > deadline:
                    break
                    
                U_curr = np.random.randn(num_A, target_rank) * 0.1
                V_curr = np.random.randn(num_B, target_rank) * 0.1
                W_curr = np.random.randn(num_C, target_rank) * 0.1
                
                # Learning rate schedule
                lr = 0.01
                for step in range(200):
                    if time.time() > deadline:
                        break
                    
                    # Compute residual
                    # T[a,b,c] for all a,b,c
                    # This is expensive, so let's only compute for valid triples
                    # and a sample of invalid ones
                    
                    # For efficiency, compute the gradient using the chain rule
                    # d/dU[a,r] = sum_{b,c} V[b,r]*W[c,r]*(sum_r' U[a,r']*V[b,r']*W[c,r'] - T[a,b,c])
                    
                    # This is O(num_A * num_B * num_C * target_rank) which is too expensive
                    # Let's use a simpler approach: just evaluate on valid triples
                    
                    # Compute current values on valid triples
                    residual_sum = 0
                    grad_U = np.zeros_like(U_curr)
                    grad_V = np.zeros_like(V_curr)
                    grad_W = np.zeros_like(W_curr)
                    
                    for (a, b, c) in valid_triples:
                        # Current value
                        val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                        err = val - 1.0
                        residual_sum += err * err
                        
                        # Gradient contributions
                        grad_U[a, :] += err * V_curr[b, :] * W_curr[c, :]
                        grad_V[b, :] += err * U_curr[a, :] * W_curr[c, :]
                        grad_W[c, :] += err * U_curr[a, :] * V_curr[b, :]
                    
                    # Also check some invalid positions (sample)
                    # For invalid (a,b,c), we want the sum to be 0
                    # Let's check a few
                    np.random.seed(restart * 100 + step)
                    for _ in range(50):
                        a = np.random.randint(0, num_A)
                        b = np.random.randint(0, num_B)
                        c = np.random.randint(0, num_C)
                        if (a, b, c) not in valid_triples:
                            val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                            err = val - 0.0
                            residual_sum += err * err * 0.1  # Weight invalid positions less
                            grad_U[a, :] += err * 0.1 * V_curr[b, :] * W_curr[c, :]
                            grad_V[b, :] += err * 0.1 * U_curr[a, :] * W_curr[c, :]
                            grad_W[c, :] += err * 0.1 * U_curr[a, :] * V_curr[b, :]
                    
                    # Update
                    U_curr -= lr * grad_U
                    V_curr -= lr * grad_V
                    W_curr -= lr * grad_W
                    
                    # Decay learning rate
                    lr *= 0.99
                    
                    if residual_sum < 1e-8:
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                # Verify the solution
                max_err = 0
                for (a, b, c) in valid_triples:
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                
                # Check invalid positions
                for a in range(num_A):
                    for b in range(num_B):
                        for c in range(num_C):
                            if (a, b, c) not in valid_triples:
                                val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                                max_err = max(max_err, abs(val))
                
                if max_err < 1e-6:
                    best_rank = target_rank
                    best_U = U_curr.tolist()
                    best_V = V_curr.tolist()
                    best_W = W_curr.tolist()
                    break
    
    # Return the best solution found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
