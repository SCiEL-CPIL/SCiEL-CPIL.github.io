"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank: n*m*p = 54
    # We'll try to find a better decomposition using optimization
    # The tensor for matrix multiplication M_{n,m,p} has entries T[i,j,k] 
    # where the multiplication is (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # For 3x3 * 3x6: A is 3x3, B is 3x6, result is 3x6
    # T is a 3-index tensor with indices (i,j,k) where i in [0,3), j in [0,3), k in [0,6)
    
    # The tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Let's use a randomized optimization approach to find low-rank decompositions
    # We'll use gradient descent on the entries
    
    def tensor_entry(i, j, k):
        """The matrix multiplication tensor entry: 1 if (i,j,k) corresponds to a term in the product, 0 otherwise.
        Actually for matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k) in the sum.
        Wait, let me reconsider. The tensor M_{n,m,p} has entries M_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
        The decomposition is: M_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        """
        return 1
    
    # Actually, the matrix multiplication tensor has all entries equal to 1.
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    
    # Let's try rank 33 first (known lower bound for 3x3x3 is 19, but 3x3x6 might be around 30-40)
    # Actually, for 3x3x6, a simple block decomposition gives 2 * rank(3x3x3) 
    # If rank(3x3x3) = 19 (Strassen-like), then 2*19 = 38
    
    # Let's try to find a decomposition of rank r using optimization
    best_rank = n * m * p  # schoolbook = 54
    best_decomp = None
    
    # Try different ranks from low to high
    for target_rank in [30, 33, 36, 39, 42, 45, 48, 51, 54]:
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        found = False
        for trial in range(5):
            if time.time() > deadline - 5:
                    break
            
            random.seed(42 + trial)
            np.random.seed(42 + trial)
            
            r = target_rank
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Use gradient descent to minimize the error
            lr = 0.01
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                # Compute the decomposed tensor
                # For each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                error = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            
                            # Compute the product
                            prod = U[idx_u, :] * V[idx_v, :] * W[idx_w, :]
                            val = np.sum(prod)
                            err = val - 1.0
                            error += err * err
                            
                            # Gradients
                            grad_U[idx_u, :] += 2 * err * V[idx_v, :] * W[idx_w, :]
                            grad_V[idx_v, :] += 2 * err * U[idx_u, :] * W[idx_w, :]
                            grad_W[idx_w, :] += 2 * err * U[idx_u, :] * V[idx_v, :]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if error < 1e-10:
                    found = True
                    break
                
                # Adaptive learning rate
                if step % 100 == 99:
                    if error < 1e-8:
                        lr = max(0.001, lr * 0.5)
                    elif error > 1.0:
                        lr = min(0.1, lr * 1.5)
            
            if found:
                # Convert to list of lists with integer/half-integer entries if possible
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded decomposition
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.sum(U_rounded[idx_u, :] * V_rounded[idx_v, :] * W_rounded[idx_w, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    best_rank = r
                    best_decomp = (U_rounded, V_rounded, W_rounded)
                    break
        
        if found:
            break
    
    # If we found a better decomposition, use it
    if best_decomp is not None and best_rank < n * m * p:
        U, V, W = best_decomp
        rank = best_rank
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
    
    # Convert to list of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
