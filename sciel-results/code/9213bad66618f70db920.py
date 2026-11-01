import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i, j, k
    # This is the standard multiplication tensor with W indexed by k*n+i.
    
    # Initialize with schoolbook rank 27
    rank_27 = n * m * p
    U_27 = np.zeros((n * m, rank_27))
    V_27 = np.zeros((m * p, rank_27))
    W_27 = np.zeros((p * n, rank_27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_27[i * m + j, col] = 1
                V_27[j * p + k, col] = 1
                W_27[k * n + i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition
    # We will try ranks from 20 to 26
    best_U, best_V, best_W = U_27, V_27, W_27
    best_rank = rank_27
    
    # Time limit
    start_time = time.time()
    time_limit = 140  # Leave 10s for overhead
    
    for rank in range(20, 27):
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a rank 'rank' decomposition
        # Use random initialization and alternating least squares
        success = False
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize to prevent divergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Alternating Least Squares
            for it in range(100):
                # Solve for U
                # Reshape V, W to tensors
                # T_target is sparse, we can compute the residual
                # But for ALS, we need to minimize || T - sum U_r V_r W_r ||
                
                # Efficiently, we can use the fact that T is sparse.
                # However, for small sizes, we can just compute the full tensor.
                # T shape: (9, 9, 9)
                
                # Compute current tensor
                # T_current[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # We want T_current to match T_target.
                
                # T_target[i*m+j, j*p+k, k*n+i] = 1
                # Let's flatten the indices for easier handling?
                # No, just iterate.
                
                # Solve for U:
                # Fix V, W.
                # Minimize sum_{i,j,k} (T_target[i,j,k] - U[i,r] V[j,r] W[k,r])^2
                # This is a linear system for U.
                
                # Construct the linear system for U.
                # For each i, the equation is:
                # sum_{j,k,r} T_target[i,j,k] V[j,r] W[k,r] U[i,r] ... no.
                
                # Let's use a simpler approach: compute the gradient or use a library.
                # Since n,m,p are small, we can use a simple iterative method.
                
                # Update U:
                # U_new = argmin || T_target - sum_r U[:,r] outer(V[:,r], W[:,r]) ||
                
                # This is equivalent to solving a linear system.
                # Let A be a matrix of size (n*m) x (rank * m*p * p*n)? No.
                
                # Let's just use a simple gradient step or ALS with full tensor construction.
                # Full tensor T is 9x9x9 = 729 elements.
                # T_target is fixed.
                
                # Compute T_target
                if not hasattr(construct, 'T_target'):
                    T_target = np.zeros((n*m, m*p, p*n))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_target[i*m+j, j*p+k, k*n+i] = 1.0
                    construct.T_target = T_target
                
                # Update U
                # Reshape V, W to (rank, m*p, p*n)? No.
                # Term r is U[:,r] (n*m) * V[:,r] (m*p) * W[:,r] (p*n)
                # This is an outer product of three vectors.
                # We can't easily do ALS for 3-way tensors without unfolding.
                
                # Unfold T along mode 1:
                # T_{i, (j,k)} = sum_r U_{i,r} (V_{j,r} W_{k,r})
                # Let M_r = outer(V[:,r], W[:,r]) flattened to (m*p*p*n)? No.
                # V is (m*p, rank), W is (p*n, rank).
                # The product V[:,r] W[:,r] is not a simple vector product.
                
                # Actually, the tensor is T_{i, j, k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
                # This is a standard CP decomposition.
                # We can use ALS by unfolding.
                
                # Unfold T along mode 1 (i):
                # T_{(1)} is (n*m) x (m*p * p*n)
                # T_{(1)}[i, idx_vw] = sum_r U[i,r] (V[r, j] W[r, k]) where idx_vw corresponds to (j,k)
                # Wait, V is indexed by j*p+k? No, V is (m*p, rank).
                # V[j*p+k, r].
                # W[k*n+i, r].
                
                # Let's check the indices again.
                # U: (n*m, rank). Index i*m+j.
                # V: (m*p, rank). Index j*p+k.
                # W: (p*n, rank). Index k*n+i.
                
                # The tensor is T_{i*m+j, j*p+k, k*n+i}.
                # This is NOT a standard CP decomposition of a 3-way tensor with dimensions (n*m, m*p, p*n).
                # It is a specific sparse tensor.
                
                # However, we can still use ALS if we construct the linear systems correctly.
                # For U:
                # Fix V, W.
                # We want to minimize || T_target - sum_r U[:,r] V[:,r] W[:,r] ||_F
                # where the product is defined by the index mapping.
                
                # Let's define a linear operator L such that L(U) = T_current.
                # Then we solve (L^T L) U = L^T T_target.
                
                # Since the size is small (9x9x9), we can just compute the full tensor and use a simple gradient descent or least squares.
                
                # Compute T_current
                T_current = np.zeros((n*m, m*p, p*n))
                for r in range(rank):
                    # Outer product
                    # U[:,r] is (n*m,)
                    # V[:,r] is (m*p,)
                    # W[:,r] is (p*n,)
                    # But the indices are coupled!
                    # T_current[i*m+j, j*p+k, k*n+i] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    
                    # This is not a simple outer product of the vectors U[:,r], V[:,r], W[:,r].
                    # Because the index j in U is coupled to j in V, and k in V is coupled to k in W, and i in W is coupled to i in U.
                    
                    # So we cannot use standard CP-ALS.
                    # We must use a custom update.
                    
                    # For a fixed r, the term is:
                    # sum_{i,j,k} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] e_{i*m+j} e_{j*p+k} e_{k*n+i}
                    
                    # This is a specific sparse structure.
                    
                    # To update U, we fix V, W.
                    # The residual is R = T_target - T_current.
                    # We want to update U to reduce ||R||.
                    # The gradient with respect to U[i*m+j, r] is:
                    # sum_{j,k} R[i*m+j, j*p+k, k*n+i] V[j*p+k, r] W[k*n+i, r]
                    
                    # This is a linear system for U.
                    
                    # Let's construct the matrix A for the least squares problem.
                    # A is (n*m) x (n*m) ? No, U has (n*m, rank).
                    # We solve for U[:,r] for each r? No, they are coupled in the sum.
                    
                    # Actually, we can solve for the whole U matrix at once.
                    # Let U_flat be (n*m * rank,).
                    # The problem is linear in U.
                    
                    # We can construct the normal equations.
                    # Given the small size, we can use a simple iterative method or just compute the gradient and do a step.
                    
                    # Let's do a simple gradient descent step.
                    # Compute Gradient for U
                    grad_U = np.zeros_like(U)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_U = i*m+j
                                idx_V = j*p+k
                                idx_W = k*n+i
                                R_val = T_target[idx_U, idx_V, idx_W] - T_current[idx_U, idx_V, idx_W]
                                for r in range(rank):
                                    grad_U[idx_U, r] -= 2 * R_val * V[idx_V, r] * W[idx_W, r]
                                    
                    # Update U
                    lr = 0.01
                    U -= lr * grad_U
                    
                    # Similarly for V and W
                    grad_V = np.zeros_like(V)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_U = i*m+j
                                idx_V = j*p+k
                                idx_W = k*n+i
                                R_val = T_target[idx_U, idx_V, idx_W] - T_current[idx_U, idx_V, idx_W]
                                for r in range(rank):
                                    grad_V[idx_V, r] -= 2 * R_val * U[idx_U, r] * W[idx_W, r]
                                    
                    V -= lr * grad_V
                    
                    grad_W = np.zeros_like(W)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx_U = i*m+j
                                idx_V = j*p+k
                                idx_W = k*n+i
                                R_val = T_target[idx_U, idx_V, idx_W] - T_current[idx_U, idx_V, idx_W]
                                for r in range(rank):
                                    grad_W[idx_W, r] -= 2 * R_val * U[idx_U, r] * V[idx_V, r]
                                    
                    W -= lr * grad_W
                    
                    # Check convergence
                    err = np.linalg.norm(T_target - T_current)
                    if err < 1e-6:
                        # Check if it can be rounded to integers/half-integers
                        # This is a strong condition.
                        # We might need to polish.
                        pass
                        
            # After iterations, check if we found a solution
            T_final = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_U = i*m+j
                            idx_V = j*p+k
                            idx_W = k*n+i
                            T_final[idx_U, idx_V, idx_W] += U[idx_U, r] * V[idx_V, r] * W[idx_W, r]
                            
            err = np.linalg.norm(T_target - T_final)
            if err < 1e-4:
                # Try to round
                # This is risky.
                pass
                
    # If no better solution found, return schoolbook
    return {
        "U": U_27.tolist(),
        "V": V_27.tolist(),
        "W": W_27.tolist()
    }
