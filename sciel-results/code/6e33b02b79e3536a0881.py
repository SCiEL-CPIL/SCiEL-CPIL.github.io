"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40
    
    # Deadline guard
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    def check_deadline():
        return time.time() + safety_margin > deadline
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] for matrix mult (A*W)[k*n+i, j] = sum A[i*m+j, k]*W[k*n+i', j]... 
    # Actually: (A*W)[k*n+i, j] = sum over k' A[i*m+j, k'] * W[k'*n+i, j]
    # The tensor T[a, b, c] where a = i*m+j (row of A), b = j*p+k (col of A = row of W), c = k*n+i (row of result)
    # T[i*m+j, j*p+k, k*n+i] = 1 if these indices are consistent (same i,j,k)
    
    # For 2x4x5: A is 2x4, W is 4x5, result is 2x5
    # T has shape (2*4, 4*5, 5*2) = (8, 20, 10)
    # T[i*4+j, j*5+k, k*2+i] = 1 for i in [0,1], j in [0,3], k in [0,4]
    
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m + j, j*p + k, k*n + i] = 1.0
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as fallback
    U_sb = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    # Try to find lower rank decompositions
    for rank in range(30, rank_schoolbook):  # Try ranks 30 to 39
        if check_deadline():
            break
            
        # Multiple random restarts
        for restart in range(20):
            if check_deadline():
                break
                
            # Initialize randomly
            np.random.seed(rank * 1000 + restart)
            U_r = np.random.randn(n*m, rank)
            V_r = np.random.randn(m*p, rank)
            W_r = np.random.randn(p*n, rank)
            
            # Gradient descent on ||T - sum U[:,r] V[:,r]^T W[:,r]^T||^2
            # Using simple SGD-like approach
            lr = 0.01
            for step in range(2000):
                if check_deadline():
                    break
                    
                # Compute reconstruction
                recon = np.zeros_like(target)
                for r in range(rank):
                    recon += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
                
                err = recon - target
                loss = np.sum(err**2)
                
                if loss < 1e-10:
                    # Check exactness with rational approximation
                    break
                
                # Compute gradients
                grad_U = np.zeros_like(U_r)
                grad_V = np.zeros_like(V_r)
                grad_W = np.zeros_like(W_r)
                
                for r in range(rank):
                    # d/dU[r] = sum err * V[r] outer W[r]
                    grad_U[:, r] = np.einsum('abc,b->a', err, V_r[:, r]) * W_r[:, r]
                    grad_V[:, r] = np.einsum('abc,a->b', err, U_r[:, r]) * W_r[:, r]
                    grad_W[:, r] = np.einsum('abc,a->c', err, U_r[:, r]) * V_r[:, r]
                
                # Update
                U_r -= lr * grad_U
                V_r -= lr * grad_V
                W_r -= lr * grad_W
                
                if step % 100 == 0 and loss < 1e-8:
                    break
            
            # Check if we got a good solution
            recon = np.zeros_like(target)
            for r in range(rank):
                recon += np.einsum('a,b,c->abc', U_r[:, r], V_r[:, r], W_r[:, r])
            
            max_err = np.max(np.abs(recon - target))
            if max_err < 1e-6:
                # Try to find exact rational/integer representation
                # Scale to integers
                try:
                    # Find common denominator
                    all_vals = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
                    # Try to round to reasonable precision
                    scale = 1.0
                    for trial in [1, 2, 4, 8, 10, 20, 50, 100]:
                        U_s = np.round(U_r * trial)
                        V_s = np.round(V_r * trial)
                        W_s = np.round(W_r * trial)
                        # Check if this gives exact decomposition
                        recon_s = np.zeros_like(target)
                        for r in range(rank):
                            recon_s += np.einsum('a,b,c->abc', U_s[:, r], V_s[:, r], W_s[:, r])
                        if np.max(np.abs(recon_s - target)) < 1e-9:
                            U_r, V_r, W_r = U_s / trial, V_s / trial, W_s / trial
                            break
                except:
                    pass
                
                best_rank = rank
                best_U = U_r
                best_V = V_r
                best_W = W_r
                break  # Found a decomposition at this rank
        
        if best_rank < rank_schoolbook:
            break  # Found improvement, no need to try higher ranks
    
    # Convert to list-of-lists format with integer/half-integer entries
    if best_U is not None and best_rank < rank_schoolbook:
        # Ensure entries are integer or half-integer
        def clean_matrix(mat):
            mat = np.round(mat * 2) / 2  # Round to half-integers
            return mat.tolist()
        
        return {"U": clean_matrix(best_U), "V": clean_matrix(best_V), "W": clean_matrix(best_W)}
    else:
        # Fallback to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# EVOLVE-BLOCK-END
