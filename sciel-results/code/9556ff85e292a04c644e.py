"""Program for matrix multiplication 3x3x5 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for multiplying a 3x3 matrix by a 3x5 matrix.
    Attempts to find a low-rank decomposition via randomized search with
    verification, falling back to schoolbook if no better solution is found.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s to be safe, leaving 10s margin

    # Build the target tensor T
    # T[a, b, c] = 1 if exists i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook decomposition as guaranteed fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, r

    # Initialize best with schoolbook
    best_U, best_V, best_W, best_rank = schoolbook()

    # Function to verify decomposition
    def verify(U, V, W):
        # Compute product tensor: P[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
        P = np.einsum('ak,bk,ck->abc', U, V, W)
        err = np.max(np.abs(P - T))
        return err

    # Try to find a lower rank decomposition using randomized search
    # We will try ranks from 30 down to 25. If we find a valid one, we stop.
    
    # Optimization parameters
    lr = 0.05
    steps_per_attempt = 50
    max_attempts = 3
    
    # We iterate through target ranks. Since finding a valid decomposition is hard,
    # we might just stick to schoolbook if search is too slow.
    
    target_ranks = [30, 35]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, target_rank) * 0.1
            V = np.random.randn(m*p, target_rank) * 0.1
            W = np.random.randn(p*n, target_rank) * 0.1
            
            for step in range(steps_per_attempt):
                if time.time() > deadline - 10:
                    break
                
                # Compute P = einsum(U, V, W)
                P = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = P - T
                
                # Gradients
                grad_U = np.einsum('abc,bk,ck->ak', diff, V, W)
                grad_V = np.einsum('abc,ak,ck->bk', diff, U, W)
                grad_W = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                err = np.max(np.abs(diff))
                if err < 1e-6:
                    if verify(U, V, W) < 1e-6:
                        best_U, best_V, best_W, best_rank = U, V, W, target_rank
                        # Found a valid decomposition, return early
                        return {
                            "U": best_U.tolist(),
                            "V": best_V.tolist(),
                            "W": best_W.tolist()
                        }
                    break
            
            # If we found a good solution, update best
            err = verify(U, V, W)
            if err < 1e-6 and target_rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, target_rank

    # Return the best solution found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
