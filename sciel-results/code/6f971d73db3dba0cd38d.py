"""Rewritten program: rank-minimal search for 3x3 * 3x5 matrix multiplication."""
import numpy as np
import itertools
import time
import sys

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Schoolbook solution as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Build the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We want to decompose this tensor as sum of rank-1 terms
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Try to find a lower rank decomposition using random initialization
    # and iterative refinement (similar to tensor decomposition / CP decomposition)
    
    # The tensor T has shape (n*m, m*p, p*n) = (9, 15, 15)
    # T[A, B, C] where A = i*m+j, B = j*p+k, C = k*n+i
    
    # Build the full tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                T[A, B, C] = 1.0
    
    # Try various ranks from low to high
    # Known lower bound: rank >= max(n*m, m*p, p*n) = 15
    # Upper bound: n*m*p = 45
    # Let's try to find if we can do better than 45
    
    # Use a randomized approach: initialize random U, V, W and try to fit
    # We'll use a simple iterative method
    
    np.random.seed(42)
    
    for rank in range(15, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        max_iters = 200
        n_trials = 5
        
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Iterative refinement: minimize ||T - sum_c U[:,c] V[:,c] W[:,c]||_F
            # Use a simple gradient-free approach: alternating least squares
            
            converged = False
            for it in range(max_iters):
                if time.time() > deadline - 3:
                    break
                    
                # Update U: for each c, U[:,c] = argmin ||T - sum U[:,c] V[:,c] W[:,c]||
                # This is a linear least squares problem
                # Residual R = T - sum_{c'} U[:,c'] V[:,c'] W[:,c']
                # For fixed c, we want to minimize ||R + U[:,c] V[:,c] W[:,c]||
                
                # Compute current tensor
                current = np.zeros_like(T)
                for c in range(rank):
                    current += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                
                residual = T - current
                
                # Check convergence
                if np.linalg.norm(residual) < 1e-10:
                    converged = True
                    break
                
                # Update each factor
                for c in range(rank):
                    # Remove current contribution
                    current -= np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                    residual = T - current
                    
                    # Solve for U[:,c] given V[:,c] and W[:,c]
                    # We want to minimize ||residual + U[:,c] V[:,c] W[:,c]||_F
                    # This is equivalent to solving a least squares problem
                    
                    # Reshape residual to 2D: (n*m, (m*p)*(p*n))
                    # The update direction for U[:,c] is related to contracting residual with V[:,c] and W[:,c]
                    
                    # Contract residual with V[:,c] and W[:,c] to get a vector of length n*m
                    # grad_U = sum_{B,C} residual[A,B,C] * V[B,c] * W[C,c]
                    grad_U = np.einsum('abc,b,c->a', residual, V[:, c], W[:, c])
                    
                    # Similarly for V and W
                    grad_V = np.einsum('abc,a,c->b', residual, U[:, c], W[:, c])
                    grad_W = np.einsum('abc,a,b->c', residual, U[:, c], V[:, c])
                    
                    # Use a simple step with learning rate
                    lr = 0.01
                    U[:, c] += lr * grad_U
                    V[:, c] += lr * grad_V
                    W[:, c] += lr * grad_W
                    
                    # Add back
                    current += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                
                if time.time() > deadline - 2:
                    break
            
            if converged:
                # Verify exact decomposition
                final_tensor = np.zeros_like(T)
                for c in range(rank):
                    final_tensor += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
                
                if np.allclose(final_tensor, T, atol=1e-9):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    break
        
        if best_rank == rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
