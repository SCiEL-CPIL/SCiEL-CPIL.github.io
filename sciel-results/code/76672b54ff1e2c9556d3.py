"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    # Build the tensor: T[i,j,k] = 1 if (i,j,k) corresponds to a valid
    # (row, col, output) triple, 0 otherwise
    # Using trace convention: W[k*n + i, r]
    # For matrix mult: C[i,k] = sum_j A[i,j] * B[j,k]
    # So tensor entry T[i, j, k] = 1 for all valid (i,j,k)
    # U index: i*m + j (size n*m)
    # V index: j*p + k (size m*p)  
    # W index: k*n + i (size p*n)

    # Naive decomposition: rank = n*m*p = 24
    rank_naive = n * m * p
    U = np.zeros((nm, rank_naive))
    V = np.zeros((mp, rank_naive))
    W = np.zeros((pn, rank_naive))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1

    def check_and_return(U, V, W, tol=1e-6):
        """Check decomposition validity and return dict if valid."""
        rank = U.shape[1]
        # Vectorized check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a] * V[b] * W[c])
                    if abs(val - 1.0) > tol:
                        return None
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions using structured approaches
    # For 2x3 x 3x4, known results suggest rank can be less than 24
    
    # Strategy: Use bilinear optimization with better initialization
    # Try ranks from naive down to a lower bound
    
    lower_bound = max(n*m, m*p, p*n)  # Basic lower bound
    # Better lower bound: max(n*p, n*m, m*p) but for bilinear rank
    # Actually for matrix mult, rank >= max(n*p, n*m, m*p) is not tight
    # Use rank >= n*p as a simple bound? No, that's not right either.
    # The tensor rank of matrix multiplication is at least max(n*m, m*p, n*p)
    # but this is not a proven lower bound. Let's just try from naive down.
    
    best_result = check_and_return(U, V, W)
    best_rank = rank_naive
    
    # Try to reduce rank using iterative optimization
    # Use a smarter approach: try to find rank r decomposition by
    # solving the system of equations
    
    def try_rank(target_rank, max_iterations=200, seed_base=0):
        """Try to find a rank-target_rank decomposition using gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with structured pattern
        U_try = np.zeros((nm, target_rank))
        V_try = np.zeros((mp, target_rank))
        W_try = np.zeros((pn, target_rank))
        
        # Random initialization
        np.random.seed(seed_base)
        U_try = np.random.randn(nm, target_rank) * 0.3
        V_try = np.random.randn(mp, target_rank) * 0.3
        W_try = np.random.randn(pn, target_rank) * 0.3
        
        # Gradient descent with vectorized operations
        lr = 0.05
        for step in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Compute error and gradients vectorized
            grad_U = np.zeros_like(U_try)
            grad_V = np.zeros_like(V_try)
            grad_W = np.zeros_like(W_try)
            
            # For each (i,j,k) triple
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U_try[a] * V_try[b] * W_try[c])
                        err = val - 1.0
                        grad_U[a] += err * (V_try[b] * W_try[c])
                        grad_V[b] += err * (U_try[a] * W_try[c])
                        grad_W[c] += err * (U_try[a] * V_try[b])
            
            U_try -= lr * grad_U
            V_try -= lr * grad_V
            W_try -= lr * grad_W
            
            # Check convergence
            max_grad = max(np.max(np.abs(grad_U)), np.max(np.abs(grad_V)), np.max(np.abs(grad_W)))
            if max_grad < 1e-7:
                break
        
        # Check if valid
        result = check_and_return(U_try, V_try, W_try)
        return result

    # Try ranks from naive down to lower bound
    for target_rank in range(rank_naive - 1, lower_bound - 1, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            result = try_rank(target_rank, max_iterations=100, seed_base=42 + attempt * 17)
            if result is not None:
                best_rank = target_rank
                best_result = result
                found = True
                break
        
        if found:
            # Try to go even lower
            continue
    
    if best_result is None:
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
