import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    # Check all (i,j,k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    """
    Constructs the factor matrices for the matrix multiplication tensor.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 42
    
    # Initialize with trivial decomposition
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    # Try to find a lower rank decomposition via local search
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    # Candidate values: small set of integers and half-integers
    candidates = [-1, -0.5, 0, 0.5, 1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by attempting to merge terms
    # Strategy: try to find a decomposition of rank 41, 40, etc.
    for target_rank in range(rank - 1, max(rank - 5, 30), -1):
        if time.time() > deadline - 30:  # Stop if we're running out of time
            break
        
        # Try multiple random restarts for this target rank
        for restart in range(50):
            if time.time() > deadline - 30:
                break
            
            # Start from trivial and randomly zero out some entries
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Zero out the last (rank - target_rank) columns
            for c in range(target_rank, rank):
                U_try[:, c] = 0
                V_try[:, c] = 0
                W_try[:, c] = 0
            
            # Randomly perturb the remaining columns
            for _ in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Pick a random column and random entry to perturb
                col = random.randint(0, target_rank - 1)
                for matrix, rows in [(U_try, n*m), (V_try, m*p), (W_try, p*n)]:
                    row = random.randint(0, rows - 1)
                    old_val = matrix[row, col]
                    new_val = random.choice(candidates)
                    matrix[row, col] = new_val
                    
                    # Check if still valid
                    if verify_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-6):
                        # Found a valid decomposition!
                        # Clean up to exact values
                        if verify_decomposition(U_try, V_try, W_try, n, m, p, tol=1e-9):
                            best_rank = target_rank
                            best_U = U_try.copy()
                            best_V = V_try.copy()
                            best_W = W_try.copy()
                            # Convert to integers if possible
                            for mat in [best_U, best_V, best_W]:
                                mat[abs(mat) < 1e-9] = 0
                                mat[abs(mat - 0.5) < 1e-9] = 0.5
                                mat[abs(mat + 0.5) < 1e-9] = -0.5
                                mat[abs(mat - 1) < 1e-9] = 1
                                mat[abs(mat + 1) < 1e-9] = -1
                            return {
                                "U": best_U.tolist(),
                                "V": best_V.tolist(),
                                "W": best_W.tolist()
                            }
                    else:
                        # Revert
                        matrix[row, col] = old_val
    
    # Return the best found (likely the trivial one)
    U_out = best_U.copy()
    V_out = best_V.copy()
    W_out = best_W.copy()
    
    # Clean up
    for mat in [U_out, V_out, W_out]:
        mat[abs(mat) < 1e-9] = 0
        mat[abs(mat - 0.5) < 1e-9] = 0.5
        mat[abs(mat + 0.5) < 1e-9] = -0.5
        mat[abs(mat - 1) < 1e-9] = 1
        mat[abs(mat + 1) < 1e-9] = -1
    
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist()
    }
