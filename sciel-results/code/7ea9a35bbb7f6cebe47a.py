import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Check if decomposition is exact with half-integer entries."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(len(U[0])))
                if abs(s - 1.0) > 1e-10:
                    return False
    # Check all entries are half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    deadline = time.time() + 140  # 140s safety margin
    
    # Candidate half-integer values (small set to keep search tractable)
    candidates = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: need at least max(n*m, m*p, p*n) / something, but let's start from 24
    # The trivial solution is rank 36. Let's try to find something better.
    
    for rank in range(24, 37):
        if time.time() > deadline - 5:
            break
        
        # Use structured initialization based on known patterns
        # For 3x3x3, Strassen uses rank 7. For 3x3x4, we can try extending.
        # Initialize with a pattern that has many zeros and small values.
        
        # Try multiple structured initializations
        for init_seed in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with candidate values
            # Use a deterministic pattern based on init_seed
            np.random.seed(42 + rank * 100 + init_seed)
            
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Fill with random candidate values, but keep many zeros
            for idx in range(n * m):
                for r in range(rank):
                    if np.random.random() < 0.3:
                        U[idx, r] = candidates[np.random.randint(len(candidates))]
            
            for idx in range(m * p):
                for r in range(rank):
                    if np.random.random() < 0.3:
                        V[idx, r] = candidates[np.random.randint(len(candidates))]
            
            for idx in range(p * n):
                for r in range(rank):
                    if np.random.random() < 0.3:
                        W[idx, r] = candidates[np.random.randint(len(candidates))]
            
            # Now do local search: try to fix the decomposition
            # Use coordinate descent on the half-integer values
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute current residual for each (i,j,k)
                max_improve = 0
                best_change = None
                
                # Try changing one entry at a time
                # This is expensive, so sample a subset
                for _ in range(50):
                    # Pick a random matrix, row, and column
                    mat_choice = np.random.randint(3)
                    if mat_choice == 0:
                        row_idx = np.random.randint(n * m)
                        col_idx = np.random.randint(rank)
                        old_val = U[row_idx, col_idx]
                    elif mat_choice == 1:
                        row_idx = np.random.randint(m * p)
                        col_idx = np.random.randint(rank)
                        old_val = V[row_idx, col_idx]
                    else:
                        row_idx = np.random.randint(p * n)
                        col_idx = np.random.randint(rank)
                        old_val = W[row_idx, col_idx]
                    
                    # Try all candidate values
                    for new_val in candidates:
                        if new_val == old_val:
                            continue
                        
                        # Compute how much this change improves the max residual
                        # This is expensive, so approximate
                        # For now, just accept if it seems to help
                        # We'll do a full check after each batch of changes
                        pass
                
                # Full residual check
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            err = abs(s - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-10:
                    # Check if all values are half-integers
                    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # If stuck, try a different initialization
                break
        
        # If we found a solution at this rank, return it
        # Otherwise, continue to next rank
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
