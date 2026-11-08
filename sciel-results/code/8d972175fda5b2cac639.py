"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _round_half(x):
    """Round to nearest half-integer."""
    return round(x * 2) / 2

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Schoolbook rank
    rank_school = n * m * p  # 32
    
    # Try to find a lower rank decomposition using structured approaches
    
    # Approach 1: Use Strassen-like block decomposition
    # For 2x4x4, split into blocks of size 2x2
    # A is 2x4, B is 4x4, C is 2x4
    # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # Split B into B1 (2x4, rows 0-1) and B2 (2x4, rows 2-3)
    # C = A1*B1 + A2*B2
    # Each A_i * B_i is a 2x2 * 2x4 multiplication
    # The tensor for 2x2*2x4 has rank at most 2*2*4=16, but can be lower
    
    # Actually, let's try a direct algebraic construction for rank < 32
    
    # Method: Start with schoolbook and try to find linear dependencies
    # or use known low-rank decompositions
    
    # For now, let's try a systematic search with better initialization
    # using the structure of the problem
    
    best_rank = rank_school
    best_result = None
    
    # Initialize with schoolbook
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try ranks from 31 down to a reasonable lower bound
    # Lower bound: rank >= max(n*m, m*p, p*n) is not valid, but rank >= n+m+p-2 = 8 for square
    # For 2x4x4, known lower bound is at least 16 (by slicing arguments)
    
    # Use a more structured search: try to build decompositions using
    # combinations of simple multiplications
    
    # Strategy: Use the fact that we can combine terms
    # Try ranks 28, 24, 20, 16 with structured random search
    
    target_ranks = [28, 26, 24, 22, 20, 18, 16]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        # Try multiple random seeds with structured initialization
        for seed in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(seed * 7919 + target_rank)
            
            # Generate random U and V with values in {-1, -0.5, 0, 0.5, 1}
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            
            U_r = np.random.choice(vals, size=(n * m, target_rank))
            V_r = np.random.choice(vals, size=(m * p, target_rank))
            
            # For each (i, k), solve for W row
            W_r = np.zeros((p * n, target_rank))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, target_rank))
                    for j in range(m):
                        A[j, :] = U_r[i * m + j, :] * V_r[j * p + k, :]
                    
                    # Solve A @ W_col = ones(m)
                    try:
                        if target_rank == m:
                            W_col = np.linalg.solve(A, np.ones(m))
                        else:
                            W_col, residuals, rank_l, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            if target_rank < m:
                                if np.linalg.norm(A @ W_col - np.ones(m)) > 1e-8:
                                    valid = False
                                    break
                        
                        # Check if W_col is close to half-integers
                        for r in range(target_rank):
                            closest = _round_half(W_col[r])
                            if abs(W_col[r] - closest) > 1e-8:
                                valid = False
                                break
                            W_r[k * n + i, r] = closest
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    
                    if not valid:
                        break
            
            if valid:
                # Verify
                if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    break  # Found for this rank, try lower
        
        if best_rank < rank_school:
            # Found something better, continue to try lower ranks
            pass
        else:
            # If we've tried several ranks without success, stop
            if target_rank <= 20:
                break
    
    return best_result
# EVOLVE-BLOCK-END
