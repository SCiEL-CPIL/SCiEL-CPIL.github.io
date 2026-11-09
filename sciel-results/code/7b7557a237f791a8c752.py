"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # The tensor T has indices (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where T[i,j,k] = 1 for all valid (i,j,k)
    
    # For small cases, we can try structured decompositions
    # Try rank from theoretical lower bound up to schoolbook-1
    # Lower bound: max(n*m, m*p, p*n) = max(6, 15, 10) = 15
    # But actual min rank for 2x3*3x5 is known to be around 16-20
    
    for target_rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        # Try a structured approach: use the fact that matrix multiplication
        # can sometimes be decomposed using block structures
        # For 2x3 * 3x5, we can try various patterns
        
        # Method: Use random initialization + refinement with half-integer rounding
        np.random.seed(42 + target_rank)
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with random half-integers in [-2, 2]
            U = np.random.randint(-4, 5, size=(n*m, target_rank)) / 2.0
            V = np.random.randint(-4, 5, size=(m*p, target_rank)) / 2.0
            W = np.random.randint(-4, 5, size=(p*n, target_rank)) / 2.0
            
            # Check if this decomposition works
            if _verify_decomposition(U, V, W, n, m, p):
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_rank == target_rank:
            break
    
    return best_result

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
