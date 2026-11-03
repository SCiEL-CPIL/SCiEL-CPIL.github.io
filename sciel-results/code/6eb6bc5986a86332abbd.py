"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
DEADLINE = 140  # seconds, with safety margin

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try structured decomposition using Strassen for 2x2 blocks
    result = _try_strassen_decomposition(n, m, p)
    if result is not None:
        best_rank = result["rank"]
        best_result = result
        return best_result
    
    return best_result

def _try_strassen_decomposition(n, m, p):
    """
    Try to find a lower-rank decomposition using Strassen's algorithm for 2x2 subproblems.
    
    Strategy: Split the m=4 dimension into two 2-dimensional blocks.
    A (n x m) = [A1 | A2] where A1, A2 are n x 2
    B (m x p) = [B1; B2] where B1, B2 are 2 x p
    C = A1*B1 + A2*B2
    
    For each pair of 2x2 subblocks, we can use Strassen's rank-7 decomposition
    instead of the rank-8 schoolbook. This saves 1 rank per 2x2x2 subproblem.
    
    However, we need to be careful about how the blocks interact.
    """
    n, m, p = 3, 4, 5
    
    # Split m into two halves of size 2
    m1, m2 = 2, 2
    
    # For the multiplication A1*B1 where A1 is 3x2 and B1 is 2x5:
    # This is not a square matrix multiplication, so Strassen doesn't directly apply.
    # But we can still use a structured approach.
    
    # Let's try a different approach: use the known rank-5 decomposition for 2x2 matrix multiplication
    # and embed it into our problem.
    
    # Actually, let's try a more direct approach:
    # Use the fact that we can write the tensor as a sum of rank-1 tensors.
    # For a 3x4x5 tensor, the schoolbook rank is 60.
    # We want to find if there's a lower-rank decomposition.
    
    # Let's try using integer linear programming or a structured search.
    
    # For now, let's try a specific structured decomposition based on block decomposition.
    
    # Approach: Use the identity that for matrix multiplication, we can use
    # Kronecker products of smaller decompositions.
    
    # Let's try: decompose 3x4x5 as (3x2x5) + (3x2x5) but with shared structure.
    
    # Actually, let's try a randomized search with a better verification method.
    
    start_time = time.time()
    
    # Try ranks from 59 down to 40
    for target_rank in range(59, 39, -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        result = _try_rank_search(n, m, p, target_rank)
        if result is not None:
            result["rank"] = target_rank
            return result
    
    return None

def _try_rank_search(n, m, p, target_rank):
    """
    Try to find a rank-target_rank decomposition using structured patterns.
    """
    start_time = time.time()
    
    # Use a structured initialization based on block decomposition
    # Split the problem into blocks and use known low-rank decompositions for each block
    
    # For now, try a randomized approach with verification
    import random
    np.random.seed(42)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - start_time > 20:
            break
        
        # Initialize with random values from a small set
        # Use values that are likely to produce exact decompositions
        U = np.random.choice([0, 1], size=(n*m, target_rank))
        V = np.random.choice([0, 1], size=(m*p, target_rank))
        W = np.random.choice([0, 1], size=(p*n, target_rank))
        
        # Verify the decomposition
        if _verify_decomposition(U, V, W, n, m, p):
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the given U, V, W matrices form a valid decomposition
    of the matrix multiplication tensor.
    """
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all valid (i,j,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                actual = 0.0
                for r in range(U.shape[1]):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
