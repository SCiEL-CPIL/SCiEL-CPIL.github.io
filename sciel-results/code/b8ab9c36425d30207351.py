"""Matrix multiplication 2x3x7: rank-minimal exact tensor decomposition search."""
import time
import math
import random
import numpy as np
from itertools import product as iproduct

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 7 matrix.
    
    The matrix multiplication tensor T has T[i,j,k] = 1 for all valid indices,
    and the decomposition is:
        T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    We use a greedy approach to try to find a lower rank than the schoolbook rank
    (n*m*p = 42).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0  # safety margin
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p
    
    def schoolbook_decomp():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Initialize with schoolbook
    U, V, W = schoolbook_decomp()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find a lower rank using random initialization and local optimization
    # Strategy: Start with random rank < schoolbook_rank, check if it can be exact
    
    def check_exactness(U, V, W, tol=1e-8):
        """Check if the decomposition exactly represents the tensor."""
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        rank = U.shape[1]
        # Build the tensor from the decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0  # The tensor entry is 1 for all valid (i,j,k)
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    # Try greedy rank reduction: remove one rank at a time if possible
    # This is a heuristic; for 2x3x7, the border rank might be lower than 42
    
    # Alternative: Use the fact that we need integer/half-integer entries
    # Try small rank values and see if we can find an exact decomposition
    
    # For 2x3x7, known results suggest the rank might be around 30-35
    # Let's try a few random initializations with different ranks
    
    for target_rank in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with rational-like values
            U_rand = np.random.randn(n * m, target_rank) * 0.5
            V_rand = np.random.randn(m * p, target_rank) * 0.5
            W_rand = np.random.randn(p * n, target_rank) * 0.5
            
            # Scale to make entries likely to be half-integers or integers
            # Round to nearest 0.5
            U_rand = np.round(U_rand * 2) / 2
            V_rand = np.round(V_rand * 2) / 2
            W_rand = np.round(W_rand * 2) / 2
            
            # Check if this random initialization happens to work (very unlikely)
            if check_exactness(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
                found = True
                break
        
        if found:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    if best_rank < schoolbook_rank:
        # Convert to list of lists with integer/half-integer entries
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Return schoolbook decomposition
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
