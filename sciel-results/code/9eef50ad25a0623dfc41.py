"""Matrix multiplication 3x4x5 - attempt to find a lower rank decomposition."""

import time
import itertools
import numpy as np
from collections import defaultdict

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    by a 4 by 5 matrix.
    
    Returns:
        dict with 'U', 'V', 'W' factor matrices
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback (rank = n*m*p = 60)
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want to find rank r such that:
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] for all i,j,k
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    
    # Let's try a greedy approach with small rational coefficients
    # First, try to find a decomposition with rank < 60
    
    # Try ranks from low to high
    # For 3x4x5, known results suggest rank is around 49 or lower
    
    # Use a randomized search with local optimization
    # The constraint is: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    def try_decomposition(rank, max_iters=5000, seed=42):
        """Try to find a decomposition of given rank using randomized approach."""
        if rank < 1:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Try with integer coefficients first: {-1, 0, 1}
        # Then try half-integers: {-0.5, 0, 0.5, 1}
        
        for coeff_set in [[-1, 0, 1], [-1, -0.5, 0, 0.5, 1], [-2, -1, -0.5, 0, 0.5, 1, 2]]:
            for attempt in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Randomly initialize U, V, W with values from coeff_set
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                W = np.zeros((p * n, rank))
                
                # Initialize with random sparse entries
                for c in range(rank):
                    # For each column, randomly pick some non-zero entries
                    for i in range(n):
                        for j in range(m):
                            if rng.random() < 0.3:
                                U[i * m + j, c] = rng.choice(coeff_set)
                    for j in range(m):
                        for k in range(p):
                            if rng.random() < 0.3:
                                V[j * p + k, c] = rng.choice(coeff_set)
                    for k in range(p):
                        for i in range(n):
                            if rng.random() < 0.3:
                                W[k * n + i, c] = rng.choice(coeff_set)
                
                # Check if this satisfies the constraints
                success = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            val = 0.0
                            for c in range(rank):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            if abs(val - 1.0) > 1e-9:
                                success = False
                                break
                        if not success:
                            break
                    if not success:
                        break
                
                if success:
                    # Convert to list of lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try to find a decomposition with rank less than 60
    # Start from lower ranks and go up
    best_result = None
    
    # Known: rank of 3x3x3 is at most 23, 2x2x2 is 7
    # For 3x4x5, let's try ranks from 40 to 59
    for r in range(40, 60):
        if time.time() > deadline:
            break
        
        result = try_decomposition(r, max_iters=1000, seed=r)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Fall back to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}
    else:
        return best_result

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Found decomposition with rank: {rank}")
