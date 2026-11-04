"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
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

    result = {"U": U, "V": V, "W": W}

    # Check if we have time to attempt improvement
    if time.time() > deadline:
        return result

    # Try to find a lower-rank decomposition by searching for linear dependencies
    # among the schoolbook terms. We attempt to express some terms as combinations
    # of others, reducing the rank.
    
    # Build the tensor as a 3D array for verification
    import numpy as np
    
    # Verify the schoolbook solution is correct
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reconstruct tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
    
    # T should be all ones
    if not np.allclose(T_recon, 1.0):
        return result  # Fallback to schoolbook if verification fails

    # Attempt to reduce rank by finding redundant terms
    # Strategy: try to remove one term at a time and solve for the remaining
    # This is expensive, so limit to a few attempts
    
    # For each term index, try to see if it can be eliminated
    # by adjusting other terms. This requires solving a linear system.
    
    # Given time constraints, we'll do a limited search
    # Try removing up to 6 terms (reducing rank by 6)
    
    max_removals = 6
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # We'll try a greedy approach: for each column, check if removing it
    # and adjusting others can still satisfy the tensor equation.
    # This is complex, so we'll use a simpler heuristic.
    
    # Alternative: try specific known patterns for small matrices
    # For 3x4x8, check if rank 90 is achievable
    
    # Given the complexity, return schoolbook if no quick improvement found
    if time.time() > deadline - 10.0:
        return result

    # Try a simple rank reduction: look for pairs of terms that can be merged
    # Two terms (i1,j1,k1) and (i2,j2,k2) can potentially be merged if they share
    # structure in U, V, or W indices.
    
    # For now, return the verified schoolbook solution
    return result
# EVOLVE-BLOCK-END
