"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition using tensor rank minimization
    # For 4x4x9, we can attempt to use known low-rank decompositions
    # The matrix multiplication tensor T_{4,4,9} has a known rank bound
    
    # Strategy: Use the standard multiplication tensor and try to compress it
    # by finding linear dependencies among the rank-1 components
    
    # Start with schoolbook
    result = schoolbook()
    rank = n * m * p
    
    # Try a simple rank-reduction: for m x p matrices, if we can share terms
    # when the output dimension allows it
    
    # For 4x4x9, let's try to see if we can do better than 144
    # One approach: use the fact that 9 = 4+4+1, and potentially share some computations
    
    # Actually, let's try a direct approach: use bilinear decomposition
    # A (4x4) * B (4x9) = C (4x9)
    # C[i][k] = sum_j A[i][j] * B[j][k]
    
    # The standard decomposition has rank 4*4*9 = 144
    # Can we do better?
    
    # Try: group by output column k, and for each k, compute A * B[:,k]
    # This gives 9 separate 4x4x4 multiplications, each of rank at most 16
    # Total: 9 * 16 = 144, same as schoolbook
    
    # Try another approach: use Winograd-style algorithm
    # For small matrices, there are known improved decompositions
    
    # Let's try to use the fact that for 4x4 matrix multiplication,
    # the rank is at most 16 (schoolbook) but potentially less
    
    # Actually, for 4x4x4, the rank is known to be at most 16 (schoolbook)
    # and at least 13 (lower bound). Let's just use schoolbook for now.
    
    # The key insight: for 4x4x9, we can potentially do better than 144
    # by using a non-trivial decomposition
    
    # Try: C = A * B where A is 4x4, B is 4x9
    # Let B = [B1 | B2 | B3] where B1, B2 are 4x4 and B3 is 4x1
    # C = [A*B1 | A*B2 | A*B3]
    # If we can compute A*B1 and A*B2 with shared work, we save
    
    # For now, return schoolbook if time is running out
    if time.time() > deadline - 5:
        return result
    
    # Attempt to find a lower rank decomposition
    # This is a hard problem, but let's try a simple heuristic
    
    # Actually, the rank of the matrix multiplication tensor T_{m,n,p}
    # satisfies certain bounds. For T_{4,4,9}, the rank is between
    # max(4*4, 4*9, 4*9) = 36 and 4*4*9 = 144
    
    # A better bound: rank(T_{m,n,p}) >= (m+n+p-1)^2 / (mn+np+pm)
    # For 4,4,9: (4+4+9-1)^2 / (16+36+36) = 169/88 ≈ 1.92, so rank >= 2
    # This is a very weak bound
    
    # The Strassen rank for 4x4x4 is 16 (same as schoolbook) or 7 if using 2x2 blocks
    # But for 4x4x9, we need a different approach
    
    # Let's just return schoolbook - it's correct and the rank is 144
    return result
# EVOLVE-BLOCK-END
