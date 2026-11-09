"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Try to find a rank-minimal decomposition
    # Start with schoolbook as fallback
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
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Try Strassen-like decomposition with overlapping blocks
    # For 2x5 x 5x6, try to split the 5-dimension into overlapping parts
    # Using the idea: if we can express the tensor as a sum of Kronecker products
    # of smaller tensors with lower total rank
    
    # Approach: Try all possible partitions of the middle dimension
    # and use known optimal ranks for small sub-problems
    
    # Known optimal ranks (from literature/computation):
    # 2x2x2: rank 7 (Strassen)
    # 2x3x3: unknown, schoolbook=18
    # 2x2x3: unknown, schoolbook=12
    # 3x3x3: unknown, schoolbook=27
    # For small cases, schoolbook is often optimal or near-optimal
    
    # Let's try a direct algebraic approach:
    # The matrix multiplication tensor T for n×m × m×p has components T[i,j,k]
    # where i in [0,n), j in [0,m), k in [0,p)
    # We want to write T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # For this specific case (2,5,6), let's try to find a decomposition
    # with rank < 60 using linear algebra
    
    # Build the tensor as a matrix (matricization)
    # T reshaped to (n*m) x (m*p) x (p*n) is not quite right
    # Better: think of it as a 3-tensor, matricize in different ways
    
    # The tensor T has entries T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T_{i,j,k} which maps (A,B) -> C[i,k]
    # As a 3-tensor of shape (n, m, p) -> but actually it's (n*m, m*p, p*n)
    
    # Let me think differently. The multiplication tensor for n×m * m×p -> n×p
    # has rank equal to the minimum number of rank-1 tensors that sum to it.
    # A rank-1 tensor is u ⊗ v ⊗ w where u in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    
    # The tensor itself: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
    
    # Let's build this tensor and try to find its rank via numerical methods
    # For exact integer/half-integer solutions, we need to be careful
    
    # Given the complexity, let's try a few structured decompositions
    
    # Strategy 1: Block diagonal with Strassen on 2x2 blocks
    # Split the 5x5 middle into: first 2x2 block, last 3x3 block
    # 2x2 x 2x6: use Strassen-like for the 2x2 part
    # 2x3 x 3x6: schoolbook
    
    # Actually, let's try a simpler approach: just use schoolbook but verify
    # and return it. The task says score is rank, lower is better.
    # If we can't find better, schoolbook at 60 is the baseline.
    
    # Let me try one more thing: use the fact that for 2×n × n×m,
    # there might be a decomposition with rank ~ 2*n*m - something
    
    # For now, return schoolbook. Future evolutions can try harder.
    return best
# EVOLVE-BLOCK-END
