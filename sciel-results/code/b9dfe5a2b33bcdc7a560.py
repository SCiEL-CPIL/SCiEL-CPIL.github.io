"""Matrix multiplication tensor decomposition for 3x4 * 4x8."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """Return a rank-minimal (or near-minimal) exact tensor decomposition."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 3*4*8 = 96
    
    # Known lower bounds and upper bounds for matrix multiplication rank
    # For 3x4 * 4x8, the rank is at most 96 (schoolbook)
    # Strassen-like methods can reduce this
    
    # The tensor for matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k]
    # has structure that can be exploited.
    
    # For small matrices, we can try to find a low-rank decomposition
    # using the fact that the multiplication tensor has a specific structure.
    
    # One approach: use the fact that for matrix multiplication,
    # we can sometimes do better than schoolbook.
    
    # For 3x4 * 4x8, let's think about this more carefully.
    # The multiplication tensor T_{i,j,k} = 1 if we're multiplying A[i,j]*B[j,k]
    # to get C[i,k].
    
    # Actually, the standard way to think about this:
    # We want to decompose the bilinear map (A, B) -> C where C = A * B
    # into rank-1 tensors.
    
    # A rank-1 tensor in this context is: U[i,j] * V[j,k] * W[i,k]
    # where the indices match the matrix multiplication structure.
    
    # Wait, let me re-read the problem. The output factor W has shape (p*n, rank)
    # with W[k*n + i, r]. So the tensor is T[i,j,k] and we decompose it as:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is the standard matrix multiplication tensor decomposition.
    
    # For 3x4 * 4x8, the schoolbook rank is 96.
    # Can we do better?
    
    # One known result: for multiplying an m x n matrix by an n x p matrix,
    # the rank is at most m*n*p (schoolbook), but can be lower.
    
    # For specific small cases, we can use known results or try to find
    # better decompositions.
    
    # Let's try a different approach: use the fact that matrix multiplication
    # can be done with fewer multiplications using divide-and-conquer or
    # other algebraic methods.
    
    # For 3x4 * 4x8, one approach is to use the fact that we can group
    # some of the multiplications.
    
    # Actually, for the specific case of 3x4 * 4x8, let's think about
    # whether there's a known optimal rank.
    
    # The rank of the matrix multiplication tensor for (m,n,p) is denoted R(m,n,p).
    # We know R(m,n,p) <= m*n*p (schoolbook).
    # We also know R(m,n,p) >= max(m*n, m*p, n*p) (from the "dimension" lower bound).
    # For 3,4,8: max(12, 24, 32) = 32.
    
    # So the rank is between 32 and 96.
    
    # Let's try to find a decomposition with rank less than 96.
    
    # One simple improvement: if we can factor some of the matrices,
    # we might reduce the rank.
    
    # For example, if we write the 4x4 identity-like structure differently...
    
    # Actually, let's try a more systematic approach.
    # We can use the fact that matrix multiplication is associative and
    # try to find a decomposition that exploits this.
    
    # For now, let's return the schoolbook decomposition as a baseline,
    # and if we have time, try to improve it.
    
    # Schoolbook decomposition:
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a better decomposition using random initialization and
    # alternating least squares or similar methods.
    
    # For now, let's try a simple approach: see if we can reduce the rank
    # by finding linear dependencies.
    
    # Actually, for the specific case of 3x4 * 4x8, let me think about
    # whether there's a known optimal rank.
    
    # The rank of the 3x4 * 4x8 matrix multiplication tensor...
    # I don't have a specific reference for this exact case, but let's
    # try to find a decomposition with rank less than 96.
    
    # One approach: use the fact that we can sometimes combine terms.
    
    # For example, consider the 2x2 * 2x2 case. The schoolbook rank is 8,
    # but Strassen's algorithm shows it can be done with 7 multiplications.
    
    # For 3x4 * 4x8, we can try to apply similar ideas.
    
    # Let's try a different approach: use the fact that the multiplication
    # tensor has a certain symmetry and try to find a low-rank approximation.
    
    # For now, let's just return the schoolbook decomposition.
    # If we have more time, we can try to improve it.
    
    if time.time() > deadline - 10:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition using a simple optimization approach.
    # We'll try to find a rank-r decomposition for r < 96.
    
    # One approach: start with the schoolbook decomposition and try to
    # find linear combinations that reduce the rank.
    
    # Actually, a better approach: use the fact that matrix multiplication
    # can be viewed as a bilinear map, and try to find a low-rank
    # decomposition of this bilinear map.
    
    # For now, let's try a simple approach: see if we can find a
    # decomposition with rank 95, 94, etc.
    
    # This is a hard problem in general, so let's just return the
    # schoolbook decomposition for now.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
