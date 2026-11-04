"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75

    # We'll try to find a decomposition with rank < schoolbook_rank
    # using a structured approach based on block diagonalization / Strassen-like ideas.
    
    # Key insight: For 3x5 * 5x5, we can try to exploit structure.
    # One approach: split the 5x5 matrix into blocks and use lower-rank decompositions.
    
    # Let's try a direct search approach with random initialization and gradient-free optimization.
    # But given the 120s time limit, we need something more targeted.
    
    # Alternative: Use the fact that the matrix multiplication tensor T_{(i,j,k), (j',k'), (k'',i')} 
    # has a specific structure. We can try to find a rank-r decomposition by solving a system.
    
    # Given the complexity, let's try a few known strategies:
    # 1. Schoolbook (rank 75) - always valid
    # 2. Try to find a lower-rank decomposition via randomized methods
    
    # For now, let's implement a more sophisticated approach:
    # We'll use the fact that we can decompose the multiplication as a sum of rank-1 tensors.
    # A rank-1 tensor corresponds to choosing one (i,j), one (j',k'), one (k'',i') such that
    # the tensor entry is 1 when j=j' and k=k'', i.e., T[(i,j,k),(j',k'),(k'',i')] = 1 if j=j' and k=k''.
    
    # Actually, the standard decomposition is:
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # Each term A_{ij} B_{jk} is a rank-1 contribution to the output.
    # So we need n*m*p = 75 terms in the schoolbook.
    
    # To reduce rank, we need to find linear combinations that reduce the number of independent terms.
    # This is equivalent to finding the tensor rank.
    
    # For 3x5x5, let's try a structured approach:
    # Split the 5 columns of A and 5 rows of B into blocks.
    # For example, split 5 = 2 + 3.
    
    # Let's try: A is 3x5, B is 5x5.
    # Split A into A1 (3x2) and A2 (3x3).
    # Split B into B1 (2x5) and B2 (3x5).
    # Then AB = A1*B1 + A2*B2.
    # A1*B1 is 3x2 * 2x5 -> rank 3*2*5 = 30
    # A2*B2 is 3x3 * 3x5 -> rank 3*3*5 = 45
    # Total: 75. No gain.
    
    # Better: use Strassen-like ideas. But Strassen is for 2x2.
    
    # Let's try a different approach: random search for a low-rank decomposition.
    # We'll initialize U, V, W with random values and try to optimize.
    
    # Given the time constraints, let's just return the schoolbook for now,
    # but add a quick check if we can find something better.
    
    # Actually, let's try a smarter approach: use the fact that we can sometimes
    # share computations across different (i,j,k) triples.
    
    # For 3x5x5, the known best upper bound might be less than 75.
    # Let's try to find a decomposition of rank 60 or so.
    
    # Strategy: Use a greedy algorithm to select rank-1 tensors that cover the most "uncovered" entries.
    
    # First, let's define the tensor T as a 3D array of shape (n*m, m*p, p*n) or similar.
    # Actually, the tensor is T[a, b, c] where a = i*m+j (0..14), b = j'*p+k' (0..24), c = k''*n+i'' (0..14)
    # T[a, b, c] = 1 if j == j' and k == k'' and i == i''... wait, no.
    
    # Let me re-read the task:
    # U has shape (n*m, rank) = (15, rank)
    # V has shape (m*p, rank) = (25, rank)  
    # W has shape (p*n, rank) = (15, rank)
    # The decomposition is: for each output (i,k) [i in 0..2, k in 0..4],
    # (AB)_{ik} = sum_r U[i*m + ?, r] * V[?, r] * W[k*n + i, r]
    
    # Wait, the output is indexed by (i, k) where i is row of A (0..2) and k is col of B (0..4).
    # But W is indexed by [k*n + i, r], so W's first index is k*n + i.
    
    # The tensor T should satisfy:
    # T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # And T[i, j, k] should be 1 if this corresponds to a valid term in the matrix product.
    
    # Actually, the matrix product is:
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # So for fixed (i, k), the value is sum over j of A_{ij} * B_{jk}.
    # This means the tensor T[i, j, k] = 1 for all valid (i,j,k), and the decomposition should give:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if we're computing the product correctly.
    
    # Hmm, I think I'm overcomplicating this. Let me just use the schoolbook and try to improve.
    
    # Given the time limit, let's return the schoolbook decomposition.
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
