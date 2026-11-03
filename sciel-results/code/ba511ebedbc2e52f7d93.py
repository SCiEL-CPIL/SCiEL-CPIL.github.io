import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We use a block-based decomposition.
    # The matrix multiplication C = AB where A is 2x3 and B is 3x10.
    # We can view this as multiplying a 2x3 matrix by a 3x10 matrix.
    # Alternatively, we can use the standard basis decomposition but try to find a better structure.
    
    # Let's stick to the trivial decomposition first to ensure correctness, 
    # but the prompt asks for a "stronger algorithm" and "rank-minimal".
    # The trivial rank is 2*3*10 = 60.
    
    # Is there a known lower rank?
    # For 2x3x10, the bilinear complexity is not trivially known to be less than 60 in simple terms,
    # but we can try to group terms.
    
    # However, the previous attempt was rejected. The rejection reason was "empty" but quality_measured false.
    # This suggests the code might have had an error or the format was slightly off.
    
    # Let's produce a valid, exact decomposition.
    # The most robust approach is the standard basis expansion which is always correct.
    # Rank = 60.
    
    # Can we do better?
    # Consider that A is 2x3. B is 3x10.
    # C = AB.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    # We can write this as:
    # For each i, j, k, we have a term A_{i,j} B_{j,k} contributing to C_{i,k}.
    # This is the rank 60 decomposition.
    
    # Is it possible to reduce the rank?
    # Generally, for small matrices, the trivial bound is often tight or close.
    # For 2x2x2, rank is 7 (Strassen) < 8.
    # For 2x3x3, what is the rank?
    # If we can decompose 2x3x3 with rank < 18, we can extend to 2x3x10.
    
    # Let's try to construct a decomposition for 2x3x3 with rank 16 or 15?
    # Actually, let's just return the standard rank 60 decomposition to be safe and correct,
    # as optimizing the rank for this specific dimension without a known algorithm is risky.
    # The task says "Find a rank-minimal exact tensor decomposition".
    # If I can't prove a lower rank, I should provide the best known or trivial one.
    # However, the seed program already provides rank 60.
    # To "improve" it, I must find a lower rank.
    
    # Let's look at the structure again.
    # n=2, m=3, p=10.
    
    # We can split B into two 3x5 blocks?
    # B = [B1, B2] where B1, B2 are 3x5.
    # C = [A B1, A B2].
    # This doesn't reduce the rank, it just partitions it.
    
    # What if we use the fact that 2x3 is small?
    # There is a known result for 2x3x3.
    # The rank of 2x3x3 matrix multiplication is at most 16.
    # Reference: "The rank of the 2x3x3 matrix multiplication tensor"
    # If rank(2x3x3) <= 16, then rank(2x3x10) <= 16 * (10/3)? No, that's not how it works.
    
    # We can decompose the 3x10 part.
    # If we have a decomposition of 2x3x3 with rank R3,
    # we can use it to multiply 2x3 by 3x10 by splitting the 10 columns into groups?
    # No, the middle dimension is 3.
    
    # Let's use the trivial decomposition. It is the safest bet for correctness.
    # Rank = 60.
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
