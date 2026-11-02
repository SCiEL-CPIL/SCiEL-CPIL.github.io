"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]
import numpy as np
import time
from itertools import product

# EVOLVE-BLOCK-START
def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i in [0,n), j in [0,m), k in [0,p)"""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W form a valid rank-r decomposition of the n×m×p multiplication tensor."""
    rank = len(U[0])
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_with_symmetry(n, m, p, rank, max_time=140):
    """
    Try to find a rank decomposition using symmetry.
    
    For n=m=3, the multiplication tensor has a block structure.
    We exploit that the tensor can be viewed as a 3×3 block matrix where
    each block is a 3×7 matrix.
    
    Strategy: Use the fact that for square-ish matrices, we can try
    decompositions that respect the block structure.
    """
    start_time = time.time()
    
    # The tensor T[i,j,k] = 1 for all valid i,j,k
    # We want: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # For n=m=3, we can try to use the structure that the tensor is
    # a sum of rank-1 tensors.
    
    # Approach: Use the identity that the multiplication tensor for 3×3×7
    # can potentially be decomposed using fewer than 63 terms by exploiting
    # the fact that the output has a specific structure.
    
    # Let's try a structured approach: assume the decomposition has a
    # block-diagonal structure in the first index.
    
    # For now, let's try a random search with structured initialization
    np.random.seed(42)
    
    best_rank_found = None
    
    # Try different structured patterns
    # Pattern 1: Use the fact that the tensor is all-ones, so we can
    # potentially use fewer terms by having each term cover multiple (i,j,k)
    
    # Actually, let's think about this differently. The multiplication tensor
    # for multiplying an n×m matrix by an m×p matrix has rank at most n*m*p.
    # For n=m=3, p=7, that's 63.
    
    # Known results: The rank of 3×3×3 multiplication is at most 23 (Strassen).
    # For 3×3×7, we might be able to do better than 63.
    
    # Let's try a direct approach: use the structure of the problem.
    # The tensor T[i,j,k] = 1 means we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # One approach: use the fact that we can write 1 as a sum of products.
    # For example, if we set U[i*m+j, c] = a_c, V[j*p+k, c] = b_c, W[k*n+i, c] = c_c
    # then sum_c a_c * b_c * c_c = 1 for all i,j,k.
    # This requires that the sum is independent of i,j,k, which means
    # the products must be constant across i,j,k for each c, or the
    # variations must cancel.
    
    # A simpler approach: use the block structure.
    # Group the (i,j) pairs and (k) values.
    
    # Let's try: for each pair (i,j), we can use a set of terms that
    # covers all k values. But we can share terms across different (i,j)
    # if the W factor is structured appropriately.
    
    # Actually, let's try a more systematic search.
    # We'll use the fact that the tensor is symmetric in some sense.
    
    # For now, return the schoolbook rank as a fallback, but let's try
    # to find something better.
    
    # Try rank values from 40 to 62
    for r in range(40, min(63, rank + 1)):
        if time.time() - start_time > max_time - 10:
            break
        
        # Try to construct a rank-r decomposition
        # Use a structured approach: assume the decomposition has
        # a block structure where terms are grouped by i-value.
        
        # For each i in {0,1,2}, we need to cover all j in {0,1,2} and k in {0,...,6}
        # That's 3*7 = 21 terms per i-block if we don't share.
        # But we can share terms across i-blocks if W is structured.
        
        # Let's try: allocate r/3 terms per i-block (approximately)
        terms_per_i = r // 3
        remainder = r % 3
        
        # This is getting complex. Let's try a different approach.
        # Use the known result that the rank of 3×3×p multiplication
        # is related to the rank of 3×3×3 and can be bounded.
        
        # For now, let's just try the schoolbook and return it.
        # The search for better decompositions would require more sophisticated
        # algorithms (e.g., using the border rank or specific constructions).
        
        pass
    
    # Fallback: return schoolbook
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a better decomposition
    start_time = time.time()
    max_time = 140  # Leave 10s margin for safety
    
    # For now, return the schoolbook decomposition
    # A full search would require implementing sophisticated algorithms
    # like those used in Strassen's algorithm or its generalizations.
    
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
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
