"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for multiplying a 3x3 matrix by a 3x8 matrix.
    Uses Strassen's algorithm on four non-overlapping 2x2 blocks of the B matrix,
    reducing the rank from the schoolbook 72 to 68.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (24, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    # We will build the decomposition term by term.
    # Each term is a triplet (u, v, w) where u is a vector of length 9, 
    # v is a vector of length 24, and w is a vector of length 24.
    # The decomposition is: T = sum_k u_k \otimes v_k \otimes w_k
    
    terms = []
    
    # Helper to add a term
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Strassen decomposition for 2x2 block multiplication
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen factors:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each of the 4 blocks in B (columns 0-1, 2-3, 4-5, 6-7),
    # we use the top-left 2x2 block of A (rows 0-1, cols 0-1).
    # The remaining terms (involving row 2 or col 2 of A) are schoolbook.
    
    # Let's define the index mappings.
    # U index: i*m + j (i in 0..2, j in 0..2) -> 0..8
    # V index: j*p + k (j in 0..2, k in 0..7) -> 0..23
    # W index: k*n + i (k in 0..7, i in 0..2) -> 0..23
    
    # For a Strassen block on B columns (c0, c1) and A rows/cols (0,1):
    # We replace the 8 schoolbook terms for (i in {0,1}, j in {0,1}, k in {c0,c1})
    # with 7 Strassen terms.
    
    # Let's write a function to add the Strassen terms for a specific block.
    def add_strassen_block(c0, c1):
        # c0, c1 are the column indices of the B block
        # The A block is rows 0,1 and cols 0,1
        # So A00 = A[0,0] -> U index 0*3+0 = 0
        # A01 = A[0,1] -> U index 0*3+1 = 1
        # A10 = A[1,0] -> U index 1*3+0 = 3
        # A11 = A[1,1] -> U index 1*3+1 = 4
        
        # B00 = B[0, c0] -> V index 0*8+c0 = c0
        # B01 = B[0, c1] -> V index 0*8+c1 = c1
        # B10 = B[1, c0] -> V index 1*8+c0 = 8+c0
        # B11 = B[1, c1] -> V index 1*8+c1 = 8+c1
        
        # C00 = C[0, c0] -> W index c0*3+0 = 3*c0
        # C01 = C[0, c1] -> W index c1*3+0 = 3*c1
        # C10 = C[1, c0] -> W index c0*3+1 = 3*c0+1
        # C11 = C[1, c1] -> W index c1*3+1 = 3*c1+1
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        u1 = {0: 1, 4: 1}
        v1 = {c0: 1, 8+c1: 1}
        w1 = {3*c0: 1, 3*c1+1: 1}  # C00 + C11
        add_term(u1, v1, w1)
        
        # Term 2: M2 = (A10+A11)B00
        u2 = {3: 1, 4: 1}
        v2 = {c0: 1}
        w2 = {3*c0+1: 1, -(3*c1+1): 1}  # C10 - C11
        add_term(u2, v2, w2)
        
        # Term 3: M3 = A00(B01-B11)
        u3 = {0: 1}
        v3 = {c1: 1, -(8+c1): 1}
        w3 = {3*c1: 1, 3*c1+1: 1}  # C01 + C11
        add_term(u3, v3, w3)
        
        # Term 4: M4 = A11(B10-B00)
        u4 = {4: 1}
        v4 = {8+c0: 1, -c0: 1}
        w4 = {3*c0: 1, 3*c0+1: 1}  # C00 + C10
        add_term(u4, v4, w4)
        
        # Term 5: M5 = (A00+A01)B11
        u5 = {0: 1, 1: 1}
        v5 = {8+c1: 1}
        w5 = {-(3*c0): 1, 3*c1: 1}  # -C00 + C01
        add_term(u5, v5, w5)
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        u6 = {3: 1, 0: -1}
        v6 = {c0: 1, c1: 1}
        w6 = {3*c1+1: 1}  # C11
        add_term(u6, v6, w6)
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        u7 = {1: 1, 4: -1}
        v7 = {8+c0: 1, 8+c1: 1}
        w7 = {3*c0: 1}  # C00
        add_term(u7, v7, w7)
    
    # Add Strassen blocks for columns (0,1), (2,3), (4,5), (6,7)
    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    add_strassen_block(6, 7)
    
    # Now add the remaining schoolbook terms.
    # These are terms where i=2 or j=2, and the term is not already covered by Strassen.
    # Actually, Strassen only covers i in {0,1} and j in {0,1}.
    # So we need to add:
    # 1. All terms with i=2 (any j, any k)
    # 2. All terms with j=2 and i in {0,1} (any k)
    # Note: Terms with i=2 and j=2 are included in 1.
    
    # Let's iterate over all (i, j, k) and add if not covered by Strassen.
    # Strassen covers i in {0,1} and j in {0,1} for all k.
    # So we skip if i in {0,1} and j in {0,1}.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in (0, 1) and j in (0, 1):
                    continue  # Covered by Strassen
                # Schoolbook term: A[i,j] * B[j,k] -> C[k,i]
                u = {i * m + j: 1}
                v = {j * p + k: 1}
                w = {k * n + i: 1}
                add_term(u, v, w)
    
    # Now, build the U, V, W matrices from the terms.
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u, v, w) in enumerate(terms):
        for idx, val in u.items():
            U[idx][col] = val
        for idx, val in v.items():
            V[idx][col] = val
        for idx, val in w.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

