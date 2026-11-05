"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # For 3x3x7 matrix multiplication, we use a block decomposition approach.
    # We can decompose 3x3x7 into 3x3x3 + 3x3x3 + 3x3x1
    # Rank(3x3x3) = 23 (using Bläser's algorithm)
    # Rank(3x3x1) = 9 (schoolbook: 3*3*1 = 9)
    # Total rank = 23 + 23 + 9 = 55
    
    # However, to be safe and ensure correctness, let's use a verified approach.
    # The schoolbook rank for 3x3x7 is 3*3*7 = 63.
    # We can improve by using rank 23 for the 3x3x3 blocks.
    
    # Let's construct the rank-23 decomposition for 3x3x3.
    # The 23 terms for 3x3 matrix multiplication (Bläser 2005) are:
    # Each term has u in R^9, v in R^9, w in R^9
    # u[i,j], v[j,k], w[k,i] with indices 0-2
    
    # The 23 bilinear forms for 3x3 matrix multiplication:
    # Using the known decomposition with integer coefficients
    
    # Actually, let me use a simpler but valid improvement.
    # I'll use the fact that we can decompose 3x3x7 as 3x3x3 + 3x3x4
    # But that doesn't help directly.
    
    # Let me just use the schoolbook to be safe, which gives rank 63.
    # To improve, I need a valid rank-23 decomposition for 3x3x3.
    
    # Here is a known rank-23 decomposition for 3x3x3:
    # The terms are (u, v, w) where u, v, w are 9-dimensional vectors
    
    def get_rank23_3x3x3():
        """Return the 23 terms for 3x3x3 matrix multiplication."""
        # Each term is a tuple (u, v, w) where u, v, w are lists of 9 elements
        # u[i*3+j], v[j*3+k], w[k*3+i] for i,j,k in {0,1,2}
        
        # Using the Bläser decomposition (2005) with integer coefficients
        # The 23 terms are:
        terms = []
        
        # Term 1: (1,0,0, 0,1,0, 0,0,1) * (0,1,0, 1,0,0, 0,0,1) * (0,0,1, 0,1,0, 1,0,0)
        # This is getting complex. Let me use a verified set.
        
        # Actually, I'll use a different approach. Let me use the fact that
        # we can pad 3x3x7 to 4x4x8 and use known decompositions, but that's complex.
        
        # For safety, I'll return the schoolbook decomposition.
        # The schoolbook for 3x3x7 has rank 63.
        return None
    
    # Since I cannot reliably provide the rank-23 decomposition from memory,
    # I'll use the schoolbook decomposition which is guaranteed correct.
    # This gives rank 63, which is the same as the current program.
    
    # To actually improve, let me think differently.
    # We can use the fact that 3x3x7 = 3x3x3 + 3x3x3 + 3x3x1
    # If we use rank 27 (schoolbook) for each 3x3x3 and rank 9 for 3x3x1,
    # total is 27 + 27 + 9 = 63.
    
    # But we can do better! Let's use a hybrid approach.
    # For the 3x3x1 part, we only need 9 terms.
    # For the 3x3x3 parts, if we can use rank 23, we get 23 + 23 + 9 = 55.
    
    # Let me try to construct a valid rank-23 decomposition.
    # I'll use a known result: the rank of 3x3 matrix multiplication is 23.
    # The decomposition uses 23 bilinear forms.
    
    # Here is one such decomposition (from literature):
    def get_blaser_23():
        """Bläser's rank-23 decomposition for 3x3x3."""
        # The 23 terms with integer coefficients
        # Format: (u, v, w) where each is a list of 9 integers
        
        terms = []
        
        # I'll use a verified set of 23 terms.
        # These are from the Bläser 2005 paper.
        
        # Term 1
        u1 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        v1 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        w1 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        terms.append((u1, v1, w1))
        
        # This is not correct. Let me use a different approach.
        # I'll construct the decomposition programmatically if possible,
        # or fall back to schoolbook.
        
        return terms
    
    # Given the complexity and risk of providing an invalid decomposition,
    # I'll use the schoolbook decomposition which is guaranteed correct.
    # This maintains the same score as the parent (1.0).
    
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
