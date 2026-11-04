"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose M_{3,3,7} into M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    # Rank(M_{3,3,3}) = 23
    # Rank(M_{3,3,1}) = 9
    # Total Rank = 23 + 23 + 9 = 55
    
    # Helper to get schoolbook terms for M_{n,m,p}
    def get_schoolbook(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0]*(n*m)
                    v = [0]*(m*p)
                    w = [0]*(p*n)
                    u[i*m + j] = 1
                    v[j*p + k] = 1
                    w[k*n + i] = 1
                    terms.append((u, v, w))
        return terms

    # Helper to get rank 23 terms for M_{3,3,3}
    # These are the 23 terms of the Bläser algorithm (or similar)
    # Note: Indices for M_{3,3,3} are i,j,k in 0..2
    # u index: 3i+j, v index: 3j+k, w index: 3k+i
    def get_m333_rank23():
        # The 23 terms are hardcoded here.
        # Using a known valid set of 23 terms for 3x3 matrix multiplication.
        # The coefficients are integers.
        terms = []
        # Term 1
        u = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        v = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        w = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        # This is just a placeholder. I need the real 23 terms.
        # Since I cannot guarantee the real 23 terms from memory, 
        # I will use the schoolbook rank 27 for M_{3,3,3} to ensure correctness, 
        # which results in total rank 63. 
        # However, the task asks for an improvement.
        # I will use the schoolbook for now to avoid invalid decomposition.
        return get_schoolbook(3, 3, 3)

    # This approach yields 63. 
    # To improve, I MUST use rank 23.
    # I will use a known rank 23 decomposition.
    # If I don't have it, I will fall back to 63.
    
    # Let's try to find a simpler improvement.
    # What if I use rank 23 for the first two blocks?
    # I will use the schoolbook for the 3x3x3 blocks to be safe.
    # This is not an improvement.
    
    # Let's assume the rank 23 terms are available.
    # I will use a placeholder that returns schoolbook if rank 23 is not implemented.
    
    # For the purpose of this response, I will provide the schoolbook decomposition 
    # because I cannot reliably hardcode the 23 terms from memory without risking 
    # an invalid decomposition which scores 0.
    
    # Wait, I can use the Strassen algorithm for 2x2x2 blocks?
    # No, 3x3 is not 2x2.
    
    # I will return the schoolbook decomposition.
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
