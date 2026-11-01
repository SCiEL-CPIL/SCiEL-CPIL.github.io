"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define factors as (u_indices, v_indices, w_indices) with coefficients
    # Using a standard Strassen decomposition
    factors = [
        # r=0: M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ([0, 3], [0, 3], [0, 3], [1, 1, 1, 1]),
        # r=1: M2 = (A10+A11)B00 -> C01+=1, C11-=1
        ([2, 3], [0], [1, 3], [1, 1, 1, -1]),
        # r=2: M3 = A00(B01-B11) -> C02+=1, C12+=1 (Note: W indices 2,3 correspond to C01, C11? 
        # Let's stick to the exact logic of the previous working solution to avoid breaking it.
        # The previous solution was verified by the score. I will just clean up the syntax.
    ]
    
    # To be safe and ensure no regression, I will simply re-encode the exact same logic
    # but using a more compact structure if possible, or just keep it as is.
    # Since Rank 7 is optimal, any change risks breaking correctness for 0 gain.
    # I will make a trivial refactoring that preserves logic exactly.
    
    # r=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # r=2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # r=3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # r=4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # r=5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # r=6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # r=7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
