"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication.
    
    Strategy: Use block decomposition with Strassen-like algebraic structure.
    
    The matrix multiplication C = A * B where:
    - A is 4x5 (n=4, m=5)
    - B is 5x5 (m=5, p=5)  
    - C is 4x5 (n=4, p=5)
    
    We split A into two 2x5 blocks: A_top (rows 0-1) and A_bot (rows 2-3)
    Similarly for C: C_top (rows 0-1) and C_bot (rows 2-3)
    
    C_top = A_top * B
    C_bot = A_bot * B
    
    Each 2x5 * 5x5 multiplication can be done with rank 10 (schoolbook for 2x5*5x5 = 2*5*5/2 = 25, but we can do better).
    
    Actually, let's use a different approach: exploit the structure using linear combinations.
    
    Key insight: For 2x2 * 2x2, Strassen gives rank 7 instead of 8.
    For 2x5 * 5x5, we can view this as computing 10 outputs, each a dot product of length 5.
    
    Alternative: Use the fact that we can share computations across the 4 rows.
    
    Let's try a rank-reduction approach using linear dependencies.
    
    Actually, let me try a completely different structural approach:
    
    The tensor T[i,j,k] = 1 for all i in [0,4), j in [0,5), k in [0,5)
    needs to be decomposed as sum_r u_r[i*5+j] * v_r[j*5+k] * w_r[k*4+i]
    
    We can try to find a decomposition with rank < 100 by exploiting algebraic structure.
    
    Approach: Use the Kronecker product structure.
    4x5*5x5 can be seen as (2*2) x (5) x (5) x (5)
    
    Let's use a hybrid approach:
    - For the "2x2" part in the first dimension, use Strassen-like rank-7 structure
    - For the "5" dimensions, use rank-5 structure
    
    This gives rank = 7 * 5 = 35? That seems too optimistic.
    
    Let me think more carefully.
    
    Actually, the correct decomposition for matrix multiplication tensor M_{n,m,p} has rank at most n*m*p (schoolbook).
    For M_{2,2,2}, Strassen shows rank 7 < 8.
    
    For M_{4,5,5}, we can try:
    - Split 4 = 2*2
    - The tensor M_{4,5,5} relates to M_{2,2,2} via some construction
    
    Let me try a direct algebraic approach: find linear combinations that reduce rank.
    
    For now, let's implement a guaranteed valid decomposition with rank <= 100,
    and try to find improvements.
    
    Schoolbook: rank = 100
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Start with schoolbook decomposition
    rank_schoolbook = n * m * p  # 100
    
    # Try to find a lower-rank decomposition using algebraic structure
    # Approach: Use the fact that 4 = 2*2 and apply Strassen-like ideas
    
    # For now, return the schoolbook decomposition as a valid baseline
    # Future work: implement Strassen-based decomposition for rank < 100
    
    rank = rank_schoolbook
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
# EVOLVE-BLOCK-END
