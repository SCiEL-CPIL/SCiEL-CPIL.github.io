"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # For 2x2x5 matrix multiplication, we can use a rank-16 decomposition
    # This exploits the structure by combining terms
    
    # The standard rank for 2x2x5 is 20 (n*m*p)
    # But we can do better by using the fact that we can combine some terms
    
    # Let's use a known optimal or near-optimal decomposition
    # For 2x2x5, the optimal rank is actually 16 (can be shown via border rank arguments)
    
    # We'll construct a rank-16 decomposition
    rank = 16
    U = [[0] * rank for _ in range(n * m)]  # 4 x 16
    V = [[0] * rank for _ in range(m * p)]  # 10 x 16
    W = [[0] * rank for _ in range(p * n)]  # 10 x 16
    
    # Matrix multiplication: C = A * B where A is 2x2, B is 2x5, C is 2x5
    # C[i][k] = sum_j A[i][j] * B[j][k]
    # Tensor: T[i,j,k] = A[i,j] * B[j,k]
    
    # For rank-1 decomposition: U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all (i,j,k) in support
    # We need to decompose the multiplication tensor
    
    # Let's use a systematic approach with some clever combinations
    # The key insight: for 2x2 matrices, we can use Strassen-like ideas
    # For 2x2x5, we can group the 5 columns of B
    
    # Actually, let me think about this more carefully.
    # The multiplication tensor for 2x2x5 has dimension 2*2*5 = 20
    # We want to find the minimum rank
    
    # A simple upper bound: 16
    # We can use the following construction:
    # For the 2x2 part, use rank-4 (or less with Strassen)
    # For each of the 5 columns, we need to multiply by the 2x2 matrix
    
    # Let's use a direct construction:
    # We'll use 16 rank-1 terms
    
    # Term structure:
    # For i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    # We need U[i*2+j], V[j*5+k], W[k*2+i]
    
    # Let's try a rank-16 construction based on combining some terms
    
    # Actually, the simplest approach that might give rank < 20:
    # Use the fact that for 2x2 matrices, we can compute products with fewer terms
    
    # Let me use a known rank-16 decomposition for 2x2x5
    
    # Construction:
    # We'll use the following 16 terms:
    
    # First, let's define the basis
    # U has 4 rows (i=0,1; j=0,1)
    # V has 10 rows (j=0,1; k=0,1,2,3,4)
    # W has 10 rows (k=0,1,2,3,4; i=0,1)
    
    # Term 1: i=0, j=0, k=0
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    # Term 2: i=0, j=0, k=1
    U[0][1] = 1; V[1][1] = 1; W[2][1] = 1
    # Term 3: i=0, j=0, k=2
    U[0][2] = 1; V[2][2] = 1; W[4][2] = 1
    # Term 4: i=0, j=0, k=3
    U[0][3] = 1; V[3][3] = 1; W[6][3] = 1
    # Term 5: i=0, j=0, k=4
    U[0][4] = 1; V[4][4] = 1; W[8][4] = 1
    
    # Term 6: i=1, j=0, k=0
    U[2][5] = 1; V[0][5] = 1; W[1][5] = 1
    # Term 7: i=1, j=0, k=1
    U[2][6] = 1; V[1][6] = 1; W[3][6] = 1
    # Term 8: i=1, j=0, k=2
    U[2][7] = 1; V[2][7] = 1; W[5][7] = 1
    # Term 9: i=1, j=0, k=3
    U[2][8] = 1; V[3][8] = 1; W[7][8] = 1
    # Term 10: i=1, j=0, k=4
    U[2][9] = 1; V[4][9] = 1; W[9][9] = 1
    
    # Term 11: i=0, j=1, k=0
    U[1][10] = 1; V[5][10] = 1; W[0][10] = 1
    # Term 12: i=0, j=1, k=1
    U[1][11] = 1; V[6][11] = 1; W[2][11] = 1
    # Term 13: i=0, j=1, k=2
    U[1][12] = 1; V[7][12] = 1; W[4][12] = 1
    # Term 14: i=0, j=1, k=3
    U[1][13] = 1; V[8][13] = 1; W[6][13] = 1
    # Term 15: i=0, j=1, k=4
    U[1][14] = 1; V[9][14] = 1; W[8][14] = 1
    
    # Term 16: i=1, j=1, k=0
    U[3][15] = 1; V[5][15] = 1; W[1][15] = 1
    
    # Wait, this is just 16 terms but it doesn't cover all cases
    # I need to reconsider
    
    # Let me use a simpler approach: just use the schoolbook method with rank 20
    # and return that. The task asks for rank-minimal, but if I can't find better,
    # I should at least return a valid decomposition.
    
    # Actually, let me just use the schoolbook decomposition with rank 20
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
