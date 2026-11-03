"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Winograd-style decomposition for 3x5x6 matrix multiplication
    # Embed into a structure where we can use Winograd transforms
    
    # The key idea: use Winograd's algorithm which exploits the fact that
    # for small matrix multiplication, we can use polynomial evaluation/interpolation
    
    # For 3x5x6, we'll use a hybrid approach:
    # 1. Use the fact that 3 = 2+1, so we can split the 3x3 part
    # 2. Use Winograd transforms for the 5 and 6 dimensions
    
    # Actually, let's use a more direct approach based on known low-rank decompositions
    # For 3x5x6, the rank is at most 3*5*6 = 90 (schoolbook)
    # We can potentially do better using structured decompositions
    
    # Let's try a direct construction using the fact that we can combine
    # some of the schoolbook terms using algebraic identities
    
    # Start with schoolbook as baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each (i,j,k) triple, we have a term
    # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    
    # Let's try to use Winograd's method for the 3x3 sub-blocks
    # Winograd 3x3x3 has rank 19 (vs 27 schoolbook)
    # But we need to handle the 5 and 6 dimensions carefully
    
    # Alternative: use the fact that we can apply Winograd to blocks
    # For 3x5x6, we can think of it as:
    # - 3x3x6 blocks (3 of them for the j dimension)
    # - Each 3x3x6 can be decomposed using Winograd for the 3x3 part
    
    # Let's implement a simpler but effective approach:
    # Use the known rank-19 decomposition for 3x3x3 and extend it
    
    # For now, let's return the schoolbook decomposition which is guaranteed correct
    # We can try to improve this in future iterations
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Check if we're within time limit
    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}
    
    # Try to improve using Winograd-style transformations
    # For the 3x3x3 sub-blocks, we can use rank-19 decomposition
    # This would reduce the rank from 27 to 19 for each 3x3x3 block
    
    # The 3x5x6 tensor can be seen as having 5*6 = 30 such "effective" 3x3 blocks
    # But they share structure, so we need to be careful
    
    # Let's try a different approach: use the fact that we can combine
    # terms that have the same (i,k) pattern but different j
    
    # For fixed i and k, we need sum_j A[i,j]*B[j,k]
    # This is a dot product of length m=5
    # We can use Winograd transforms for this dot product
    
    # Winograd for length-5 dot product can be done with 4 multiplications
    # (instead of 5) using the fact that (a0+a1+a2+a3+a4)(b0+b1+b2+b3+b4)
    # can be combined with other products
    
    # Actually, for a single dot product of length 5, the minimum number of
    # multiplications is 5 (you can't do better for a single dot product)
    # But when you have many dot products sharing structure, you can do better
    
    # Let's stick with the schoolbook for now and return it
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
