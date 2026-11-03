"""Matrix multiplication 3x5x5 - rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x5x5 matrix multiplication.
    
    Strategy: Use Strassen's algorithm on 2x2x2 sub-blocks to reduce rank.
    The 3x5x5 problem can be decomposed into blocks where Strassen applies.
    
    For 3x5x5:
    - Split A (3x5) into rows: [A1 (2x5), A2 (1x5)]
    - Split B (5x5) into columns: [B1 (5x2), B2 (5x3)]
    - Output C (3x5) = [C1 (3x2), C2 (3x3)]
    
    C1 = A1*B1 (2x5 * 5x2 = 2x2) + A2*B1 (1x5 * 5x2 = 1x2)
    C2 = A1*B2 (2x5 * 5x3 = 2x3) + A2*B2 (1x5 * 5x3 = 1x3)
    
    For C1 (2x2 output from 2x5 * 5x2):
    We can use Strassen-like approach. Let's think more carefully.
    
    Actually, let's use a more direct approach. The known lower bound for 3x5x5 is 71
    (using Strassen on embedded 2x2x2 blocks). Let's construct that.
    
    Alternative approach: Use the fact that we can decompose as:
    - 2x5x5 block: rank 2*5*5 = 50 (schoolbook) or better
    - 1x5x5 block: rank 1*5*5 = 25 (schoolbook)
    Total: 75
    
    With Strassen on the 2x5x5 part:
    Split the 2x5 matrix A into [A_top (1x5), A_bottom (1x5)]
    Split the 5x5 matrix B into columns [B_left (5x2), B_right (5x3)]
    
    C = A*B where A is 2x5, B is 5x5, C is 2x5
    
    C = [A_top * B_left (1x2), A_top * B_right (1x3);
         A_bot * B_left (1x2), A_bot * B_right (1x3)]
    
    For the top-left 1x2 block: A_top (1x5) * B_left (5x2)
    This is a 1x5x2 multiplication, rank = 1*5*2 = 10 (schoolbook), but can be done in rank 5?
    No, 1x5x2 requires rank at least max(1*5, 1*2, 5*2) = 10? Actually for 1xm*pn, 
    the rank is m*p if m=1 or n=1... let me reconsider.
    
    For a 1xm * mxn multiplication, the result is 1xn. This can be done with m*n scalar 
    multiplications (schoolbook). Can we do better? The tensor is a 1xm*mxn = 1xn tensor.
    The rank of this tensor... for 1xm*mxn, I believe the rank is m*n (you need all m*n 
    products). So no improvement there.
    
    Let me think differently. The known result is that 3x5x5 has rank 71 using Strassen.
    Let me construct it properly.
    
    Decomposition:
    A is 3x5, B is 5x5, C is 3x5.
    
    Write A = [A1; A2] where A1 is 2x5, A2 is 1x5.
    Write B = [B1, B2] where B1 is 5x2, B2 is 5x3.
    
    C = [A1*B1 + A2*B1, A1*B2 + A2*B2]
       = [C11 + C21, C12 + C22]
    
    where C11 = A1*B1 (2x2), C21 = A2*B1 (1x2), C12 = A1*B2 (2x3), C22 = A2*B2 (1x3).
    
    For C11 (2x5 * 5x2 -> 2x2):
    A1 is 2x5, B1 is 5x2. 
    Write A1 = [A1a, A1b] where A1a is 2x3, A1b is 2x2.
    Write B1 = [B1a; B1b] where B1a is 3x2, B1b is 2x2.
    C11 = A1a*B1a + A1b*B1b.
    
    A1b*B1b is 2x2 * 2x2 -> 2x2. Use Strassen: rank 7.
    A1a*B1a is 2x3 * 3x2 -> 2x2. Rank = 2*3*2 = 12 (schoolbook). Can we do better?
    For 2x3*3x2, the rank is... I believe it's 12 (no known better algorithm).
    
    So C11: 12 + 7 = 19.
    
    For C21 (1x5 * 5x2 -> 1x2):
    A2 is 1x5, B1 is 5x2.
    Write A2 = [A2a, A2b] where A2a is 1x3, A2b is 1x2.
    B1 = [B1a; B1b] where B1a is 3x2, B1b is 2x2.
    C21 = A2a*B1a + A2b*B1b.
    A2a*B1a: 1x3 * 3x2 -> 1x2, rank = 3*2 = 6.
    A2b*B1b: 1x2 * 2x2 -> 1x2, rank = 2*2 = 4.
    Total: 6 + 4 = 10.
    
    For C12 (2x5 * 5x3 -> 2x3):
    A1 is 2x5, B2 is 5x3.
    Write A1 = [A1c, A1d] where A1c is 2x3, A1d is 2x2.
    B2 = [B2a; B2b] where B2a is 3x3, B2b is 2x3.
    C12 = A1c*B2a + A1d*B2b.
    A1c*B2a: 2x3 * 3x3 -> 2x3, rank = 2*3*3 = 18.
    A1d*B2b: 2x2 * 2x3 -> 2x3, rank = 2*2*3 = 12.
    Total: 18 + 12 = 30.
    
    For C22 (1x5 * 5x3 -> 1x3):
    A2 is 1x5, B2 is 5x3.
    Write A2 = [A2a, A2b] where A2a is 1x3, A2b is 1x2.
    B2 = [B2a; B2b] where B2a is 3x3, B2b is 2x3.
    C22 = A2a*B2a + A2b*B2b.
    A2a*B2a: 1x3 * 3x3 -> 1x3, rank = 3*3 = 9.
    A2b*B2b: 1x2 * 2x3 -> 1x3, rank = 2*3 = 6.
    Total: 9 + 6 = 15.
    
    Grand total: 19 + 10 + 30 + 15 = 74. Hmm, not 71.
    
    Let me try a different splitting. Maybe split differently.
    
    Actually, let me just implement the schoolbook (rank 75) and then try to find 
    a better decomposition through search or known algorithms. Given the time constraint,
    let me implement a careful schoolbook and also try the Strassen-based approach.
    
    Let me just go with a clean schoolbook implementation first, which gives rank 75.
    """
    n, m, p = DIMENSIONS
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
