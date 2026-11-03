"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    
    Strategy:
    1. Try to find a low-rank decomposition using structured patterns.
    2. The schoolbook rank is 3*3*4 = 36.
    3. We attempt to find a decomposition with rank < 36 using known algebraic techniques
       or by exploiting the rectangular structure.
    
    For 3x3 * 3x4, we can try to use the fact that the smaller dimension is 3.
    One approach: decompose the 4-column output using rank-1 updates on the 3x3 block.
    
    Actually, let's think about this more carefully. The tensor is of size 3x3x4 (input1 x input2 x output).
    The standard rank is 36.
    
    Known lower bounds: For m x n * n x p, the rank is at least max(mn, np, mp) * something... 
    Actually, for square n x n * n x n, the best known is O(n^2.79). For 3x3, the rank is known to be 23 (Strassen-like).
    But here we have 3x3 * 3x4.
    
    Let's try a constructive approach:
    We can view the 3x4 output as two 3x2 blocks or four 3x1 blocks.
    
    Alternative: Use the identity that A*B where A is 3x3 and B is 3x4.
    Let B = [B1, B2, B3, B4] where Bi are 3x1 columns.
    Then A*B = [A*B1, A*B2, A*B3, A*B4].
    Each A*Bi is a 3x1 vector, which is a linear combination of the columns of A.
    
    This doesn't immediately help reduce rank below 36 unless we find shared structure.
    
    Let's try to find a decomposition of rank r by solving a system of equations.
    The tensor T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B contribute to output (i,k).
    Wait, the convention is W[k*n+i, r]. So output index is (k, i) where k is row of B (0..m-1) and i is column of A (0..n-1)?
    Let's re-read: "output factor follows the official trace convention W[k*n+i,r]".
    Input U is (n*m, rank), so index is i*m+j? Or i*m + j?
    Input V is (m*p, rank), so index is j*p + k?
    Output W is (p*n, rank), so index is k*n + i?
    
    Standard matrix multiplication: C = A * B.
    A is n x m (3x3). B is m x p (3x4). C is n x p (3x4).
    C[i, k] = sum_j A[i, j] * B[j, k].
    
    The tensor is T[i, j, k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    So the support of the tensor is {(i,j,k) : i in 0..2, j in 0..2, k in 0..3}.
    Total terms: 3*3*4 = 36.
    
    U has rows indexed by (i, j) -> i*m + j = i*3 + j.
    V has rows indexed by (j, k) -> j*p + k = j*4 + k.
    W has rows indexed by (k, i) -> k*n + i = k*3 + i.
    
    We need to find vectors u_c, v_c, w_c for c=1..r such that:
    T[i, j, k] = sum_c U[i*3+j, c] * V[j*4+k, c] * W[k*3+i, c].
    
    This is a tensor decomposition problem.
    
    Let's try a small rank first, say r=20, and see if we can find a solution.
    Since we need exact integer/half-integer solutions, we can use rational arithmetic.
    
    However, solving a system of 36 equations in r*9 + r*12 + r*12 = 33r unknowns is feasible for small r.
    But the equations are bilinear (product of U and V and W), so it's a system of polynomial equations.
    
    A simpler approach: Use known decompositions.
    For 3x3 * 3x3, the rank is 23 (Laderman's algorithm is 23 for 4x4, Strassen is 7 for 2x2).
    Actually, the best known rank for 3x3 matrix multiplication is 23.
    For 3x3 * 3x4, we can pad the 3x4 to 4x4 and use a 4x4 * 4x4 decomposition?
    Or we can use a 3x3 * 3x3 decomposition for the first 3 columns and add one more column.
    
    Let's try to construct a decomposition of rank 23 + something.
    
    Actually, let's just try to find a low-rank decomposition using a random initialization and gradient descent, then round to nearest half-integer.
    But the problem requires EXACT decomposition, so rounding might not work.
    
    Alternative: Use the fact that the tensor is small and brute-force search over small integer coefficients.
    
    Let's try rank 24. We can set up the system of equations and solve it using sympy or a custom solver.
    But sympy is not guaranteed to be available, and it might be slow.
    
    Let's try a different approach: Use the "winograd" style decomposition.
    
    For 3x3 * 3x4, we can write:
    C = A * B
    Let A = [a1, a2, a3] (rows), B = [b1, b2, b3, b4] (columns).
    C[i, k] = a_i . b_k.
    
    We can use the identity:
    (x1+x2)(y1+y2) = x1y1 + x1y2 + x2y1 + x2y2 - (x1-x2)(y1-y2)
    This is the 2x2 Strassen identity.
    
    For 3x3, we can group the 3x3 matrix into 2x2 blocks and use Strassen.
    But 3 is not a power of 2.
    
    Let's try to find a decomposition of rank 23 for 3x3 * 3x3, and then extend it to 3x3 * 3x4.
    
    Actually, let's just try to find a decomposition of rank r by solving the system of equations.
    We'll use a simple iterative method: fix two of the matrices and solve for the third.
    
    But this is complex. Let's try a simpler approach: return the schoolbook decomposition if we can't find a better one quickly.
    
    Given the time constraint, let's try to find a decomposition of rank 24 using a structured approach.
    
    Here's a plan:
    1. Start with the schoolbook decomposition of rank 36.
    2. Try to merge pairs of terms using the Strassen identity.
    3. Each merge reduces the rank by 1.
    4. We can do this for several pairs.
    
    For 3x3 * 3x3, we can do at most 23-23=0 merges? No, Strassen for 2x2 reduces 8 to 7.
    For 3x3, we can partition into 2x2 and 1x1 blocks.
    
    This is getting complicated. Let's just return the schoolbook decomposition for now, as it's guaranteed to work.
    
    But the task asks for a rank-minimal decomposition. The schoolbook rank is 36, which is likely not minimal.
    
    Let's try a random search for a low-rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10 seconds margin
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition
    # We'll try ranks from 20 up to 35
    for target_rank in range(20, min(36, rank_schoolbook)):
        if time.time() > deadline - 5.0:
            break
        
        # Try to find a decomposition of rank target_rank
        # We'll use a random initialization and try to solve the system
        # For simplicity, we'll try a few random initializations
        
        for attempt in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Random initialization with small integers
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check if this decomposition is exact
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # T[i,j,k] should be 1 for all i,j,k
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
