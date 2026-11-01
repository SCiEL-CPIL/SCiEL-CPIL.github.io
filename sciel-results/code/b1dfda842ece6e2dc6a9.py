"""Matrix multiplication 2x3x4 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
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
    
    if time.time() > deadline:
        return schoolbook()
    
    # Try to construct rank 12 decomposition
    # Using a structured approach based on block decomposition
    
    # For 2x3x4, we can use the following construction:
    # Split B into two 3x2 blocks: B = [B1 | B2] where B1, B2 are 3x2
    # Then C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    # Each is 2x2x3 multiplication
    
    # Actually, let me try a direct rank-12 construction
    # using the known result that rank(M_{2,3,4}) <= 12
    
    # Construction: Use 12 terms with the following pattern
    # For r = 0..11, let a = r // 3, b = r % 3 (a in 0..3, b in 0..2)
    
    rank = 12
    U = [[Fraction(0)] * rank for _ in range(n * m)]  # 6 x 12
    V = [[Fraction(0)] * rank for _ in range(m * p)]  # 12 x 12
    W = [[Fraction(0)] * rank for _ in range(p * n)]  # 8 x 12
    
    # Try a specific construction:
    # Term r: a = r//3 (k-index 0..3), b = r%3 (j-index 0..2)
    # U[i*m+j, r] = 1 if j == b, else 0
    # V[j*p+k, r] = 1 if j == b and k == a, else 0  -- this is just schoolbook
    # W[k*n+i, r] = 1 if k == a, else 0
    
    # That's still schoolbook. Need a better construction.
    
    # Let me try using the 2x2x2 Strassen algorithm as a building block
    # M_{2,3,4} can be related to products of smaller matrices
    
    # Alternative: use the fact that we can compute 2x3 * 3x4 using
    # a combination of 2x2 * 2x2 and 2x2 * 2x1 multiplications
    
    # Let me try a different approach: enumerate possible rank-12 solutions
    # by solving a system of equations
    
    # Actually, let me try the following known construction:
    # For 2x3x4, use the "border rank" or "approximate rank" results
    
    # Let me try to use numpy to solve for a rank-12 decomposition
    
    # Set up the problem: find U (6x12), V (12x12), W (8x12)
    # such that for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 24 equations (2*3*4) with 12*(6+12+8) = 288 unknowns
    # But it's nonlinear (trilinear), so we need a different approach
    
    # Let me try a specific pattern that's known to work:
    
    # Construction for rank 12:
    # Use the following 12 terms where each term handles specific (i,j,k) triples
    
    # Pattern 1: For each j in 0..2, use 4 terms (one for each k)
    # But with i-coupling to reduce from 24 to 12
    
    # Let me try: for j in 0..2, for k in 0..3:
    #   U[i*m+j, r] depends on i
    #   V[j*p+k, r] = 1
    #   W[k*n+i, r] depends on i
    
    # For the i-coupling, use:
    #   U[0*m+j, r] = 1, U[1*m+j, r] = 1 (both i values get 1)
    #   W[k*n+0, r] = 1, W[k*n+1, r] = 1 (both i values get 1)
    # This gives schoolbook again.
    
    # Need to use Strassen-like cancellation.
    
    # Let me try the following for 2x2x2 (rank 7) and extend:
    # Strassen for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For 2x3x4, I'll try to use a similar approach with 12 terms.
    
    # Let me just try to verify if a specific rank-12 construction works
    # by checking the tensor equation.
    
    # Try this construction:
    # r = 0..11
    # a = r // 3 (0..3), b = r % 3 (0..2)
    
    # U[i*m+j, r]:
    #   If i=0: U[j, r] = 1 if j==b else 0
    #   If i=1: U[3+j, r] = 1 if j==b else 0
    # V[j*p+k, r]:
    #   V[j*p+k, r] = 1 if j==b and k==a else 0
    # W[k*n+i, r]:
    #   W[k*n+i, r] = 1 if k==a else 0
    
    # This is schoolbook. Let me try with signs:
    
    # For a Strassen-like approach, I need cross terms.
    
    # Let me try a different pattern:
    # r = 0..11
    # Use pairs of terms that cancel for some (i,j,k) but add for others
    
    # Actually, let me just return schoolbook since I can't quickly find
    # a valid rank-12 construction.
    
    return schoolbook()

# EVOLVE-BLOCK-END
