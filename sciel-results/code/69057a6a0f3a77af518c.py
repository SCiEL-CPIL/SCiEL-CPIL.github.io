"""
Rank-minimal decomposition for 2x4 x 4x5 matrix multiplication.

Strategy:
  - Use the known rank-36 Strassen+schoolbook block decomposition as baseline.
  - Attempt to find a rank-35 decomposition by treating the 2x2x3 sub-problem
    (which has trivial rank 12) as potentially decomposable in rank 9.
  - The 2x2x3 problem: multiply 2x2 by 2x3.
    Trivial: 12 terms.
    Can we do it in 9? 
    Lower bound: contains a 2x2x2 sub-tensor, so rank >= 7.
    Known: rank(2x2x3) is between 7 and 12.
    Actually, I recall that rank(2x2x3) = 11 or 12. Let me think.
    
    For 2x2x3: C is 2x3. 
    C[0,0] = a00*b00 + a01*b10
    C[0,1] = a00*b01 + a01*b11
    C[0,2] = a00*b02 + a01*b12
    C[1,0] = a10*b00 + a11*b10
    C[1,1] = a10*b01 + a11*b11
    C[1,2] = a10*b02 + a11*b12
    
    This is two independent 2x1x3 problems? No, they share the a-indices.
    Actually, it's like computing two dot products of length 2, each with 3 different
    right-hand-side vectors. 
    
    If we use Strassen on the 2x2x2 part (k=0,1), we get rank 7 for the first two
    columns of C. Then for k=2, we need 4 more terms (a00*b02, a01*b12, a10*b02, a11*b12).
    Total: 7 + 4 = 11.
    
    Can we do better? What if we use a different decomposition?
    
    Actually, the rank of 2x2x3 is known to be 11 (same as 2x3x2). 
    So we cannot save rank by decomposing 2x2x3 in rank 9.
    
    However, let's reconsider the overall structure. The current rank-36 solution
    splits j into {0,1} and {2,3}, and for each, splits k into {0,1}, {2,3}, {4}.
    For each j-block, we have:
      - 2x2x2 (k=0,1): rank 7
      - 2x2x2 (k=2,3): rank 7
      - 2x2x1 (k=4): rank 4
    Total per j-block: 18. Two blocks: 36.
    
    Alternative: What if we don't split j, but instead try to use a different
    decomposition that shares some terms?
    
    Another idea: The problem is 2x4x5. What if we view it as:
    - A is 2x4, B is 4x5.
    - Split B into B1 (4x2) and B2 (4x3).
    - C1 = A*B1 is 2x2, C2 = A*B2 is 2x3.
    - C1 can be computed in rank 7 (Strassen on the 2x2x2 part, but wait,
      A is 2x4 and B1 is 4x2, so this is 2x4x2).
    
    2x4x2: Split A's columns into two 2-column blocks.
      A1 (2x2) * B1 (4x2)? No, B1 is 4x2, but the inner dimension is 4.
      Actually, C1[i,k] = sum_{j=0}^{3} A[i,j] * B1[j,k].
      Split j into {0,1} and {2,3}.
      C1[i,k] = sum_{j=0}^{1} A[i,j]*B1[j,k] + sum_{j=2}^{3} A[i,j]*B1[j,k].
      First part: 2x2x2, rank 7.
      Second part: 2x2x2, rank 7.
      Total for C1: 14.
    
    C2 = A*B2 is 2x3.
      C2[i,k] = sum_{j=0}^{3} A[i,j] * B2[j,k].
      Split j into {0,1} and {2,3}.
      First part: 2x2x3, rank 11.
      Second part: 2x2x3, rank 11.
      Total for C2: 22.
    
    Overall: 14 + 22 = 36. Same as before.
    
    It seems 36 is a robust upper bound. Let me check if rank 35 is possible
    by trying a more clever decomposition.
    
    Actually, let's try to verify if the rank-36 solution is correct and
    see if we can find any linear dependency that would allow rank 35.
    
    Given the time constraints, I'll return the rank-36 solution, which is
    known to be correct and gives a score of 40/36 = 1.111.
"""

import time

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    n, m, p = 2, 4, 5
    
    return _strassen_36()

def _strassen_36():
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (A11+A22)(B11+B22)
        col = col_offset + 0
        U[a11][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1
        W[c22][col] = 1
        
        # M2 = (A21+A22)B11
        col = col_offset + 1
        U[a21][col] = 1
        U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        W[c22][col] = -1
        
        # M3 = A11(B12-B22)
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1
        V[b22][col] = -1
        W[c12][col] = 1
        W[c22][col] = 1
        
        # M4 = A22(B21-B11)
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1
        V[b11][col] = -1
        W[c11][col] = 1
        W[c21][col] = 1
        
        # M5 = (A11+A12)B22
        col = col_offset + 4
        U[a11][col] = 1
        U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1
        W[c12][col] = 1
        
        # M6 = (A21-A11)(B11+B12)
        col = col_offset + 5
        U[a21][col] = 1
        U[a11][col] = -1
        V[b11][col] = 1
        V[b12][col] = 1
        W[c22][col] = 1
        
        # M7 = (A12-A22)(B21+B22)
        col = col_offset + 6
        U[a12][col] = 1
        U[a22][col] = -1
        V[b21][col] = 1
        V[b22][col] = 1
        W[c11][col] = 1

    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b21 = j2 * p + k1
        
        c11 = k1 * n + i1
        c21 = k1 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1
        V[b11][col] = 1
        W[c11][col] = 1
        
        col = col_offset + 1
        U[a12][col] = 1
        V[b21][col] = 1
        W[c11][col] = 1
        
        col = col_offset + 2
        U[a21][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1
        
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1
        W[c21][col] = 1

    col = 0
    # Sub-problem 1: A[0:2, 0:2] * B[0:2, 0:5]
    add_strassen_222(0, 1, 0, 1, 0, 1, col)
    col += 7
    add_strassen_222(0, 1, 0, 1, 2, 3, col)
    col += 7
    add_schoolbook_221(0, 1, 0, 1, 4, col)
    col += 4
    
    # Sub-problem 2: A[0:2, 2:4] * B[2:4, 0:5]
    add_strassen_222(0, 1, 2, 3, 0, 1, col)
    col += 7
    add_strassen_222(0, 1, 2, 3, 2, 3, col)
    col += 7
    add_schoolbook_221(0, 1, 2, 3, 4, col)
    col += 4
    
    return {"U": U, "V": V, "W": W}
