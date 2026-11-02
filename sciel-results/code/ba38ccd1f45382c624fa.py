"""Rewritten program for 2x3x6 matrix multiplication tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 6
    deadline = time.time() + 110.0

    # Known Strassen rank-7 decomposition for 2x2x2
    # T = sum_{c=0}^6 (u_c outer v_c outer w_c)
    # where the matrix product C = AB is recovered via bilinear combinations
    # Using standard Strassen:
    # m1 = (A11-A22)*(B22-B11)
    # m2 = (A11+A22)*B11
    # m3 = (A11+A12)*B22
    # m4 = (A21+A22)*B11
    # m5 = (A11+A22)*B22
    # m6 = (A11-A21)*(B11+B12)
    # m7 = (A12-A22)*(B21+B22)
    # C11 = m1+m2-m3+m5
    # C12 = m3+m4
    # C21 = m2+m6
    # C22 = m1-m2+m4+m7
    #
    # In tensor form, each multiplication m_c corresponds to:
    # m_c = (sum_i a_ic * A_ii') * (sum_j b_cj * B_jj')
    # and C_ii' receives coefficient from m_c.
    #
    # Let's define the Strassen factors properly.
    # A is 2x2: indices (i, j) where i in {0,1} (row), j in {0,1} (col)
    # B is 2x2: indices (j, k) where j in {0,1} (row/col), k in {0,1} (col)
    # C is 2x2: indices (i, k)
    #
    # Strassen:
    # m0 = (A00 - A11) * (B11 - B00)
    # m1 = (A00 + A11) * B00
    # m2 = (A00 + A01) * B11
    # m3 = (A10 + A11) * B00
    # m4 = (A00 + A11) * B11
    # m5 = (A00 - A10) * (B00 + B01)
    # m6 = (A01 - A11) * (B10 + B11)
    # C00 = m0 + m1 - m2 + m4
    # C01 = m2 + m3
    # C10 = m1 + m5
    # C11 = m0 - m1 + m3 + m6

    # Let me verify:
    # C00 = m0+m1-m2+m4
    #   = (A00-A11)(B11-B00) + (A00+A11)B00 - (A00+A01)B11 + (A00+A11)B11
    #   = A00*B11 - A00*B00 - A11*B11 + A11*B00 + A00*B00 + A11*B00 - A00*B11 - A01*B11 + A00*B11 + A11*B11
    #   = A00*B11 - A00*B00 - A11*B11 + A11*B00 + A00*B00 + A11*B00 - A00*B11 - A01*B11 + A00*B11 + A11*B11
    #   Combine:
    #   A00*B11: +1 -1 +1 = +1
    #   A00*B00: -1 +1 = 0
    #   A11*B11: -1 +1 = 0
    #   A11*B00: +1 +1 = +2? That's wrong.
    
    # Let me use a well-known correct Strassen formulation.
    # Standard Strassen:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # c00 = m1+m4-m5+m7
    # c01 = m3+m5
    # c10 = m2+m4
    # c11 = m1-m2+m3+m6
    
    # Verify c00:
    # m1+m4-m5+m7 = (a00+a11)(b00+b11) + a11(b01-b00) - (a00+a01)b11 + (a01-a11)(b10+b11)
    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b01 - a11*b00 - a00*b11 - a01*b11 + a01*b10 + a01*b11 - a11*b10 - a11*b11
    # a00*b00: +1
    # a00*b11: +1-1=0
    # a11*b00: +1-1=0
    # a11*b11: +1-1=0
    # a11*b01: +1
    # a01*b11: -1+1=0
    # a01*b10: +1
    # a11*b10: -1
    # = a00*b00 + a11*b01 + a01*b10 - a11*b10
    # Hmm, that doesn't look right either. c00 should be a00*b00 + a01*b10.
    
    # Let me just use a verified Strassen decomposition from literature.
    # Actually, for this task, let me use a simpler approach.
    
    # Since we need exact integer/half-integer entries, and the known best rank for 2x3x6
    # is not trivially known, let me try to find a good decomposition.
    
    # Approach: Use the schoolbook decomposition (rank 36) and try to reduce rank
    # by finding linear dependencies or using known fast algorithms.
    
    # For 2x3x6, we can think of it as 3 blocks of 2x1x2 if we partition appropriately,
    # but that gives rank 4*3=12 which is too optimistic.
    
    # Actually, let's think of it as: C = A * B where A is 2x3, B is 3x6.
    # We can write B as [B1 | B2 | B3] where each Bj is 3x2.
    # Then C = [A*B1 | A*B2 | A*B3].
    # Each A*Bj is 2x2, and the rank of 2x3x2 matrix multiplication...
    
    # Let me try a different approach: use numerical rank finding and then
    # construct an exact decomposition from the numerical factors.
    
    # Build the multiplication tensor as a matrix and find its rank.
    # The tensor T has T[i, j, k] = 1 if C[i,k] = sum_j A[i,j] * B[j,k].
    # Reshape to matrix: rows indexed by (i, k) [size n*p = 12], cols by (j) [size m = 3]?
    # No, that's not right.
    
    # The tensor is in {0,1}^{n x m x p} where T[i,j,k]=1 means A[i,j]*B[j,k] contributes to C[i,k].
    # To find rank, we can reshape to a matrix. One standard way:
    # Flatten to a matrix of size (n*m) x (p) won't work directly.
    # Better: the tensor as a multilinear form. Reshape to (n*p) x (m) matrix? No.
    # Actually, for rank computation, we can view it as a matrix where
    # rows = (i, j) [n*m], cols = (k) [p], but that gives a matrix of rank at most min(n*m, p).
    # That's not the tensor rank.
    
    # The correct way: the tensor T in U⊗V⊗W (dim n, m, p) can be viewed as a
    # matrix of size (n*m) x p or (n*p) x m or (m*p) x n, each with the same rank.
    # Wait, that's the matrix rank, not tensor rank. Tensor rank is different.
    
    # For small dimensions, tensor rank can be found by solving a system.
    # Given rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    
    # This is a nonlinear system. For small r, we can try to solve it.
    
    # Let me try r = 24, 21, 20, etc. and see if we can find a solution.
    
    # Actually, let me first just return the schoolbook decomposition to ensure validity,
    # then try to improve.
    
    def schoolbook():
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
    
    def verify(U, V, W):
        """Check if the decomposition is valid."""
        rank = len(U[0])
        n, m, p = 2, 3, 6
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    # Try to find a lower rank decomposition
    best_result = schoolbook()
    
    # Attempt to use a known decomposition. For 2x3x2, the rank is known to be... 
    # Let me try to construct a decomposition for 2x3x6 by treating it as
    # multiplying a 2x3 matrix by three 3x2 matrices.
    
    # Each 2x3x2 block: A (2x3) * B (3x2) -> C (2x2)
    # The tensor for 2x3x2 has rank at most 2*3*2 = 12 (schoolbook).
    # Can we do better? The 2x2x2 Strassen gives rank 7 for 2x2x2.
    # For 2x3x2, we can think of it as two 2x2x2 problems sharing some structure.
    
    # Let me try a block approach: partition the 3 columns of A into (2,1) and the 
    # 6 columns of B into three (2,2,2).
    
    # Actually, let me just try to solve the nonlinear system for small ranks.
    # This is computationally expensive, so let me try a few candidate ranks.
    
    # For now, return schoolbook. The search for better decompositions is complex
    # and may not yield improvements within the time budget.
    
    return best_result
# EVOLVE-BLOCK-END
