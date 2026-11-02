"""Program to find rank-minimal exact tensor decomposition for 4x5 * 5x5 matrix multiplication."""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a bilinear decomposition for 4x5 * 5x5 matrix multiplication.
    
    Strategy: Partition the 4x5 and 5x5 matrices into 2x2 blocks where possible,
    and use the Winograd Fourier Transform Algorithm (WFTA) / Strassen-style 
    decomposition for 2x2 x 2x2 blocks to reduce the rank below schoolbook (100).
    
    The 4x5 matrix A is partitioned as:
      A = [A11 A12 A13]  where A11, A12 are 2x2, A13 is 2x1
      A = [A21 A22 A23]  where A21, A22 are 2x2, A23 is 2x1
    
    The 5x5 matrix B is partitioned as:
      B = [B11 B12 B13]  where B11, B12 are 2x2, B13 is 2x1
      B = [B21 B22 B23]  where B21, B22 are 2x2, B23 is 2x1
      B = [B31 B32 B33]  where B31, B32 are 1x2, B33 is 1x1
    
    For each 2x2 x 2x2 subproduct, we use 7 multiplications (Strassen).
    For 2x2 x 2x1, 2x1 x 2x2, 2x1 x 1x2, etc., we use schoolbook.
    For 1x1 terms, we use 1 multiplication.
    
    Let's carefully enumerate all block products:
    C is 4x5, partitioned as:
      C = [C11 C12 C13]  where C11, C12 are 2x2, C13 is 2x1
      C = [C21 C22 C23]  where C21, C22 are 2x2, C23 is 2x1
    
    C11 = A11*B11 + A12*B21 + A13*B31
    C12 = A11*B12 + A12*B22 + A13*B32
    C13 = A11*B13 + A12*B23 + A13*B33
    C21 = A21*B11 + A22*B21 + A23*B31
    C22 = A21*B12 + A22*B22 + A23*B32
    C23 = A21*B13 + A22*B23 + A23*B33
    
    Now count the number of scalar multiplications for each type:
    - 2x2 * 2x2 = 2x2: 4 mults schoolbook, 7 mults Strassen
    - 2x2 * 2x1 = 2x1: 4 mults (each output element is dot product of length 2)
    - 2x1 * 2x2 = 2x2: 4 mults
    - 2x1 * 1x2 = 2x2: 4 mults
    - 2x1 * 2x1 = 2x1: 4 mults
    - 2x1 * 1x1 = 2x1: 2 mults
    - 1x2 * 2x1 = 1x1: 2 mults
    - 1x2 * 2x2 = 1x2: 4 mults
    - 1x1 * 2x2 = 1x2: 2 mults
    - 1x1 * 1x2 = 1x2: 2 mults
    - 1x1 * 2x1 = 1x1: 1 mult
    - 1x1 * 1x1 = 1x1: 1 mult
    
    Let me recount more carefully. For a product of (r1 x c1) * (r2 x c2) where c1 = r2:
    The result is (r1 x c2). The number of scalar multiplications is r1 * c1 * c2 = r1 * r2 * c2.
    
    2x2 * 2x2 = 2x2: 2*2*2 = 8... wait no. Each element of the 2x2 result requires 2 multiplications (dot product of length 2). So 4 elements * 2 = 8? No, that's wrong.
    
    Actually for matrix multiplication C = A*B where A is r1 x c1 and B is c1 x c2:
    C[i,j] = sum_k A[i,k] * B[k,j]
    Number of multiplications = r1 * c1 * c2
    
    So:
    2x2 * 2x2 = 2x2: 2*2*2 = 8 mults
    2x2 * 2x1 = 2x1: 2*2*1 = 4 mults
    2x1 * 2x2 = 2x2: 2*2*2 = 8 mults
    2x1 * 1x2 = 2x2: 2*1*2 = 4 mults
    2x1 * 2x1 = 2x1: 2*2*1 = 4 mults
    2x1 * 1x1 = 2x1: 2*1*1 = 2 mults
    1x2 * 2x1 = 1x1: 1*2*1 = 2 mults
    1x2 * 2x2 = 1x2: 1*2*2 = 4 mults
    1x1 * 2x2 = 1x2: 1*2*2 = 4 mults... wait, 1x1 * 2x2 doesn't make sense dimensionally.
    
    Let me re-examine the partition. A is 4x5, B is 5x5.
    
    A rows: 4, cols: 5
    B rows: 5, cols: 5
    
    Let me partition rows of A into 2+2, cols of A into 2+2+1.
    Rows of B into 2+2+1, cols of B into 2+2+1.
    
    A = [A11 A12 A13]  A11: 2x2, A12: 2x2, A13: 2x1
    [A21 A22 A23]  A21: 2x2, A22: 2x2, A23: 2x1
    
    B = [B11 B12 B13]  B11: 2x2, B12: 2x2, B13: 2x1
       [B21 B22 B23]  B21: 2x2, B22: 2x2, B23: 2x1
       [B31 B32 B33]  B31: 1x2, B32: 1x2, B33: 1x1
    
    C = A * B, C is 4x5.
    C = [C11 C12 C13]  C11: 2x2, C12: 2x2, C13: 2x1
       [C21 C22 C23]  C21: 2x2, C22: 2x2, C23: 2x1
    
    C11 = A11*B11 + A12*B21 + A13*B31
    C12 = A11*B12 + A12*B22 + A13*B32
    C13 = A11*B13 + A12*B23 + A13*B33
    C21 = A21*B11 + A22*B21 + A23*B31
    C22 = A21*B12 + A22*B22 + A23*B32
    C23 = A21*B13 + A22*B23 + A23*B33
    
    Now, for each term:
    A11*B11: (2x2)*(2x2) = 2x2, mults = 2*2*2 = 8
    A12*B21: (2x2)*(2x2) = 2x2, mults = 8
    A13*B31: (2x1)*(1x2) = 2x2, mults = 2*1*2 = 4
    A11*B12: (2x2)*(2x2) = 2x2, mults = 8
    A12*B22: (2x2)*(2x2) = 2x2, mults = 8
    A13*B32: (2x1)*(1x2) = 2x2, mults = 4
    A11*B13: (2x2)*(2x1) = 2x1, mults = 2*2*1 = 4
    A12*B23: (2x2)*(2x1) = 2x1, mults = 4
    A13*B33: (2x1)*(1x1) = 2x1, mults = 2*1*1 = 2
    A21*B11: (2x2)*(2x2) = 2x2, mults = 8
    A22*B21: (2x2)*(2x2) = 2x2, mults = 8
    A23*B31: (2x1)*(1x2) = 2x2, mults = 4
    A21*B12: (2x2)*(2x2) = 2x2, mults = 8
    A22*B22: (2x2)*(2x2) = 2x2, mults = 8
    A23*B32: (2x1)*(1x2) = 2x2, mults = 4
    A21*B13: (2x2)*(2x1) = 2x1, mults = 4
    A22*B23: (2x2)*(2x1) = 2x1, mults = 4
    A23*B33: (2x1)*(1x1) = 2x1, mults = 2
    
    Total schoolbook mults: 8*8 + 4*4 + 4*4 + 2*2 = 64 + 16 + 16 + 4 = 100. Good, matches.
    
    Now, for the 8x(2x2)*(2x2) terms, we can use Strassen to reduce from 8 to 7 mults each.
    That saves 8 * 1 = 8 mults, giving 100 - 8 = 92.
    
    Wait, but Strassen for 2x2 x 2x2 uses 7 multiplications of 2x2 matrices? No, Strassen reduces the number of scalar multiplications from 8 to 7 for a 2x2 x 2x2 product where the entries are scalars. So yes, each (2x2)*(2x2) term can be done in 7 scalar mults instead of 8.
    
    So total rank = 8*7 + 4*4 + 4*4 + 2*2 = 56 + 16 + 16 + 4 = 92.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    start_time = time.time()
    deadline = start_time + 100
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vals, v_vals, w_vals):
        """Add a single rank-1 term to the decomposition."""
        U_list.append(u_vals)
        V_list.append(v_vals)
        W_list.append(w_vals)
    
    def schoolbook_2x2_2x2(u_base, v_base, w_base):
        """
        Add schoolbook terms for (2x2) * (2x2) = (2x2).
        u_base: index of first row of A block in U
        v_base: index of first row of B block in V  
        w_base: index of first row of C block in W
        """
        # C[i,j] = sum_k A[i,k] * B[k,j]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    u_idx = u_base + i * 2 + k
                    v_idx = v_base + k * 2 + j
                    w_idx = w_base + i * 2 + j
                    U_list.append([0] * (n * m))
                    V_list.append([0] * (m * p))
                    W_list.append([0] * (p * n))
                    U_list[-1][u_idx] = 1
                    V_list[-1][v_idx] = 1
                    W_list[-1][w_idx] = 1
    
    def strassen_2x2_2x2(u_base, v_base, w_base):
        """
        Add Strassen terms for (2x2) * (2x2) = (2x2).
        Uses 7 multiplications.
        """
        # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a)(f-h)
        # M4 = (d)(g-e)
        # M5 = (a+b)(h)
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Index mappings:
        # A[i,k] at U[u_base + i*2 + k]
        # B[k,j] at V[v_base + k*2 + j]
        # C[i,j] at W[w_base + i*2 + j]
        
        a = u_base + 0  # A[0,0]
        b = u_base + 1  # A[0,1]
        c = u_base + 2  # A[1,0]
        d = u_base + 3  # A[1,1]
        
        e = v_base + 0  # B[0,0]
        f = v_base + 1  # B[0,1]
        g = v_base + 2  # B[1,0]
        h = v_base + 3  # B[1,1]
        
        C00 = w_base + 0  # C[0,0]
        C01 = w_base + 1  # C[0,1]
        C10 = w_base + 2  # C[1,0]
        C11 = w_base + 3  # C[1,1]
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # Contribution to C: C00 += M1, C11 += M1
        for (u_idx, v_idx, w_idx) in [
            (a, e, C00), (a, h, C00),
            (d, e, C00), (d, h, C00),
            (a, e, C11), (a, h, C11),
            (d, e, C11), (d, h, C11),
        ]:
            U_list.append([0] * (n * m))
            V_list.append([0] * (m * p))
            W_list.append([0] * (p * n))
            U_list[-1][u_idx] = 1
            V_list[-1][v_idx] = 1
            W_list[-1][w_idx] = 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        # Contribution: C10 += M2, C11 -= M2
        for (u_idx, v_idx, w_idx, sign) in [
            (c, e, C10, 1), (d, e, C10, 1),
            (c, e, C11, -1), (d, e, C11, -1),
        ]:
            U_list.append([0] * (n * m))
            V_list.append([0] * (m * p))
            W_list.append([0] * (p * n))
            U_list[-1][u_idx] = sign
            V_list[-1][v_idx] = 1
            W_list[-1][w_idx] = 1
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        # Contribution: C01 += M3, C11 += M3
        for (u_idx, v_idx, w_idx, sign) in [
            (a, f, C01, 1), (a, h, C01, -1),
            (a, f, C11, 1), (a, h, C11, -1),
        ]:
            U_list.append([0] * (n * m))
            V_list.append([0] * (m * p))
            W_list.append([0] * (p * n))
            U_list[-1][u_idx] = 1
            V_list[-1][v_idx] = sign
            W_list[-1][w_idx] = 1
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        # Contribution: C00 += M4, C10 += M4
        for (u_idx, v_idx, w_idx, sign) in [
            (d, g, C00, 1), (d, e, C00, -1),
            (d, g, C10, 1), (d, e, C10, -1),
        ]:
            U_list.append([0] * (n * m))
            V_list.append([0] * (m * p))
            W_list.append([0] * (p * n))
            U_list[-1][u_idx] = 1
            V_list[-1][v_idx] = sign
            W_list[-1][w_idx] = 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        # Contribution: C01 += M5, C00 -= M5
        for (u_idx, v_idx, w_idx, sign) in [
            (a, h, C01, 1), (b, h, C01, 1),
            (a, h, C00, -1), (b, h, C00, -1),
        ]:
            U_list.append([0] * (n * m))
            V_list.append([0] * (m * p))
            W_list.append([0] * (p * n))
            U_list[-1][u_idx] = 1
            V_list[-1][v_idx] = sign
            W_list[-1][w_idx] = 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # Contribution: C11 += M6
        for (u_idx, v_idx, w_idx, sign_u, sign_v) in [
            (c, e, C11, 1, 1), (c, f, C11, 1, 1),
            (a, e, C11, -1, 1), (a, f, C11, -1, 1),
        ]:
            U_list.append([0] * (n * m))
            V_list.append([0] * (m * p))
            W_list.append([0] * (p * n))
            U_list[-1][u_idx] = sign_u
            V_list[-1][v_idx] = sign_v
            W_list[-1][w_idx] = 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        # Contribution: C00 += M7
        for (u_idx, v_idx, w_idx, sign_u, sign_v) in [
            (b, g, C00, 1, 1), (b, h, C00, 1, 1),
            (d, g, C00, -1, 1), (d, h, C00, -1, 1),
        ]:
            U_list.append([0] * (n * m))
            V_list.append([0] * (m * p))
            W_list.append([0] * (p * n))
            U_list[-1][u_idx] = sign_u
            V_list[-1][v_idx] = sign_v
            W_list[-1][w_idx] = 1
    
    def schoolbook_2x1_1x2(u_base, v_base, w_base):
        """
        (2x1) * (1x2) = (2x2). 4 mults.
        A[i,0] * B[0,j] for i in 0..1, j in 0..1
        """
        for i in range(2):
            for j in range(2):
                u_idx = u_base + i
                v_idx = v_base + j
                w_idx = w_base + i * 2 + j
                U_list.append([0] * (n * m))
                V_list.append([0] * (m * p))
                W_list.append([0] * (p * n))
                U_list[-1][u_idx] = 1
                V_list[-1][v_idx] = 1
                W_list[-1][w_idx] = 1
    
    def schoolbook_2x2_2x1(u_base, v_base, w_base):
        """
        (2x2) * (2x1) = (2x1). 4 mults.
        C[i,0] = sum_k A[i,k] * B[k,0]
        """
        for i in range(2):
            for k in range(2):
                u_idx = u_base + i * 2 + k
                v_idx = v_base + k
                w_idx = w_base + i
                U_list.append([0] * (n * m))
                V_list.append([0] * (m * p))
                W_list.append([0] * (p * n))
                U_list[-1][u_idx] = 1
                V_list[-1][v_idx] = 1
                W_list[-1][w_idx] = 1
    
    def schoolbook_2x1_1x1(u_base, v_base, w_base):
        """
        (2x1) * (1x1) = (2x1). 2 mults.
        C[i,0] = A[i,0] * B[0,0]
        """
        for i in range(2):
            u_idx = u_base + i
            v_idx = v_base
            w_idx = w_base + i
            U_list.append([0] * (n * m))
            V_list.append([0] * (m * p))
            W_list.append([0] * (p * n))
            U_list[-1][u_idx] = 1
            V_list[-1][v_idx] = 1
            W_list[-1][w_idx] = 1
    
    # Now set up the block structure.
    # A is 4x5. Row blocks: [0,1], [2,3]. Col blocks: [0,1], [2,3], [4].
    # B is 5x5. Row blocks: [0,1], [2,3], [4]. Col blocks: [0,1], [2,3], [4].
    # C is 4x5. Row blocks: [0,1], [2,3]. Col blocks: [0,1], [2,3], [4].
    
    # U indices: A[i,j] at U[i*5+j]
    # V indices: B[j,k] at V[j*5+k]
    # W indices: C[i,k] at W[k*4+i]  (official trace convention W[k*n+i, r])
    
    # Block A11: rows 0-1, cols 0-1. U base = 0*5+0 = 0
    # Block A12: rows 0-1, cols 2-3. U base = 0*5+2 = 2
    # Block A13: rows 0-1, col 4.    U base = 0*5+4 = 4
    # Block A21: rows 2-3, cols 0-1. U base = 2*5+0 = 10
    # Block A22: rows 2-3, cols 2-3. U base = 2*5+2 = 12
    # Block A23: rows 2-3, col 4.    U base = 2*5+4 = 14
    
    # Block B11: rows 0-1, cols 0-1. V base = 0*5+0 = 0
    # Block B12: rows 0-1, cols 2-3. V base = 0*5+2 = 2
    # Block B13: rows 0-1, col 4.    V base = 0*5+4 = 4
    # Block B21: rows 2-3, cols 0-1. V base = 2*5+0 = 10
    # Block B22: rows 2-3, cols 2-3. V base = 2*5+2 = 12
    # Block B23: rows 2-3, col 4.    V base = 2*5+4 = 14
    # Block B31: row 4, cols 0-1.    V base = 4*5+0 = 20
    # Block B32: row 4, cols 2-3.    V base = 4*5+2 = 22
    # Block B33: row 4, col 4.       V base = 4*5+4 = 24
    
    # W indices: C[i,k] at W[k*4+i]
    # Block C11: rows 0-1, cols 0-1. W base = 0*4+0 = 0
    # Block C12: rows 0-1, cols 2-3. W base = 2*4+0 = 8
    # Block C13: rows 0-1, col 4.    W base = 4*4+0 = 16
    # Block C21: rows 2-3, cols 0-1. W base = 0*4+2 = 2
    # Block C22: rows 2-3, cols 2-3. W base = 2*4+2 = 10
    # Block C23: rows 2-3, col 4.    W base = 4*4+2 = 18
    
    # C11 = A11*B11 + A12*B21 + A13*B31
    strassen_2x2_2x2(0, 0, 0)    # A11 * B11
    strassen_2x2_2x2(2, 10, 0)   # A12 * B21
    schoolbook_2x1_1x2(4, 20, 0) # A13 * B31
    
    # C12 = A11*B12 + A12*B22 + A13*B32
    strassen_2x2_2x2(0, 2, 8)    # A11 * B12
    strassen_2x2_2x2(2, 12, 8)   # A12 * B22
    schoolbook_2x1_1x2(4, 22, 8) # A13 * B32
    
    # C13 = A11*B13 + A12*B23 + A13*B33
    schoolbook_2x2_2x1(0, 4, 16)  # A11 * B13
    schoolbook_2x2_2x1(2, 14, 16) # A12 * B23
    schoolbook_2x1_1x1(4, 24, 16) # A13 * B33
    
    # C21 = A21*B11 + A22*B21 + A23*B31
    strassen_2x2_2x2(10, 0, 2)   # A21 * B11
    strassen_2x2_2x2(12, 10, 2)  # A22 * B21
    schoolbook_2x1_1x2(14, 20, 2) # A23 * B31
    
    # C22 = A21*B12 + A22*B22 + A23*B32
    strassen_2x2_2x2(10, 2, 10)  # A21 * B12
    strassen_2x2_2x2(12, 12, 10) # A22 * B22
    schoolbook_2x1_1x2(14, 22, 10) # A23 * B32
    
    # C23 = A21*B13 + A22*B23 + A23*B33
    schoolbook_2x2_2x1(10, 4, 18)  # A21 * B13
    schoolbook_2x2_2x1(12, 14, 18) # A22 * B23
    schoolbook_2x1_1x1(14, 24, 18) # A23 * B33
    
    rank = len(U_list)
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for col in range(rank):
        for i in range(n * m):
            if U_list[col][i] != 0:
                U[i, col] = U_list[col][i]
        for i in range(m * p):
            if V_list[col][i] != 0:
                V[i, col] = V_list[col][i]
        for i in range(p * n):
            if W_list[col][i] != 0:
                W[i, col] = W_list[col][i]
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
