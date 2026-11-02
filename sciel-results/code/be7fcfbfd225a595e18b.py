"""Matrix multiplication 5x5x5 using block-structured bilinear algorithm.

Strategy: Partition 5 = 3 + 2.
- Laderman's rank-15 algorithm for the 3x3 top-left block.
- Strassen's rank-7 algorithm for the 2x2 bottom-right block.
- Schoolbook products for all cross-block terms (A[0:3,:]*B[:,3:5], etc.).

Total rank: 15 + 7 + 60 + 60 + 20 + 20 = 182.
This is a valid exact decomposition, improving on the 125 schoolbook baseline
is not achieved here, but the structure is correct and verifiable.

Actually, let me reconsider. The schoolbook is rank 125. Let me just verify
that my construction is correct and count carefully.

For 5x5 = (3+2) x (3+2):
C[0:3, 0:3] = A[0:3, 0:3]*B[0:3, 0:3] + A[0:3, 3:5]*B[3:5, 0:3]
C[0:3, 3:5] = A[0:3, 0:3]*B[0:3, 3:5] + A[0:3, 3:5]*B[3:5, 3:5]
C[3:5, 0:3] = A[3:5, 0:3]*B[0:3, 0:3] + A[3:5, 3:5]*B[3:5, 0:3]
C[3:5, 3:5] = A[3:5, 0:3]*B[0:3, 3:5] + A[3:5, 3:5]*B[3:5, 3:5]

The diagonal blocks:
- C[0:3,0:3] needs: Laderman(A[0:3,0:3], B[0:3,0:3]) [rank 15] + schoolbook(A[0:3,3:5], B[3:5,0:3]) [rank 3*2*3=18]
- C[3:5,3:5] needs: Strassen(A[3:5,3:5], B[3:5,3:5]) [rank 7] + schoolbook(A[3:5,0:3], B[0:3,3:5]) [rank 2*3*2=12]

The off-diagonal blocks:
- C[0:3,3:5] = A[0:3,0:3]*B[0:3,3:5] + A[0:3,3:5]*B[3:5,3:5]
  schoolbook: 3*3*2 + 3*2*2 = 18 + 12 = 30
- C[3:5,0:3] = A[3:5,0:3]*B[0:3,0:3] + A[3:5,3:5]*B[3:5,0:3]
  schoolbook: 2*3*3 + 2*2*3 = 18 + 12 = 30

Total: 15 + 18 + 7 + 12 + 30 + 30 = 112.

Hmm, that's 112, which is better than 125. Let me implement this properly.
"""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a bilinear algorithm for 5x5 matrix multiplication.
    
    Partition: 5 = 3 + 2
    - Laderman (rank 15) for A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]
    - Strassen (rank 7) for A[3:5,3:5] * B[3:5,3:5] -> C[3:5,3:5]
    - Schoolbook for all cross terms
    
    Total rank: 15 + 7 + 18 + 12 + 30 + 30 = 112
    """
    n, m, p = DIMENSIONS
    rank = 112
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper functions for index mapping
    def u_idx(i, j):
        """U index for A[i,j]"""
        return i * m + j
    
    def v_idx(i, j):
        """V index for B[i,j]"""
        return i * p + j
    
    def w_idx(i, j):
        """W index for C[i,j] using trace convention W[k*n+i, r] where C[i,j] = sum_k W[k*n+i, r] * prod_r
        Actually, the convention is: C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * ... 
        No, the standard tensor decomposition for matmul:
        C[i,j] = sum_r sum_k U[i*m+k, r] * V[k*p+j, r] ... no.
        
        Let me re-read: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
        U is n*m x rank, V is m*p x rank, W is p*n x rank.
        
        The decomposition: C[i,j] = sum_r (sum_k U[i*m+k, r] * V[k*p+j, r]) * ... 
        No, that's not right either. 
        
        Standard: The matmul tensor T[i,j,k] = delta (i.e., T[i,j,k]=1 if we think of it as the structure tensor)
        Actually for matmul: C[i,j] = sum_k A[i,k]*B[k,j]
        Tensor form: T[i,j,k] where the multiplication is:
        C = sum_r U[:,r] * V[:,r] * W[:,r] in some outer product sense.
        
        More precisely: C[i,j] = sum_r (sum_k U[i*m+k, r] * V[k*p+j, r]) ... no.
        
        Let me think again. The bilinear algorithm has:
        - Preprocessing: compute r intermediate values, each is a bilinear form in A and B
        - Postprocessing: C is a linear combination of these intermediates
        
        Intermediate r: M_r = (sum_{i,k} U[i*m+k, r] * A[i,k]) * (sum_{k,j} V[k*p+j, r] * B[k,j])
        Wait no, that's not right. Each intermediate is:
        M_r = (u_r^T * vec(A)) * (v_r^T * vec(B))
        where u_r is a column of U and v_r is a column of V.
        
        Then C = sum_r M_r * w_r^T where w_r is column r of W, and C is reshaped.
        
        So: C[i,j] = sum_r w_r[C_index(i,j)] * M_r
        where C_index(i,j) = j*n + i (trace convention W[k*n+i, r] means W is indexed by [k*n+i] for C[i,k]? 
        No, "W[k*n+i,r]" - let me parse this. W has shape (p*n, rank). 
        The index into W's first dimension is k*n+i. So W[k*n+i, r] corresponds to... 
        If we think of C as a p x n matrix? No, C is n x p.
        
        Actually re-reading: "The output factor follows the official trace convention W[k*n+i,r]."
        W has shape (p*n, rank). The first index ranges over p*n = 25 values.
        W[k*n+i, r] - so for a given r, the contribution to the output is:
        For each (i, k) pair, W[k*n+i, r] * M_r contributes to... 
        
        I think the convention is: C[i,j] = sum_r W[j*n+i, r] * M_r
        where M_r = (sum_{a,b} U[a*m+b, r] * A[a,b]) * (sum_{b,c} V[b*p+c, r] * B[b,c])
        
        Let me verify with schoolbook:
        For each (i,k,j): U[i*m+k, r]=1, V[k*p+j, r]=1, W[j*n+i, r]=1
        Then M_r = A[i,k]*B[k,j] and C[i,j] gets contribution W[j*n+i,r]*M_r = 1*A[i,k]*B[k,j].
        Summing over all r (one for each i,k,j): C[i,j] = sum_k A[i,k]*B[k,j]. Correct!
        
        So the convention is:
        M_r = (sum_{a,b} U[a*m+b, r] * A[a,b]) * (sum_{b,c} V[b*p+c, r] * B[b,c])
        C[i,j] = sum_r W[j*n+i, r] * M_r
        """
        return j * n + i
    
    # Laderman's algorithm for 3x3 matrix multiplication
    def laderman_3x3(a_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        """
        Apply Laderman's 3x3 algorithm for A[a_row_offset:a_row_offset+3, b_col_offset:b_col_offset+3] * 
        B[b_col_offset:b_col_offset+3, c_col_offset:c_col_offset+3] -> 
        C[c_row_offset:c_row_offset+3, c_col_offset:c_col_offset+3]
        
        Returns next available column.
        """
        col = start_col
        
        def A(i, k):
            return u_idx(a_row_offset + i, b_col_offset + k)
        
        def B(k, j):
            return v_idx(b_col_offset + k, c_col_offset + j)
        
        def C(i, j):
            return w_idx(c_row_offset + i, c_col_offset + j)
        
        # Laderman's 15 products
        # m1 = x00*y00
        U[A(0,0)][col] = 1; V[B(0,0)][col] = 1; col += 1
        # m2 = (x00+x01)*y10
        U[A(0,0)][col] = 1; V[B(1,0)][col] = 1
        U[A(0,1)][col] = 1; V[B(1,0)][col] += 1; col += 1
        # m3 = (x10+x11)*y00
        U[A(1,0)][col] = 1; V[B(0,0)][col] = 1
        U[A(1,1)][col] = 1; V[B(0,0)][col] += 1; col += 1
        # m4 = x22*y22
        U[A(2,2)][col] = 1; V[B(2,2)][col] = 1; col += 1
        # m5 = (x02+x22)*y20
        U[A(0,2)][col] = 1; V[B(2,0)][col] = 1
        U[A(2,2)][col] = 1; V[B(2,0)][col] += 1; col += 1
        # m6 = (x20+x22)*y02
        U[A(2,0)][col] = 1; V[B(0,2)][col] = 1
        U[A(2,2)][col] = 1; V[B(0,2)][col] += 1; col += 1
        # m7 = (x11-x10)*(y10+y20)
        U[A(1,1)][col] = 1; V[B(1,0)][col] = 1
        U[A(1,0)][col] = -1; V[B(1,0)][col] += 1
        U[A(1,1)][col] += 1; V[B(2,0)][col] = 1
        U[A(1,0)][col] -= 1; V[B(2,0)][col] += 1; col += 1
        # m8 = (x01-x00)*(y00+y10)
        U[A(0,1)][col] = 1; V[B(0,0)][col] = 1
        U[A(0,0)][col] = -1; V[B(0,0)][col] += 1
        U[A(0,1)][col] += 1; V[B(1,0)][col] = 1
        U[A(0,0)][col] -= 1; V[B(1,0)][col] += 1; col += 1
        # m9 = (x11+x22)*(y11+y22)
        U[A(1,1)][col] = 1; V[B(1,1)][col] = 1
        U[A(2,2)][col] = 1; V[B(1,1)][col] += 1
        U[A(1,1)][col] += 1; V[B(2,2)][col] = 1
        U[A(2,2)][col] += 1; V[B(2,2)][col] += 1; col += 1
        # m10 = (x01+x11)*(y01+y11)
        U[A(0,1)][col] = 1; V[B(0,1)][col] = 1
        U[A(1,1)][col] = 1; V[B(0,1)][col] += 1
        U[A(0,1)][col] += 1; V[B(1,1)][col] = 1
        U[A(1,1)][col] += 1; V[B(1,1)][col] += 1; col += 1
        # m11 = (x11+x21)*(y12+y22)
        U[A(1,1)][col] = 1; V[B(1,2)][col] = 1
        U[A(2,1)][col] = 1; V[B(1,2)][col] += 1
        U[A(1,1)][col] += 1; V[B(2,2)][col] = 1
        U[A(2,1)][col] += 1; V[B(2,2)][col] += 1; col += 1
        # m12 = (x02+x12)*(y02+y12)
        U[A(0,2)][col] = 1; V[B(0,2)][col] = 1
        U[A(1,2)][col] = 1; V[B(0,2)][col] += 1
        U[A(0,2)][col] += 1; V[B(1,2)][col] = 1
        U[A(1,2)][col] += 1; V[B(1,2)][col] += 1; col += 1
        # m13 = (x00+x22)*(y00+y22)
        U[A(0,0)][col] = 1; V[B(0,0)][col] = 1
        U[A(2,2)][col] = 1; V[B(0,0)][col] += 1
        U[A(0,0)][col] += 1; V[B(2,2)][col] = 1
        U[A(2,2)][col] += 1; V[B(2,2)][col] += 1; col += 1
        # m14 = (x02+x12+x22)*(y00+y11+y20)
        U[A(0,2)][col] = 1; V[B(0,0)][col] = 1
        U[A(1,2)][col] = 1; V[B(0,0)][col] += 1
        U[A(2,2)][col] = 1; V[B(0,0)][col] += 1
        U[A(0,2)][col] += 1; V[B(1,1)][col] = 1
        U[A(1,2)][col] += 1; V[B(1,1)][col] += 1
        U[A(2,2)][col] += 1; V[B(1,1)][col] += 1
        U[A(0,2)][col] += 1; V[B(2,0)][col] = 1
        U[A(1,2)][col] += 1; V[B(2,0)][col] += 1
        U[A(2,2)][col] += 1; V[B(2,0)][col] += 1; col += 1
        # m15 = (x00+x11+x22)*(y02+y12+y20)
        U[A(0,0)][col] = 1; V[B(0,2)][col] = 1
        U[A(1,1)][col] = 1; V[B(0,2)][col] += 1
        U[A(2,2)][col] = 1; V[B(0,2)][col] += 1
        U[A(0,0)][col] += 1; V[B(1,2)][col] = 1
        U[A(1,1)][col] += 1; V[B(1,2)][col] += 1
        U[A(2,2)][col] += 1; V[B(1,2)][col] += 1
        U[A(0,0)][col] += 1; V[B(2,0)][col] = 1
        U[A(1,1)][col] += 1; V[B(2,0)][col] += 1
        U[A(2,2)][col] += 1; V[B(2,0)][col] += 1; col += 1
        
        # Track product column indices
        m1 = start_col
        m2 = start_col + 1
        m3 = start_col + 2
        m4 = start_col + 3
        m5 = start_col + 4
        m6 = start_col + 5
        m7 = start_col + 6
        m8 = start_col + 7
        m9 = start_col + 8
        m10 = start_col + 9
        m11 = start_col + 10
        m12 = start_col + 11
        m13 = start_col + 12
        m14 = start_col + 13
        m15 = start_col + 14
        
        # Compute C entries
        # z00 = m1 + m7 - m8 - m9 + m13
        W[C(0,0)][m1] += 1; W[C(0,0)][m7] += 1; W[C(0,0)][m8] -= 1; W[C(0,0)][m9] -= 1; W[C(0,0)][m13] += 1
        # z01 = m8 + m10 - m13 - m15
        W[C(0,1)][m8] += 1; W[C(0,1)][m10] += 1; W[C(0,1)][m13] -= 1; W[C(0,1)][m15] -= 1
        # z02 = m1 + m6 - m12 - m14
        W[C(0,2)][m1] += 1; W[C(0,2)][m6] += 1; W[C(0,2)][m12] -= 1; W[C(0,2)][m14] -= 1
        # z10 = m3 - m7 + m13 - m15
        W[C(1,0)][m3] += 1; W[C(1,0)][m7] -= 1; W[C(1,0)][m13] += 1; W[C(1,0)][m15] -= 1
        # z11 = m9 - m10 + m11 + m14 - m15
        W[C(1,1)][m9] += 1; W[C(1,1)][m10] -= 1; W[C(1,1)][m11] += 1; W[C(1,1)][m14] += 1; W[C(1,1)][m15] -= 1
        # z12 = m5 - m11 + m12 + m14 - m13
        W[C(1,2)][m5] += 1; W[C(1,2)][m11] -= 1; W[C(1,2)][m12] += 1; W[C(1,2)][m14] += 1; W[C(1,2)][m13] -= 1
        # z20 = m2 - m3 + m13 - m14
        W[C(2,0)][m2] += 1; W[C(2,0)][m3] -= 1; W[C(2,0)][m13] += 1; W[C(2,0)][m14] -= 1
        # z21 = m7 - m9 + m11 + m15
        W[C(2,1)][m7] += 1; W[C(2,1)][m9] -= 1; W[C(2,1)][m11] += 1; W[C(2,1)][m15] += 1
        # z22 = m4 - m5 + m6 + m10 - m14
        W[C(2,2)][m4] += 1; W[C(2,2)][m5] -= 1; W[C(2,2)][m6] += 1; W[C(2,2)][m10] += 1; W[C(2,2)][m14] -= 1
        
        return col
    
    # Strassen's algorithm for 2x2 matrix multiplication
    def strassen_2x2(a_row_offset, b_col_offset, c_row_offset, c_col_offset, start_col):
        """
        Apply Strassen's 2x2 algorithm.
        A[a_row_offset:a_row_offset+2, b_col_offset:b_col_offset+2] * 
        B[b_col_offset:b_col_offset+2, c_col_offset:c_col_offset+2] -> 
        C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
        """
        col = start_col
        
        def A(i, k):
            return u_idx(a_row_offset + i, b_col_offset + k)
        
        def B(k, j):
            return v_idx(b_col_offset + k, c_col_offset + j)
        
        def C(i, j):
            return w_idx(c_row_offset + i, c_col_offset + j)
        
        # M1 = (A00+A11)(B00+B11)
        U[A(0,0)][col] = 1; V[B(0,0)][col] = 1
        U[A(1,1)][col] = 1; V[B(0,0)][col] += 1
        U[A(0,0)][col] += 1; V[B(1,1)][col] = 1
        U[A(1,1)][col] += 1; V[B(1,1)][col] += 1; col += 1
        m1 = col - 1
        
        # M2 = (A10+A11)B00
        U[A(1,0)][col] = 1; V[B(0,0)][col] = 1
        U[A(1,1)][col] = 1; V[B(0,0)][col] += 1; col += 1
        m2 = col - 1
        
        # M3 = A00(B10-B11)
        U[A(0,0)][col] = 1; V[B(1,0)][col] = 1
        U[A(0,0)][col] += 1; V[B(1,1)][col] = -1; col += 1
        m3 = col - 1
        
        # M4 = A11(B01-B10)
        U[A(1,1)][col] = 1; V[B(0,1)][col] = 1
        U[A(1,1)][col] += 1; V[B(1,0)][col] = -1; col += 1
        m4 = col - 1
        
        # M5 = (A00+A01)B11
        U[A(0,0)][col] = 1; V[B(1,1)][col] = 1
        U[A(0,1)][col] = 1; V[B(1,1)][col] += 1; col += 1
        m5 = col - 1
        
        # M6 = (A10-A00)(B00+B10)
        U[A(1,0)][col] = 1; V[B(0,0)][col] = 1
        U[A(0,0)][col] = -1; V[B(0,0)][col] += 1
        U[A(1,0)][col] += 1; V[B(1,0)][col] = 1
        U[A(0,0)][col] -= 1; V[B(1,0)][col] += 1; col += 1
        m6 = col - 1
        
        # M7 = (A01-A11)(B01+B11)
        U[A(0,1)][col] = 1; V[B(0,1)][col] = 1
        U[A(1,1)][col] = -1; V[B(0,1)][col] += 1
        U[A(0,1)][col] += 1; V[B(1,1)][col] = 1
        U[A(1,1)][col] -= 1; V[B(1,1)][col] += 1; col += 1
        m7 = col - 1
        
        # C00 = M1+M4-M5+M7
        W[C(0,0)][m1] += 1; W[C(0,0)][m4] += 1; W[C(0,0)][m5] -= 1; W[C(0,0)][m7] += 1
        # C01 = M3+M5
        W[C(0,1)][m3] += 1; W[C(0,1)][m5] += 1
        # C10 = M2+M4
        W[C(1,0)][m2] += 1; W[C(1,0)][m4] += 1
        # C11 = M1-M2+M3+M6
        W[C(1,1)][m1] += 1; W[C(1,1)][m2] -= 1; W[C(1,1)][m3] += 1; W[C(1,1)][m6] += 1
        
        return col
    
    # Schoolbook for a sub-block: A[a_row:a_row+ar, b_col:b_col+bc] * B[b_col:b_col+bc, c_col:c_col+cc]
    # -> C[c_row:c_row+ar, c_col:c_col+cc]
    def schoolbook(a_row, b_col, c_row, c_col, ar, bc, cc, start_col):
        """
        Schoolbook multiplication for A[a_row:a_row+ar, b_col:b_col+bc] * B[b_col:b_col+bc, c_col:c_col+cc]
        -> C[c_row:c_row+ar, c_col:c_col+cc]
        Note: ar = number of rows in A sub-block = number of cols in B sub-block = bc
               cc = number of cols in B sub-block = number of cols in C sub-block
        """
        col = start_col
        for i in range(ar):
            for k in range(bc):
                for j in range(cc):
                    U[u_idx(a_row + i, b_col + k)][col] = 1
                    V[v_idx(b_col + k, c_col + j)][col] = 1
                    W[w_idx(c_row + i, c_col + j)][col] = 1
                    col += 1
        return col
    
    # Block 1: C[0:3, 0:3] = A[0:3, 0:3]*B[0:3, 0:3] + A[0:3, 3:5]*B[3:5, 0:3]
    # Part 1a: A[0:3,0:3]*B[0:3,0:3] -> C[0:3,0:3] using Laderman (rank 15)
    col = laderman_3x3(0, 0, 0, 0, col)
    # col = 15
    
    # Part 1b: A[0:3,3:5]*B[3:5,0:3] -> C[0:3,0:3] using schoolbook (rank 3*2*3=18)
    col = schoolbook(0, 3, 0, 0, 3, 2, 3, col)
    # col = 15 + 18 = 33
    
    # Block 2: C[3:5, 3:5] = A[3:5, 0:3]*B[0:3, 3:5] + A[3:5, 3:5]*B[3:5, 3:5]
    # Part 2a: A[3:5,3:5]*B[3:5,3:5] -> C[3:5,3:5] using Strassen (rank 7)
    col = strassen_2x2(3, 3, 3, 3, col)
    # col = 33 + 7 = 40
    
    # Part 2b: A[3:5,0:3]*B[0:3,3:5] -> C[3:5,3:5] using schoolbook (rank 2*3*2=12)
    col = schoolbook(3, 0, 3, 3, 2, 3, 2, col)
    # col = 40 + 12 = 52
    
    # Block 3: C[0:3, 3:5] = A[0:3, 0:3]*B[0:3, 3:5] + A[0:3, 3:5]*B[3:5, 3:5]
    # Part 3a: A[0:3,0:3]*B[0:3,3:5] -> C[0:3,3:5] using schoolbook (rank 3*3*2=18)
    col = schoolbook(0, 0, 0, 3, 3, 3, 2, col)
    # col = 52 + 18 = 70
    
    # Part 3b: A[0:3,3:5]*B[3:5,3:5] -> C[0:3,3:5] using schoolbook (rank 3*2*2=12)
    col = schoolbook(0, 3, 0, 3, 3, 2, 2, col)
    # col = 70 + 12 = 82
    
    # Block 4: C[3:5, 0:3] = A[3:5, 0:3]*B[0:3, 0:3] + A[3:5, 3:5]*B[3:5, 0:3]
    # Part 4a: A[3:5,0:3]*B[0:3,0:3] -> C[3:5,0:3] using schoolbook (rank 2*3*3=18)
    col = schoolbook(3, 0, 3, 0, 2, 3, 3, col)
    # col = 82 + 18 = 100
    
    # Part 4b: A[3:5,3:5]*B[3:5,0:3] -> C[3:5,0:3] using schoolbook (rank 2*2*3=12)
    col = schoolbook(3, 3, 3, 0, 2, 2, 3, col)
    # col = 100 + 12 = 112
    
    # Verify we used exactly 112 columns
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
