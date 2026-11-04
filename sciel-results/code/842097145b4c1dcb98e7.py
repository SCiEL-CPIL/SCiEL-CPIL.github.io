"""
Program to find a rank-minimal exact tensor decomposition for multiplying
a 3x4 matrix by a 4x7 matrix.

Strategy:
1. Use a recursive block decomposition.
2. For 2x2x2 blocks, use the Strassen decomposition (rank 7).
3. For smaller or odd blocks, use the schoolbook decomposition (rank n*m*p).
4. Combine these blocks to cover the entire 3x4x7 tensor.
5. Verify the decomposition exactly using integer arithmetic or rational arithmetic.

The decomposition is constructed as a sum of rank-1 terms.
Each term corresponds to a linear combination of inputs and outputs.
We represent the decomposition by three matrices U, V, W such that:
   T = sum_c U[:,c] \otimes V[:,c] \otimes W[:,c]
where T is the matrix multiplication tensor.

Indices:
   U is (n*m) x rank
   V is (m*p) x rank
   W is (p*n) x rank
   T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
"""

import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for evaluator
    
    n, m, p = DIMENSIONS
    
    # We will build up the decomposition block by block.
    # U, V, W will be lists of columns (each column is a vector).
    # We use Fractions for exactness.
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    # Helper to add a schoolbook block
    # Block covers rows i0..i0+ni-1, cols j0..j0+nj-1, outputs k0..k0+nk-1
    def add_schoolbook_block(i0, j0, k0, ni, nj, nk):
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    u_vec = [0] * (n*m)
                    v_vec = [0] * (m*p)
                    w_vec = [0] * (p*n)
                    
                    u_vec[(i0+i)*m + (j0+j)] = 1
                    v_vec[(j0+j)*p + (k0+k)] = 1
                    w_vec[(k0+k)*n + (i0+i)] = 1
                    
                    add_term(u_vec, v_vec, w_vec)
    
    # Helper to add a Strassen block (2x2x2)
    # Block covers rows i0, i0+1; cols j0, j0+1; outputs k0, k0+1
    def add_strassen_block(i0, j0, k0):
        # Indices relative to block
        # U indices: (i, j) -> (i0+i)*m + (j0+j)
        # V indices: (j, k) -> (j0+j)*p + (k0+k)
        # W indices: (k, i) -> (k0+k)*n + (i0+i)
        
        # 1. (A00 + A11) * (B00 + B11)
        u1 = [0]*(n*m); u1[(i0)*m + j0] = 1; u1[(i0+1)*m + (j0+1)] = 1
        v1 = [0]*(m*p); v1[j0*p + k0] = 1; v1[(j0+1)*p + (k0+1)] = 1
        w1 = [0]*(p*n); w1[k0*n + i0] = 1; w1[k0*n + (i0+1)] = 1; w1[(k0+1)*n + i0] = 1; w1[(k0+1)*n + (i0+1)] = 1
        # Wait, W is output coefficient.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = (A00+A11)(B00+B11)
        # Contributes to C00 (+1) and C11 (+1)
        # So W vector for M1 has 1 at (k0, i0) and 1 at (k0+1, i0+1)
        w1 = [0]*(p*n); w1[k0*n + i0] = 1; w1[(k0+1)*n + (i0+1)] = 1
        add_term(u1, v1, w1)
        
        # M2 = (A10+A11)B00
        # Contributes to C10 (+1) and C11 (-1)
        u2 = [0]*(n*m); u2[(i0+1)*m + j0] = 1; u2[(i0+1)*m + (j0+1)] = 1
        v2 = [0]*(m*p); v2[j0*p + k0] = 1
        w2 = [0]*(p*n); w2[k0*n + (i0+1)] = 1; w2[(k0+1)*n + (i0+1)] = -1
        add_term(u2, v2, w2)
        
        # M3 = A00(B10-B11)
        # Contributes to C01 (+1) and C11 (+1)
        u3 = [0]*(n*m); u3[i0*m + j0] = 1
        v3 = [0]*(m*p); v3[j0*p + (k0+1)] = 1; v3[(j0+1)*p + (k0+1)] = -1
        w3 = [0]*(p*n); w3[(k0+1)*n + i0] = 1; w3[(k0+1)*n + (i0+1)] = 1
        add_term(u3, v3, w3)
        
        # M4 = A11(B01-B00)
        # Contributes to C00 (+1) and C10 (+1)
        u4 = [0]*(n*m); u4[(i0+1)*m + (j0+1)] = 1
        v4 = [0]*(m*p); v4[(j0+1)*p + k0] = 1; v4[j0*p + k0] = -1
        w4 = [0]*(p*n); w4[k0*n + i0] = 1; w4[k0*n + (i0+1)] = 1
        add_term(u4, v4, w4)
        
        # M5 = (A00+A01)B11
        # Contributes to C00 (-1) and C01 (+1)
        u5 = [0]*(n*m); u5[i0*m + j0] = 1; u5[i0*m + (j0+1)] = 1
        v5 = [0]*(m*p); v5[(j0+1)*p + (k0+1)] = 1
        w5 = [0]*(p*n); w5[k0*n + i0] = -1; w5[(k0+1)*n + i0] = 1
        add_term(u5, v5, w5)
        
        # M6 = (A10-A00)(B00+B01)
        # Contributes to C11 (+1)
        u6 = [0]*(n*m); u6[(i0+1)*m + j0] = 1; u6[i0*m + j0] = -1
        v6 = [0]*(m*p); v6[j0*p + k0] = 1; v6[(j0+1)*p + k0] = 1
        w6 = [0]*(p*n); w6[(k0+1)*n + (i0+1)] = 1
        add_term(u6, v6, w6)
        
        # M7 = (A01-A11)(B10+B11)
        # Contributes to C00 (+1)
        u7 = [0]*(n*m); u7[i0*m + (j0+1)] = 1; u7[(i0+1)*m + (j0+1)] = -1
        v7 = [0]*(m*p); v7[(j0+1)*p + (k0+1)] = 1; v7[j0*p + (k0+1)] = 1
        w7 = [0]*(p*n); w7[k0*n + i0] = 1
        add_term(u7, v7, w7)

    # Decomposition Plan for 3x4x7:
    # Split n=3 into [2, 1]
    # Split m=4 into [2, 2]
    # Split p=7 into [2, 2, 2, 1]
    
    # Blocks:
    # 1. n[0:2], m[0:2], p[0:2] -> Strassen
    # 2. n[0:2], m[0:2], p[2:4] -> Strassen
    # 3. n[0:2], m[0:2], p[4:6] -> Strassen
    # 4. n[0:2], m[0:2], p[6:7] -> Schoolbook (2x2x1)
    # 5. n[0:2], m[2:4], p[0:2] -> Strassen
    # 6. n[0:2], m[2:4], p[2:4] -> Strassen
    # 7. n[0:2], m[2:4], p[4:6] -> Strassen
    # 8. n[0:2], m[2:4], p[6:7] -> Schoolbook (2x2x1)
    # 9. n[2:3], m[0:2], p[0:2] -> Schoolbook (1x2x2)
    # 10. n[2:3], m[0:2], p[2:4] -> Schoolbook (1x2x2)
    # 11. n[2:3], m[0:2], p[4:6] -> Schoolbook (1x2x2)
    # 12. n[2:3], m[0:2], p[6:7] -> Schoolbook (1x2x1)
    # 13. n[2:3], m[2:4], p[0:2] -> Schoolbook (1x2x2)
    # 14. n[2:3], m[2:4], p[2:4] -> Schoolbook (1x2x2)
    # 15. n[2:3], m[2:4], p[4:6] -> Schoolbook (1x2x2)
    # 16. n[2:3], m[2:4], p[6:7] -> Schoolbook (1x2x1)
    
    # Apply Strassen blocks
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply Schoolbook blocks
    # n[0:2], m[0:2], p[6:7]
    add_schoolbook_block(0, 0, 6, 2, 2, 1)
    # n[0:2], m[2:4], p[6:7]
    add_schoolbook_block(0, 2, 6, 2, 2, 1)
    
    # n[2:3], m[0:2], p[0:2]
    add_schoolbook_block(2, 0, 0, 1, 2, 2)
    # n[2:3], m[0:2], p[2:4]
    add_schoolbook_block(2, 0, 2, 1, 2, 2)
    # n[2:3], m[0:2], p[4:6]
    add_schoolbook_block(2, 0, 4, 1, 2, 2)
    # n[2:3], m[0:2], p[6:7]
    add_schoolbook_block(2, 0, 6, 1, 2, 1)
    
    # n[2:3], m[2:4], p[0:2]
    add_schoolbook_block(2, 2, 0, 1, 2, 2)
    # n[2:3], m[2:4], p[2:4]
    add_schoolbook_block(2, 2, 2, 1, 2, 2)
    # n[2:3], m[2:4], p[4:6]
    add_schoolbook_block(2, 2, 4, 1, 2, 2)
    # n[2:3], m[2:4], p[6:7]
    add_schoolbook_block(2, 2, 6, 1, 2, 1)
    
    # Construct final matrices
    rank = len(U_cols)
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for c in range(rank):
        for i in range(n*m):
            if U_cols[c][i] != 0:
                U[i][c] = int(U_cols[c][i])
        for j in range(m*p):
            if V_cols[c][j] != 0:
                V[j][c] = int(V_cols[c][j])
        for k in range(p*n):
            if W_cols[c][k] != 0:
                W[k][c] = int(W_cols[c][k])
                
    # Optional: Verify
    # if time.time() < deadline - 5:
    #     verify(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
