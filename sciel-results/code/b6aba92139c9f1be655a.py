"""Rewritten program: build a rank-44 block decomposition, then optionally search lower ranks."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def _matmul_tensor(n, m, p):
    """Return the matrix-multiplication tensor with indices (i,j,k) -> W-index convention."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _verify_decomp(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[:,c] V[:,c] W[:,c] reconstructs the matmul tensor in W[k*n+i, r] convention."""
    rank = len(U[0])
    # Build the implied tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                T[i, j, k] = val
    # All entries should be 1
    return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

def _build_block_44():
    """
    Build a rank-44 decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy: Split the 3x4 matrix A into 3 blocks of size 1x4 (rows).
    Actually, let's think of A as 3x4 and B as 4x4.
    
    A cleaner block approach:
    - A is (3,4), B is (4,4). 
    - We can partition A into two parts: A1 = A[0:2, :] (2x4) and A2 = A[2:3, :] (1x4).
    - Then C = A*B = [A1*B; A2*B] where A1*B is 2x4 and A2*B is 1x4.
    
    For A1 (2x4) * B (4x4): Use a rank-10 decomposition? 
    Actually, 2x4 * 4x4: schoolbook is 2*4*4 = 32.
    We can do better by treating the 4x4 B as two 4x2 blocks? 
    Or use known results: 2x4 * 4x4 can be done in rank <= 24? Not sure.
    
    Alternative: Use the known rank-10 for 2x2*2x2 (Strassen).
    Partition the 4x4 matrix B into four 2x2 blocks:
    B = [[B11, B12], [B21, B22]]
    
    A is 3x4. Partition A into:
    A = [A1 | A2] where A1 is 3x2 and A2 is 3x2.
    
    Then C = [A1*B11 + A2*B21, A1*B12 + A2*B22].
    
    Each term A1*B11 is 3x2 * 2x2 = 3x2.
    For 3x2 * 2x2: We can partition A1 into A11 (2x2) and A12 (1x2).
    A1*B11 = A11*B11 + A12*B11.
    A11*B11: 2x2 * 2x2 -> rank 7 (Strassen).
    A12*B11: 1x2 * 2x2 -> rank 4 (schoolbook, or can we do rank 3? 1x2*2x2 = 4 mults. Rank is 4).
    
    So A1*B11 costs 7 + 4 = 11.
    Similarly A1*B12: 11.
    A2*B21: 11.
    A2*B22: 11.
    Total: 44.
    
    This matches the known rank-44 result!
    """
    n, m, p = 3, 4, 4
    
    U = []  # list of vectors (n*m,)
    V = []  # list of vectors (m*p,)
    W = []  # list of vectors (p*n,)
    
    # We need to build 44 rank-1 terms.
    # The structure:
    # C[0:3, 0:2] = A[0:3, 0:2] * B[0:2, 0:2] + A[0:3, 2:4] * B[2:4, 0:2]
    # C[0:3, 2:4] = A[0:3, 0:2] * B[0:2, 2:4] + A[0:3, 2:4] * B[2:4, 2:4]
    
    # Let's define the 4 blocks of B:
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # And A split into A_left = A[:, 0:2] (3x2) and A_right = A[:, 2:4] (3x2)
    # A_left split into A_L1 = A[0:2, 0:2] (2x2) and A_L2 = A[2:3, 0:2] (1x2)
    # A_right split into A_R1 = A[0:2, 2:4] (2x2) and A_R2 = A[2:3, 2:4] (1x2)
    
    # C[0:3, 0:2] = A_L1*B11 + A_L2*B11 + A_R1*B21 + A_R2*B21
    # C[0:3, 2:4] = A_L1*B12 + A_L2*B12 + A_R1*B22 + A_R2*B22
    
    # We need decompositions for:
    # 1. A_L1 (2x2) * B11 (2x2) -> 7 terms
    # 2. A_L2 (1x2) * B11 (2x2) -> 4 terms
    # 3. A_R1 (2x2) * B21 (2x2) -> 7 terms
    # 4. A_R2 (1x2) * B21 (2x2) -> 4 terms
    # 5. A_L1 (2x2) * B12 (2x2) -> 7 terms
    # 6. A_L2 (1x2) * B12 (2x2) -> 4 terms
    # 7. A_R1 (2x2) * B22 (2x2) -> 7 terms
    # 8. A_R2 (1x2) * B22 (2x2) -> 4 terms
    # Total: 7+4+7+4+7+4+7+4 = 44.
    
    # Strassen 2x2*2x2 (7 terms):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # 1x2 * 2x2 (4 terms, schoolbook):
    # A = [a1, a2], B = [[b11, b12], [b21, b22]]
    # C1 = a1*b11 + a2*b21
    # C2 = a1*b12 + a2*b22
    
    # Let's build each piece.
    
    def add_strassen(a_start_i, b_start_k, c_start_i, c_start_k):
        """
        Add 7 Strassen terms for 2x2*2x2.
        A is at A[a_start_i:a_start_i+2, :] but we only use columns 0:2 relative to the block.
        Actually, A_L1 is A[0:2, 0:2]. In the full A (3x4), these are at i=0,1 and j=0,1.
        B11 is B[0:2, 0:2]. In the full B (4x4), these are at j=0,1 and k=0,1.
        The result C is 2x2, placed at C[c_start_i:c_start_i+2, c_start_k:c_start_k+2].
        """
        a11_i, a11_j = a_start_i, 0  # relative to A block
        a12_i, a12_j = a_start_i, 1
        a21_i, a21_j = a_start_i+1, 0
        a22_i, a22_j = a_start_i+1, 1
        
        b11_j, b11_k = 0, b_start_k  # relative to B block
        b12_j, b12_k = 0, b_start_k+1
        b21_j, b21_k = 1, b_start_k
        b22_j, b22_k = 1, b_start_k+1
        
        # M1 = (A11+A22)*(B11+B22)
        # U: A11 and A22 are 1, V: B11 and B22 are 1, W: contributes to C11 and C22
        # This is getting complex to write by hand. Let's use a generic approach.
        pass
    
    # This manual construction is error-prone. Let's use a different approach:
    # Use the known explicit 44-term decomposition from the literature or construct it via
    # the block method with explicit term generation.
    
    # Actually, let's just use the schoolbook 48 and then try to optimize if time permits.
    # But the prompt says rank-44 is achievable. Let me implement the 44 properly.
    
    # Let's define helper functions to add terms.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Strassen 2x2*2x2 terms (7 terms)
    # We need to map local 2x2 indices to global indices.
    
    def strassen_block(a_i0, b_k0, c_i0, c_k0):
        """
        Add 7 Strassen terms for A[a_i0:a_i0+2, :2] * B[:2, b_k0:b_k0+2] -> C[c_i0:c_i0+2, c_k0:c_k0+2]
        Wait, the B block is 2x2, so it uses 2 rows of B and 2 cols of B.
        """
        # A elements:
        a11 = (a_i0, 0)
        a12 = (a_i0, 1)
        a21 = (a_i0+1, 0)
        a22 = (a_i0+1, 1)
        
        # B elements:
        b11 = (0, b_k0)
        b12 = (0, b_k0+1)
        b21 = (1, b_k0)
        b22 = (1, b_k0+1)
        
        # C elements:
        c11 = (c_i0, c_k0)
        c12 = (c_i0, c_k0+1)
        c21 = (c_i0+1, c_k0)
        c22 = (c_i0+1, c_k0+1)
        
        # M1 = (A11+A22)*(B11+B22)
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[a11[0]*m+a11[1]] = 1; u1[a22[0]*m+a22[1]] = 1
        v1[b11[0]*p+b11[1]] = 1; v1[b22[0]*p+b22[1]] = 1
        w1[c11[1]*n+c11[0]] = 1; w1[c22[1]*n+c22[0]] = 1
        add_term(u1, v1, w1)
        
        # M2 = (A21+A22)*B11
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[a21[0]*m+a21[1]] = 1; u2[a22[0]*m+a22[1]] = 1
        v2[b11[0]*p+b11[1]] = 1
        w2[c21[1]*n+c21[0]] = 1
        add_term(u2, v2, w2)
        
        # M3 = A11*(B12-B22)
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[a11[0]*m+a11[1]] = 1
        v3[b12[0]*p+b12[1]] = 1; v3[b22[0]*p+b22[1]] = -1
        w3[c12[1]*n+c12[0]] = 1
        add_term(u3, v3, w3)
        
        # M4 = A22*(B21-B11)
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[a22[0]*m+a22[1]] = 1
        v4[b21[0]*p+b21[1]] = 1; v4[b11[0]*p+b11[1]] = -1
        w4[c11[1]*n+c11[0]] = 1; w4[c21[1]*n+c21[0]] = 1
        add_term(u4, v4, w4)
        
        # M5 = (A11+A12)*B22
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        u5[a11[0]*m+a11[1]] = 1; u5[a12[0]*m+a12[1]] = 1
        v5[b22[0]*p+b22[1]] = 1
        w5[c11[1]*n+c11[0]] = -1; w5[c12[1]*n+c12[0]] = 1
        add_term(u5, v5, w5)
        
        # M6 = (A21-A11)*(B11+B12)
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        u6[a21[0]*m+a21[1]] = 1; u6[a11[0]*m+a11[1]] = -1
        v6[b11[0]*p+b11[1]] = 1; v6[b12[0]*p+b12[1]] = 1
        w6[c22[1]*n+c22[0]] = 1
        add_term(u6, v6, w6)
        
        # M7 = (A12-A22)*(B21+B22)
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        u7[a12[0]*m+a12[1]] = 1; u7[a22[0]*m+a22[1]] = -1
        v7[b21[0]*p+b21[1]] = 1; v7[b22[0]*p+b22[1]] = 1
        w7[c11[1]*n+c11[0]] = 1; w7[c22[1]*n+c22[0]] = -1
        add_term(u7, v7, w7)
    
    def linear_block(a_i, b_k0, c_i0, c_k0):
        """
        Add 4 schoolbook terms for A[a_i, :2] * B[:2, b_k0:b_k0+2] -> C[c_i0:c_i0+1, c_k0:c_k0+2]
        """
        # A element: a_i, 0 and a_i, 1
        # B elements: (0, b_k0), (0, b_k0+1), (1, b_k0), (1, b_k0+1)
        # C elements: (c_i0, c_k0) and (c_i0, c_k0+1)
        
        # C1 = A[i,0]*B[0,k0] + A[i,1]*B[1,k0]
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[a_i*m+0] = 1
        v1[0*p+b_k0] = 1
        w1[c_k0*n+c_i0] = 1
        add_term(u1, v1, w1)
        
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[a_i*m+1] = 1
        v2[1*p+b_k0] = 1
        w2[c_k0*n+c_i0] = 1
        add_term(u2, v2, w2)
        
        # C2 = A[i,0]*B[0,k0+1] + A[i,1]*B[1,k0+1]
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[a_i*m+0] = 1
        v3[0*p+b_k0+1] = 1
        w3[(c_k0+1)*n+c_i0] = 1
        add_term(u3, v3, w3)
        
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[a_i*m+1] = 1
        v4[1*p+b_k0+1] = 1
        w4[(c_k0+1)*n+c_i0] = 1
        add_term(u4, v4, w4)
    
    # Now assemble the 44 terms.
    # C[0:3, 0:2] = A[0:3, 0:2]*B[0:2, 0:2] + A[0:3, 2:4]*B[2:4, 0:2]
    # C[0:3, 2:4] = A[0:3, 0:2]*B[0:2, 2:4] + A[0:3, 2:4]*B[2:4, 2:4]
    
    # A[0:3, 0:2] is 3x2. Split into A[0:2, 0:2] (2x2) and A[2:3, 0:2] (1x2).
    # A[0:3, 2:4] is 3x2. Split into A[0:2, 2:4] (2x2) and A[2:3, 2:4] (1x2).
    
    # B[0:2, 0:2] is 2x2. b_k0 = 0.
    # B[0:2, 2:4] is 2x2. b_k0 = 2.
    # B[2:4, 0:2] is 2x2. b_k0 = 0, but B rows are 2:4. So b_row_offset = 2.
    # B[2:4, 2:4] is 2x2. b_k0 = 2, b_row_offset = 2.
    
    # My helper functions assume B rows start at 0. I need to adjust.
    # Let me modify the helpers to take b_row_offset.
    
    # Actually, let's just rewrite the helpers to be more general.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def strassen_block(a_i0, a_j0, b_r0, b_k0, c_i0, c_k0):
        """
        Strassen 2x2*2x2.
        A is at A[a_i0:a_i0+2, a_j0:a_j0+2]
        B is at B[b_r0:b_r0+2, b_k0:b_k0+2]
        C is at C[c_i0:c_i0+2, c_k0:c_k0+2]
        """
        a11 = (a_i0, a_j0)
        a12 = (a_i0, a_j0+1)
        a21 = (a_i0+1, a_j0)
        a22 = (a_i0+1, a_j0+1)
        
        b11 = (b_r0, b_k0)
        b12 = (b_r0, b_k0+1)
        b21 = (b_r0+1, b_k0)
        b22 = (b_r0+1, b_k0+1)
        
        c11 = (c_i0, c_k0)
        c12 = (c_i0, c_k0+1)
        c21 = (c_i0+1, c_k0)
        c22 = (c_i0+1, c_k0+1)
        
        def make_u(indices):
            u = [0]*(n*m)
            for (i,j) in indices:
                u[i*m+j] = 1
            return u
        
        def make_v(indices):
            v = [0]*(m*p)
            for (r,k) in indices:
                v[r*p+k] = 1
            return v
        
        def make_w(indices, signs=None):
            w = [0]*(p*n)
            for idx, (i,k) in enumerate(indices):
                sign = 1 if signs is None else signs[idx]
                w[k*n+i] = sign
            return w
        
        # M1 = (A11+A22)*(B11+B22)
        u1 = make_u([a11, a22])
        v1 = make_v([b11, b22])
        w1 = make_w([c11, c22])
        add_term(u1, v1, w1)
        
        # M2 = (A21+A22)*B11
        u2 = make_u([a21, a22])
        v2 = make_v([b11])
        w2 = make_w([c21])
        add_term(u2, v2, w2)
        
        # M3 = A11*(B12-B22)
        u3 = make_u([a11])
        v3 = make_v([b12, b22], )
        # Need signs for v
        v3 = [0]*(m*p)
        v3[b12[0]*p+b12[1]] = 1
        v3[b22[0]*p+b22[1]] = -1
        w3 = make_w([c12])
        add_term(u3, v3, w3)
        
        # M4 = A22*(B21-B11)
        u4 = make_u([a22])
        v4 = [0]*(m*p)
        v4[b21[0]*p+b21[1]] = 1
        v4[b11[0]*p+b11[1]] = -1
        w4 = make_w([c11, c21])
        add_term(u4, v4, w4)
        
        # M5 = (A11+A12)*B22
        u5 = make_u([a11, a12])
        v5 = make_v([b22])
        w5 = [0]*(p*n)
        w5[c11[1]*n+c11[0]] = -1
        w5[c12[1]*n+c12[0]] = 1
        add_term(u5, v5, w5)
        
        # M6 = (A21-A11)*(B11+B12)
        u6 = [0]*(n*m)
        u6[a21[0]*m+a21[1]] = 1
        u6[a11[0]*m+a11[1]] = -1
        v6 = make_v([b11, b12])
        w6 = make_w([c22])
        add_term(u6, v6, w6)
        
        # M7 = (A12-A22)*(B21+B22)
        u7 = [0]*(n*m)
        u7[a12[0]*m+a12[1]] = 1
        u7[a22[0]*m+a22[1]] = -1
        v7 = make_v([b21, b22])
        w7 = [0]*(p*n)
        w7[c11[1]*n+c11[0]] = 1
        w7[c22[1]*n+c22[0]] = -1
        add_term(u7, v7, w7)
    
    def linear_block(a_i, a_j0, b_r0, b_k0, c_i0, c_k0):
        """
        Schoolbook 1x2 * 2x2.
        A is at A[a_i, a_j0:a_j0+2]
        B is at B[b_r0:b_r0+2, b_k0:b_k0+2]
        C is at C[c_i0, c_k0:c_k0+2]
        """
        # C1 = A[a_i, a_j0]*B[b_r0, b_k0] + A[a_i, a_j0+1]*B[b_r0+1, b_k0]
        u1 = [0]*(n*m)
        u1[a_i*m+a_j0] = 1
        v1 = [0]*(m*p)
        v1[b_r0*p+b_k0] = 1
        w1 = [0]*(p*n)
        w1[c_k0*n+c_i0] = 1
        add_term(u1, v1, w1)
        
        u2 = [0]*(n*m)
        u2[a_i*m+a_j0+1] = 1
        v2 = [0]*(m*p)
        v2[(b_r0+1)*p+b_k0] = 1
        w2 = [0]*(p*n)
        w2[c_k0*n+c_i0] = 1
        add_term(u2, v2, w2)
        
        # C2 = A[a_i, a_j0]*B[b_r0, b_k0+1] + A[a_i, a_j0+1]*B[b_r0+1, b_k0+1]
        u3 = [0]*(n*m)
        u3[a_i*m+a_j0] = 1
        v3 = [0]*(m*p)
        v3[b_r0*p+b_k0+1] = 1
        w3 = [0]*(p*n)
        w3[(c_k0+1)*n+c_i0] = 1
        add_term(u3, v3, w3)
        
        u4 = [0]*(n*m)
        u4[a_i*m+a_j0+1] = 1
        v4 = [0]*(m*p)
        v4[(b_r0+1)*p+b_k0+1] = 1
        w4 = [0]*(p*n)
        w4[(c_k0+1)*n+c_i0] = 1
        add_term(u4, v4, w4)
    
    # Assemble:
    # C[0:3, 0:2] = A[0:3, 0:2]*B[0:2, 0:2] + A[0:3, 2:4]*B[2:4, 0:2]
    # A[0:3, 0:2]: A[0:2, 0:2] is Strassen (a_i0=0, a_j0=0), A[2:3, 0:2] is Linear (a_i=2, a_j0=0)
    # B[0:2, 0:2]: b_r0=0, b_k0=0
    # C[0:3, 0:2]: c_i0=0, c_k0=0
    
    # Term 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    strassen_block(0, 0, 0, 0, 0, 0)
    # Term 2: A[2:3, 0:2] * B[0:2, 0:2] -> C[2:3, 0:2]
    linear_block(2, 0, 0, 0, 2, 0)
    # Term 3: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    strassen_block(0, 2, 2, 0, 0, 0)
    # Term 4: A[2:3, 2:4] * B[2:4, 0:2] -> C[2:3, 0:2]
    linear_block(2, 2, 2, 0, 2, 0)
    
    # C[0:3, 2:4] = A[0:3, 0:2]*B[0:2, 2:4] + A[0:3, 2:4]*B[2:4, 2:4]
    # Term 5: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    strassen_block(0, 0, 0, 2, 0, 2)
    # Term 6: A[2:3, 0:2] * B[0:2, 2:4] -> C[2:3, 2:4]
    linear_block(2, 0, 0, 2, 2, 2)
    # Term 7: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    strassen_block(0, 2, 2, 2, 0, 2)
    # Term 8: A[2:3, 2:4] * B[2:4, 2:4] -> C[2:3, 2:4]
    linear_block(2, 2, 2, 2, 2, 2)
    
    # Now U_cols, V_cols, W_cols should have 44 terms.
    rank = len(U_cols)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for c in range(rank):
        U[:, c] = U_cols[c]
        V[:, c] = V_cols[c]
        W[:, c] = W_cols[c]
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_schoolbook_48():
    n, m, p = 3, 4, 4
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
    return U, V, W

def _als_attempt(n, m, p, rank, max_iter=50, tol=1e-6, seed=0, time_budget=5.0):
    """
    Try to find a rank-r decomposition via ALS.
    Returns (U, V, W) if successful, else None.
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, rank) * 0.1
    V = rng.randn(m*p, rank) * 0.1
    W = rng.randn(p*n, rank) * 0.1
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        # Fix U, V, solve for W
        # For each (i,j,k), T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Let A_c = U[i*m+j,c] * V[j*p+k,c]
        # Then sum_c A_c * W[k*n+i,c] = T[i,j,k] = 1
        # This is a linear system for W[:, k*n+i]
        
        W_new = W.copy()
        for i in range(n):
            for k in range(p):
                w_idx = k * n + i
                # Build system: sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[w_idx, c] = 1 for all j
                # Matrix shape: m x rank
                A = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # Solve A * W[w_idx, :] = 1
                try:
                    W_new[w_idx, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Fix U, W, solve for V
        V_new = V.copy()
        for j in range(m):
            for k in range(p):
                v_idx = j * p + k
                A = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                try:
                    V_new[v_idx, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Fix V, W, solve for U
        U_new = U.copy()
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                A = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                try:
                    U_new[u_idx, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        U, V, W = U_new, V_new, W_new
    
    # Check residual
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                T[i, j, k] = val
    if np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0):
        return U.tolist(), V.tolist(), W.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the rank-44 decomposition
    U44, V44, W44 = _build_block_44()
    
    # Verify
    if _verify_decomp(U44, V44, W44, n, m, p):
        best = (44, U44, V44, W44)
    else:
        # Fallback to schoolbook
        U48, V48, W48 = _build_schoolbook_48()
        best = (48, U48, V48, W48)
    
    # Try to improve with ALS
    start_time = time.time()
    deadline = 120  # seconds
    
    # Try ranks from 43 down to 30
    for r in range(43, 29, -1):
        if time.time() - start_time > deadline - 5:
            break
        
        # Try a few random restarts
        for seed in range(5):
            if time.time() - start_time > deadline - 5:
                break
            
            result = _als_attempt(n, m, p, r, max_iter=30, tol=1e-6, seed=seed, time_budget=3.0)
            if result is not None:
                U, V, W = result
                if _verify_decomp(U, V, W, n, m, p):
                    best = (r, U, V, W)
                    break  # Found a valid rank-r decomposition, move to next r
    
    return {"U": best[1], "V": best[2], "W": best[3]}
