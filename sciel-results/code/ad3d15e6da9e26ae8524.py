"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Use Strassen's rank-7 decomposition for 2x2*2x2 blocks as building blocks.
    Partition the 3x4*4x8 multiplication into:
    - 8 blocks of 2x2*2x2 (using Strassen, rank 7 each) = 56
    - 2 blocks of 1x4*4x4 (using schoolbook, rank 4 each) = 8
    Total rank: 64
    
    The 3x4 matrix A has rows 0,1,2 and columns 0,1,2,3.
    The 4x8 matrix B has rows 0,1,2,3 and columns 0,1,2,3,4,5,6,7.
    The output C is 3x8.
    
    We partition:
    - A rows: {0,1} and {2}
    - A columns / B rows: {0,1} and {2,3}
    - B columns: {0,1}, {2,3}, {4,5}, {6,7}
    
    Blocks:
    1. C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2]  (2x2*2x2)
    2. C[0:2, 0:2] += A[0:2, 2:4] * B[2:4, 0:2]  (2x2*2x2)
    3. C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4]  (2x2*2x2)
    4. C[0:2, 2:4] += A[0:2, 2:4] * B[2:4, 2:4]  (2x2*2x2)
    5. C[0:2, 4:6] = A[0:2, 0:2] * B[0:2, 4:6]  (2x2*2x2)
    6. C[0:2, 4:6] += A[0:2, 2:4] * B[2:4, 4:6]  (2x2*2x2)
    7. C[0:2, 6:8] = A[0:2, 0:2] * B[0:2, 6:8]  (2x2*2x2)
    8. C[0:2, 6:8] += A[0:2, 2:4] * B[2:4, 6:8]  (2x2*2x2)
    9. C[2:3, 0:4] = A[2:3, :] * B[:, 0:4]  (1x4*4x4)
    10. C[2:3, 4:8] = A[2:3, :] * B[:, 4:8]  (1x4*4x4)
    """
    n, m, p = DIMENSIONS
    
    terms = []  # list of (u, v, w) tuples where u is in R^n, v in R^m, w in R^p
    
    def add_strassen_block(i0, j0, k0):
        """
        Add a 2x2*2x2 Strassen decomposition block.
        A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
        
        Strassen's algorithm for 2x2 matrices:
        A = [[a11, a12], [a21, a22]]
        B = [[b11, b12], [b21, b22]]
        
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)b11
        M3 = a11(b12-b22)
        M4 = a22(b21-b11)
        M5 = (a11+a12)b22
        M6 = (a21-a11)(b11+b12)
        M7 = (a12-a22)(b21+b22)
        
        C = [[M1+M4-M5+M7, M3+M5],
             [M2+M4, M1-M2+M3+M6]]
        """
        # M1 = (a11+a22)(b11+b22)
        u = [0.0]*n; v = [0.0]*m; w = [0.0]*p
        u[i0] = 1; u[i0+1] = 1
        v[j0] = 1; v[j0+2] = 1
        w[k0] = 1; w[k0+1] = 1
        terms.append((u[:], v[:], w[:]))
        
        # M2 = (a21+a22)b11
        u = [0.0]*n; v = [0.0]*m; w = [0.0]*p
        u[i0+1] = 1; u[i0+1] = 1  # a21 and a22 both have coefficient 1
        # Wait, a21 is at position (i0+1, j0), a22 is at (i0+1, j0+1)
        # So u[i0+1] should be 1 for both? No, u is in R^n (rows of A)
        # a21: row i0+1, col j0 -> u[i0+1]=1, v[j0]=1
        # a22: row i0+1, col j0+1 -> u[i0+1]=1, v[j0+1]=1
        # So for M2: (a21+a22)*b11, we need u[i0+1]=1, v[j0]=1 AND v[j0+1]=1? No.
        # Actually M2 = (a21+a22) * b11
        # This is a single term: coefficient of a21 is 1, coefficient of a22 is 1, coefficient of b11 is 1
        # So u[i0+1] = 1, v[j0] = 1, v[j0+1] = 1? No, that's not right.
        # In tensor decomposition, each term is u ⊗ v ⊗ w where u, v, w are vectors.
        # For M2: the contribution is 1*a21*1*b11 + 1*a22*1*b11
        # This can be written as: (e_{i0+1}) ⊗ (e_{j0} + e_{j0+1}) ⊗ (e_{k0})
        # So u[i0+1] = 1, v[j0] = 1, v[j0+1] = 1, w[k0] = 1
        u = [0.0]*n; v = [0.0]*m; w = [0.0]*p
        u[i0+1] = 1
        v[j0] = 1
        v[j0+1] = 1
        w[k0] = 1
        terms.append((u[:], v[:], w[:]))
        
        # M3 = a11(b12-b22)
        u = [0.0]*n; v = [0.0]*m; w = [0.0]*p
        u[i0] = 1
        v[j0+1] = 1
        v[j0+2] = -1
        w[k0+1] = 1
        terms.append((u[:], v[:], w[:]))
        
        # M4 = a22(b21-b11)
        u = [0.0]*n; v = [0.0]*m; w = [0.0]*p
        u[i0+1] = 1
        v[j0] = -1
        v[j0+2] = 1
        w[k0] = 1
        terms.append((u[:], v[:], w[:]))
        
        # M5 = (a11+a12)b22
        u = [0.0]*n; v = [0.0]*m; w = [0.0]*p
        u[i0] = 1
        v[j0+1] = 1
        v[j0+2] = 1
        w[k0+1] = 1
        terms.append((u[:], v[:], w[:]))
        
        # M6 = (a21-a11)(b11+b12)
        u = [0.0]*n; v = [0.0]*m; w = [0.0]*p
        u[i0] = -1
        u[i0+1] = 1
        v[j0] = 1
        v[j0+1] = 1
        w[k0] = 1
        terms.append((u[:], v[:], w[:]))
        
        # M7 = (a12-a22)(b21+b22)
        u = [0.0]*n; v = [0.0]*m; w = [0.0]*p
        u[i0] = 1
        u[i0+1] = -1
        v[j0+2] = 1
        v[j0+3] = 1
        w[k0+1] = 1
        terms.append((u[:], v[:], w[:]))
    
    def add_1x4x4_block(i0, j0, k0):
        """
        Add a 1x4*4x4 schoolbook block.
        A[i0:i0+1, j0:j0+4] * B[j0:j0+4, k0:k0+4] -> C[i0:i0+1, k0:k0+4]
        """
        for j in range(4):
            for k in range(4):
                u = [0.0]*n; v = [0.0]*m; w = [0.0]*p
                u[i0] = 1
                v[j0+j] = 1
                w[k0+k] = 1
                terms.append((u[:], v[:], w[:]))
    
    # Add all Strassen blocks
    # C[0:2, 0:2]
    add_strassen_block(i0=0, j0=0, k0=0)
    add_strassen_block(i0=0, j0=2, k0=0)
    
    # C[0:2, 2:4]
    add_strassen_block(i0=0, j0=0, k0=2)
    add_strassen_block(i0=0, j0=2, k0=2)
    
    # C[0:2, 4:6]
    add_strassen_block(i0=0, j0=0, k0=4)
    add_strassen_block(i0=0, j0=2, k0=4)
    
    # C[0:2, 6:8]
    add_strassen_block(i0=0, j0=0, k0=6)
    add_strassen_block(i0=0, j0=2, k0=6)
    
    # Add 1x4x4 blocks for row 2 of A
    add_1x4x4_block(i0=2, j0=0, k0=0)
    add_1x4x4_block(i0=2, j0=0, k0=4)
    
    rank = len(terms)
    
    # Build U, V, W matrices
    # U has shape (n*m, rank), indexed by [i*m + j, r]
    # V has shape (m*p, rank), indexed by [j*p + k, r]
    # W has shape (p*n, rank), indexed by [k*n + i, r]
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(n):
            for j in range(m):
                # u[i] is the coefficient for A[i,j] in this term
                # But wait, u is in R^n, not R^{n*m}
                # I think I made an error. Let me reconsider.
                pass
    
    # Let me reconsider the structure. In the tensor decomposition:
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, that's not standard. Let me think again.
    
    # The matrix multiplication tensor is T[i,j,k] = delta_{j,j} for the product C[i,k] = sum_j A[i,j]*B[j,k]
    # A rank-r decomposition is: C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_k V[j,k,r]*B[j,k]) * W[k,i,r]
    # No, that's not right either.
    
    # Standard tensor decomposition for matmul:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Rank-r decomposition: C[i,k] = sum_r M_r[i] * N_r[k] * P_r[j]... 
    # Actually, the standard form is:
    # C[i,k] = sum_r (sum_j U_r[i,j] * A[i,j]) * (sum_j V_r[j,k] * B[j,k]) * W_r[i,k]
    # No, that's not standard.
    
    # Let me look at the interface again:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The convention is W[k*n+i, r]
    
    # I think the decomposition is:
    # C[i,k] = sum_r (sum_{j} U[i*m+j, r] * A[i,j]) * (sum_{j} V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # No, that doesn't make sense dimensionally.
    
    # Actually, I think it's:
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
    # But U and V have fixed indices...
    
    # Let me think about this differently. The tensor T has indices (i, j, k) where:
    # i is row of A (0..n-1)
    # j is column of A / row of B (0..m-1)  
    # k is column of B (0..p-1)
    
    # T[i,j,k] = 1 if this is a valid matmul term, 0 otherwise
    # Actually T[i,j,k] = delta for the matmul tensor, meaning T[i,j,k] = 1 for all valid (i,j,k)
    # and the matmul is C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
    
    # A rank-r decomposition is: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r is in R^{m*p}, w_r is in R^{p*n}
    # and T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    
    # So for each term r, we have vectors u_r, v_r, w_r such that:
    # sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all i,j,k (for matmul)
    
    # This means: for each (i,j,k), the sum over r of the product equals 1.
    
    # For the schoolbook: each term has u_r with a single 1 at position i*m+j,
    # v_r with a single 1 at position j*p+k, w_r with a single 1 at position k*n+i.
    
    # For Strassen, we have linear combinations.
    
    # Let me redo this properly.
    
    terms = []  # list of (u, v, w) where u in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    
    def add_strassen_block_v2(i0, j0, k0):
        """
        Add a 2x2*2x2 Strassen decomposition block.
        A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]
        
        For each Strassen product M_s, we add a term where:
        - u has 1s at the A positions involved
        - v has 1s at the B positions involved  
        - w has 1s at the C positions that receive this contribution
        
        But actually, in the tensor decomposition, each term contributes to ALL (i,j,k) triples.
        The key insight is that the Strassen algorithm computes intermediate products
        that are linear combinations of A and B entries, and then combines them.
        
        Let me think about this more carefully. The Strassen algorithm for 2x2:
        
        M1 = (a11+a22)(b11+b22)
        M2 = (a21+a22)b11
        M3 = a11(b12-b22)
        M4 = a22(b21-b11)
        M5 = (a11+a12)b22
        M6 = (a21-a11)(b11+b12)
        M7 = (a12-a22)(b21+b22)
        
        C11 = M1+M4-M5+M7
        C12 = M3+M5
        C21 = M2+M4
        C22 = M1-M2+M3+M6
        
        In the tensor decomposition, each M_s corresponds to a rank-1 term.
        For M1 = (a11+a22)(b11+b22):
        - This contributes to C11 with coefficient +1 and to C22 with coefficient +1
        - The A part: a11 + a22, so u has 1 at (i0,j0) and 1 at (i0+1,j0+1)
        - The B part: b11 + b22, so v has 1 at (j0,k0) and 1 at (j0+1,k0+1)
        - The C part: contributes to C(i0,k0) with +1 and C(i0+1,k0+1) with +1
          so w has 1 at (k0, i0) and 1 at (k0+1, i0+1)
        
        Wait, but M1 contributes to BOTH C11 and C22. So w should have entries for both.
        
        Let me define the terms properly:
        """
        # M1 = (a11+a22)(b11+b22)
        # Contributes to C11 (+1) and C22 (+1)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[i0*m + j0] = 1       # a11
        u[(i0+1)*m + (j0+1)] = 1  # a22
        v[j0*p + k0] = 1       # b11
        v[(j0+1)*p + (k0+1)] = 1  # b22
        w[k0*n + i0] = 1       # C11
        w[(k0+1)*n + (i0+1)] = 1  # C22
        terms.append((u[:], v[:], w[:]))
        
        # M2 = (a21+a22)b11
        # Contributes to C21 (+1) and C22 (-1)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[(i0+1)*m + j0] = 1       # a21
        u[(i0+1)*m + (j0+1)] = 1  # a22
        v[j0*p + k0] = 1           # b11
        w[k0*n + (i0+1)] = 1       # C21
        w[(k0+1)*n + (i0+1)] = -1  # C22
        terms.append((u[:], v[:], w[:]))
        
        # M3 = a11(b12-b22)
        # Contributes to C12 (+1) and C22 (+1)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[i0*m + j0] = 1           # a11
        v[j0*p + (k0+1)] = 1       # b12
        v[(j0+1)*p + (k0+1)] = -1  # b22
        w[(k0+1)*n + i0] = 1       # C12
        w[(k0+1)*n + (i0+1)] = 1   # C22
        terms.append((u[:], v[:], w[:]))
        
        # M4 = a22(b21-b11)
        # Contributes to C11 (+1) and C21 (+1)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[(i0+1)*m + (j0+1)] = 1  # a22
        v[(j0+1)*p + k0] = 1       # b21
        v[j0*p + k0] = -1          # b11
        w[k0*n + i0] = 1           # C11
        w[k0*n + (i0+1)] = 1       # C21
        terms.append((u[:], v[:], w[:]))
        
        # M5 = (a11+a12)b22
        # Contributes to C11 (-1) and C12 (+1)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[i0*m + j0] = 1           # a11
        u[i0*m + (j0+1)] = 1       # a12
        v[(j0+1)*p + (k0+1)] = 1   # b22
        w[k0*n + i0] = -1          # C11
        w[(k0+1)*n + i0] = 1       # C12
        terms.append((u[:], v[:], w[:]))
        
        # M6 = (a21-a11)(b11+b12)
        # Contributes to C22 (+1)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[i0*m + j0] = -1          # a11
        u[(i0+1)*m + j0] = 1       # a21
        v[j0*p + k0] = 1           # b11
        v[j0*p + (k0+1)] = 1       # b12
        w[(k0+1)*n + (i0+1)] = 1   # C22
        terms.append((u[:], v[:], w[:]))
        
        # M7 = (a12-a22)(b21+b22)
        # Contributes to C11 (+1)
        u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
        u[i0*m + (j0+1)] = 1       # a12
        u[(i0+1)*m + (j0+1)] = -1  # a22
        v[(j0+1)*p + k0] = 1       # b21
        v[(j0+1)*p + (k0+1)] = 1   # b22
        w[k0*n + i0] = 1           # C11
        terms.append((u[:], v[:], w[:]))
    
    def add_1x4x4_block_v2(i0, j0, k0):
        """
        Add a 1x4*4x4 schoolbook block.
        A[i0, j0:j0+4] * B[j0:j0+4, k0:k0+4] -> C[i0, k0:k0+4]
        """
        for j in range(4):
            for k in range(4):
                u = [0.0]*(n*m); v = [0.0]*(m*p); w = [0.0]*(p*n)
                u[i0*m + (j0+j)] = 1
                v[(j0+j)*p + (k0+k)] = 1
                w[(k0+k)*n + i0] = 1
                terms.append((u[:], v[:], w[:]))
    
    # Clear terms and rebuild
    terms = []
    
    # Add all Strassen blocks
    # C[0:2, 0:2]
    add_strassen_block_v2(i0=0, j0=0, k0=0)
    add_strassen_block_v2(i0=0, j0=2, k0=0)
    
    # C[0:2, 2:4]
    add_strassen_block_v2(i0=0, j0=0, k0=2)
    add_strassen_block_v2(i0=0, j0=2, k0=2)
    
    # C[0:2, 4:6]
    add_strassen_block_v2(i0=0, j0=0, k0=4)
    add_strassen_block_v2(i0=0, j0=2, k0=4)
    
    # C[0:2, 6:8]
    add_strassen_block_v2(i0=0, j0=0, k0=6)
    add_strassen_block_v2(i0=0, j0=2, k0=6)
    
    # Add 1x4x4 blocks for row 2 of A
    add_1x4x4_block_v2(i0=2, j0=0, k0=0)
    add_1x4x4_block_v2(i0=2, j0=0, k0=4)
    
    rank = len(terms)
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx in range(n * m):
            if u[idx] != 0:
                U[idx][r] = u[idx]
        for idx in range(m * p):
            if v[idx] != 0:
                V[idx][r] = v[idx]
        for idx in range(p * n):
            if w[idx] != 0:
                W[idx][r] = w[idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
