"""Matrix multiplication 6x6x6 - Recursive Strassen decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _strassen_3x3_decomposition():
    """Return U, V, W for 3x3x3 matrix multiplication using Strassen (rank 7).
    
    The 7 Strassen products for multiplying A (3x3) by B (3x3) to get C (3x3):
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11) B00
    M3 = A00 (B10 - B11)
    M4 = A11 (B01 - B00)
    M5 = (A00 + A01) B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    For the full tensor, we need to express each element of C as a sum over the 7 products.
    Each product Mj is a scalar computed from specific elements of A and B.
    The tensor decomposition has rank 7 (one column per Strassen product).
    
    U[i*m + j, c]: coefficient for A element (i,j) in product c
    V[j*p + k, c]: coefficient for B element (j,k) in product c
    W[k*n + i, c]: coefficient for C element (k,i) in product c (trace convention: W[k*n+i, c])
    
    Here n=3, m=3, p=3.
    A has elements A[i][j] for i in 0..2, j in 0..2
    B has elements B[j][k] for j in 0..2, k in 0..2
    C has elements C[i][k] for i in 0..2, k in 0..2
    
    W trace convention: W[k*n + i, c] means W index is k*n+i where k is the column of C (from B) and i is the row of C (from A).
    So W[k*3 + i, c] is the coefficient for C[i][k] in product c.
    """
    n, m, p = 3, 3, 3
    rank = 7
    U = [[0] * rank for _ in range(n * m)]  # 9 x 7
    V = [[0] * rank for _ in range(m * p)]  # 9 x 7
    W = [[0] * rank for _ in range(p * n)]  # 9 x 7
    
    # A element index: i*m + j = i*3 + j
    # B element index: j*p + k = j*3 + k
    # W element index: k*n + i = k*3 + i
    
    # M1 = (A00 + A11)(B00 + B11)
    # A coefficients: A00=1, A11=1
    # B coefficients: B00=1, B11=1
    c = 0
    U[0*3+0][c] = 1  # A00
    U[1*3+1][c] = 1  # A11
    V[0*3+0][c] = 1  # B00
    V[1*3+1][c] = 1  # B11
    # C contributions: C00 += M1, C11 += M1
    W[0*3+0][c] = 1  # C00 (k=0, i=0)
    W[1*3+1][c] = 1  # C11 (k=1, i=1)
    
    # M2 = (A10 + A11) B00
    c = 1
    U[1*3+0][c] = 1  # A10
    U[1*3+1][c] = 1  # A11
    V[0*3+0][c] = 1  # B00
    # C contributions: C10 += M2, C11 -= M2
    W[0*3+1][c] = 1  # C10 (k=0, i=1)
    W[1*3+1][c] = -1 # C11 (k=1, i=1)
    
    # M3 = A00 (B10 - B11)
    c = 2
    U[0*3+0][c] = 1  # A00
    V[1*3+0][c] = 1  # B10
    V[1*3+1][c] = -1 # B11
    # C contributions: C01 += M3, C11 += M3
    W[0*3+0][c] = 1  # C01 (k=1, i=0)
    W[1*3+1][c] = 1  # C11 (k=1, i=1)
    
    # M4 = A11 (B01 - B00)
    c = 3
    U[1*3+1][c] = 1  # A11
    V[0*3+1][c] = 1  # B01
    V[0*3+0][c] = -1 # B00
    # C contributions: C00 += M4, C10 += M4
    W[0*3+0][c] = 1  # C00 (k=0, i=0)
    W[0*3+1][c] = 1  # C10 (k=0, i=1)
    
    # M5 = (A00 + A01) B11
    c = 4
    U[0*3+0][c] = 1  # A00
    U[0*3+1][c] = 1  # A01
    V[1*3+1][c] = 1  # B11
    # C contributions: C00 -= M5, C01 += M5
    W[0*3+0][c] = -1 # C00 (k=0, i=0)
    W[1*3+0][c] = 1  # C01 (k=1, i=0)
    
    # M6 = (A10 - A00)(B00 + B01)
    c = 5
    U[1*3+0][c] = 1  # A10
    U[0*3+0][c] = -1 # A00
    V[0*3+0][c] = 1  # B00
    V[0*3+1][c] = 1  # B01
    # C contributions: C11 += M6
    W[1*3+1][c] = 1  # C11 (k=1, i=1)
    
    # M7 = (A01 - A11)(B10 + B11)
    c = 6
    U[0*3+1][c] = 1  # A01
    U[1*3+1][c] = -1 # A11
    V[1*3+0][c] = 1  # B10
    V[1*3+1][c] = 1  # B11
    # C contributions: C00 += M7
    W[0*3+0][c] = 1  # C00 (k=0, i=0)
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    import numpy as np
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, k) in C, check that sum over j of A[i,j]*B[j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Actually, the tensor condition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_k' V[j*p+k', c] * B[j,k']) ... 
    # Wait, let me think more carefully.
    #
    # The decomposition says: for all A, B, C = A*B iff for each (i,k):
    # C[i,k] = sum_j A[i,j]*B[j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] summed appropriately
    #
    # More precisely, the tensor T[i,j,k] = delta (the multiplication tensor) satisfies:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # Let's verify this.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 1.0  # T[i,j,k] = 1 for matrix multiplication
                rhs = 0.0
                for c in range(rank):
                    rhs += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(lhs - rhs) > 1e-10:
                    return False
    return True


def _combine_decompositions(U1, V1, W1, U2, V2, W2, n, m, p):
    """Combine two decompositions to multiply matrices of size n*m x m*p -> n*p.
    
    This is for the recursive step: we split 6x6 into blocks of 3x3.
    The top-level 6x6x6 uses Strassen at the block level (3x3 blocks, each 2x2... no wait).
    
    Actually, for 6x6 = 3x3 blocks of 2x2, we'd use Strassen with 7 products, each being a 2x2x2 multiplication.
    But 2x2x2 has rank 7 (Strassen) or 8 (schoolbook).
    
    Alternative: 6x6 = 2x2 blocks of 3x3. Use Strassen at block level: 7 products of 3x3x3.
    Each 3x3x3 uses Strassen (rank 7). Total rank = 7 * 7 = 49? No that's too optimistic.
    
    Wait, let me reconsider. The known result: recursive Strassen with 7 level-1 terms each using schoolbook 3x3 gives rank 189.
    That means: 6x6x6 = 7 (Strassen at 3x3 block level) * 27 (schoolbook 3x3x3) = 189.
    
    So the structure is:
    - Split 6x6 matrices into 2x2 blocks of 3x3 matrices.
    - Apply Strassen's algorithm to compute the 2x2 block matrix product using 7 products.
    - Each product is a 3x3 matrix multiplication, done by schoolbook (rank 27).
    - Total rank = 7 * 27 = 189.
    """
    pass


def construct():
    """Construct rank-minimal decomposition for 6x6x6 matrix multiplication.
    
    Uses recursive Strassen: split 6x6 into 2x2 blocks of 3x3.
    Strassen gives 7 products at the block level.
    Each block product is a 3x3x3 matrix multiplication done by schoolbook (rank 27).
    Total rank = 7 * 27 = 189.
    
    The 6x6 matrices are viewed as 2x2 block matrices where each block is 3x3.
    A = [[A00, A01], [A10, A11]] where each Aij is 3x3.
    B = [[B00, B01], [B10, B11]] where each Bij is 3x3.
    
    Strassen products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each Mj is a 3x3 matrix product. To decompose this at the scalar level,
    we use schoolbook for each 3x3 product: 27 scalar products per Mj.
    
    For the full 6x6 decomposition:
    - U has 6*6 = 36 rows, 189 columns
    - V has 6*6 = 36 rows, 189 columns  
    - W has 6*6 = 36 rows, 189 columns
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Block structure: 2x2 blocks of 3x3
    # Block (bi, bj) of A corresponds to rows 3*bi..3*bi+2, cols 3*bj..3*bj+2
    # i.e., A element at global (i, j) where i = 3*bi + li, j = 3*bj + lj
    
    rank = 7 * 27  # 189
    
    U = [[0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen coefficients for the 2x2 block-level:
    # For each Strassen product j (0..6):
    #   U_block[bi, bj, j] = coefficient for block A[bi,bj] in product j
    #   V_block[bj, bk, j] = coefficient for block B[bj,bk] in product j
    #   W_block[bk, bi, j] = coefficient for block C[bi,bk] in product j
    
    # M1 = (A00+A11)(B00+B11): U: A00=1,A11=1; V: B00=1,B11=1; W: C00=1,C11=1
    # M2 = (A10+A11)B00: U: A10=1,A11=1; V: B00=1; W: C10=1,C11=-1
    # M3 = A00(B10-B11): U: A00=1; V: B10=1,B11=-1; W: C01=1,C11=1
    # M4 = A11(B01-B00): U: A11=1; V: B01=1,B00=-1; W: C00=1,C10=1
    # M5 = (A00+A01)B11: U: A00=1,A01=1; V: B11=1; W: C00=-1,C01=1
    # M6 = (A10-A00)(B00+B01): U: A10=1,A00=-1; V: B00=1,B01=1; W: C11=1
    # M7 = (A01-A11)(B10+B11): U: A01=1,A11=-1; V: B10=1,B11=1; W: C00=1
    
    # Strassen product definitions: (product_idx, [(block_row, block_col, coeff) for U], [(block_row, block_col, coeff) for V], [(block_row, block_col, coeff) for W])
    # U block: (bi, bj) means A block at row bi, col bj
    # V block: (bj, bk) means B block at row bj, col bk
    # W block: (bk, bi) means C block at row bi, col bk (but stored as W[k*n+i] so block index is bk*2+bi... wait)
    
    # Actually W is indexed by [k*n + i] where k is from B (column of C) and i is from A (row of C).
    # In block terms: W block (bk, bi) where bk is column block of C, bi is row block of C.
    # Global W index for element in block (bk, bi) at local position (lk, li): (3*bk + lk)*6 + (3*bi + li)
    
    strassen_products = [
        # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10+A11)B00 -> C10 += 1, C11 -= 1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        # M3: A00(B10-B11) -> C01 += 1, C11 += 1
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        # M4: A11(B01-B00) -> C00 += 1, C10 += 1
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M5: (A00+A01)B11 -> C00 -= 1, C01 += 1
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
        # M6: (A10-A00)(B00+B01) -> C11 += 1
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7: (A01-A11)(B10+B11) -> C00 += 1
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]
    
    # For each Strassen product, we need to decompose the 3x3 block multiplication.
    # Schoolbook 3x3x3: for local (li, lj, lk), U_local[li*3+lj] = 1, V_local[lj*3+lk] = 1, W_local[lk*3+li] = 1
    # There are 27 such terms (li, lj, lk) each contributing one rank-1 term.
    
    for prod_idx in range(7):
        u_blocks, v_blocks, w_blocks = strassen_products[prod_idx]
        
        for li in range(3):  # local row of A block / row of C
            for lj in range(3):  # local col of A block / row of B
                for lk in range(3):  # local col of B block / col of C
                    col = prod_idx * 27 + li * 9 + lj * 3 + lk
                    
                    # For each U block term: A block (bi, bj) with coefficient cu
                    # The local element is at global position: row = 3*bi + li, col = 3*bj + lj
                    # U index: (3*bi + li) * 6 + (3*bj + lj)
                    for (bi, bj, cu) in u_blocks:
                        u_idx = (3 * bi + li) * m + (3 * bj + lj)
                        U[u_idx][col] += cu
                    
                    # For each V block term: B block (bj, bk) with coefficient cv
                    # The local element is at global position: row = 3*bj + lj, col = 3*bk + lk
                    # V index: (3*bj + lj) * 6 + (3*bk + lk)
                    for (bj, bk, cv) in v_blocks:
                        v_idx = (3 * bj + lj) * p + (3 * bk + lk)
                        V[v_idx][col] += cv
                    
                    # For each W block term: C block (bk, bi) with coefficient cw
                    # The local element is at global position: row = 3*bi + li, col = 3*bk + lk
                    # W index (trace convention): k*n + i where k = 3*bk + lk, i = 3*bi + li
                    for (bk, bi, cw) in w_blocks:
                        w_idx = (3 * bk + lk) * n + (3 * bi + li)
                        W[w_idx][col] += cw
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
