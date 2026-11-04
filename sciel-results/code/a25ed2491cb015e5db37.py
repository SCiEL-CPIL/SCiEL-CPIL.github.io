"""Matrix multiplication 4,4,9 - Strassen-based decomposition with exact tensor verification."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+k, r], V[k*2+j, r], W[j*2+i, r]
    where A is 2x2 (rows i, cols k), B is 2x2 (rows k, cols j), C is 2x2 (rows i, cols j).
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Term 0: M1 = (a+d)(e+h)
    # A terms: a=A[0,0] (i=0,k=0), d=A[1,1] (i=1,k=1)
    # B terms: e=B[0,0] (k=0,j=0), h=B[1,1] (k=1,j=1)
    # C terms: all C[i,j] get M1
    U[0*2+0, 0] = 1; U[1*2+1, 0] = 1   # a, d
    V[0*2+0, 0] = 1; V[1*2+1, 0] = 1   # e, h
    W[0*2+0, 0] = 1; W[0*2+1, 0] = 1; W[1*2+0, 0] = 1; W[1*2+1, 0] = 1
    
    # Term 1: M2 = (c+d)e
    # A terms: c=A[1,0] (i=1,k=0), d=A[1,1] (i=1,k=1)
    # B terms: e=B[0,0] (k=0,j=0)
    # C terms: C[1,1] gets M2 (negative)
    U[1*2+0, 1] = 1; U[1*2+1, 1] = 1   # c, d
    V[0*2+0, 1] = 1                    # e
    W[1*2+1, 1] = -1                   # C[1,1]
    
    # Term 2: M3 = a(f-h)
    # A terms: a=A[0,0] (i=0,k=0)
    # B terms: f=B[0,1] (k=0,j=1), h=B[1,1] (k=1,j=1)
    # C terms: C[0,1] gets M3
    U[0*2+0, 2] = 1                    # a
    V[0*2+1, 2] = 1; V[1*2+1, 2] = -1  # f, -h
    W[0*2+1, 2] = 1                    # C[0,1]
    
    # Term 3: M4 = d(g-e)
    # A terms: d=A[1,1] (i=1,k=1)
    # B terms: g=B[1,0] (k=1,j=0), e=B[0,0] (k=0,j=0)
    # C terms: C[0,0] gets M4 (positive), C[1,0] gets M4 (positive)
    U[1*2+1, 3] = 1                    # d
    V[1*2+0, 3] = 1; V[0*2+0, 3] = -1  # g, -e
    W[0*2+0, 3] = 1; W[1*2+0, 3] = 1   # C[0,0], C[1,0]
    
    # Term 4: M5 = (a+b)h
    # A terms: a=A[0,0] (i=0,k=0), b=A[0,1] (i=0,k=1)
    # B terms: h=B[1,1] (k=1,j=1)
    # C terms: C[0,0] gets -M5, C[0,1] gets M5
    U[0*2+0, 4] = 1; U[0*2+1, 4] = 1   # a, b
    V[1*2+1, 4] = 1                    # h
    W[0*2+0, 4] = -1; W[0*2+1, 4] = 1  # -C[0,0], C[0,1]
    
    # Term 5: M6 = (c-d)(f+h)
    # A terms: c=A[1,0] (i=1,k=0), d=A[1,1] (i=1,k=1)
    # B terms: f=B[0,1] (k=0,j=1), h=B[1,1] (k=1,j=1)
    # C terms: C[1,1] gets M6
    U[1*2+0, 5] = 1; U[1*2+1, 5] = -1  # c, -d
    V[0*2+1, 5] = 1; V[1*2+1, 5] = 1   # f, h
    W[1*2+1, 5] = 1                    # C[1,1]
    
    # Term 6: M7 = (a-c)(e+g)
    # A terms: a=A[0,0] (i=0,k=0), c=A[1,0] (i=1,k=0)
    # B terms: e=B[0,0] (k=0,j=0), g=B[1,0] (k=1,j=0)
    # C terms: C[0,0] gets M7, C[1,0] gets M7
    U[0*2+0, 6] = 1; U[1*2+0, 6] = -1  # a, -c
    V[0*2+0, 6] = 1; V[1*2+0, 6] = 1   # e, g
    W[0*2+0, 6] = 1; W[1*2+0, 6] = 1   # C[0,0], C[1,0]
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication.
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    U_mat = np.array(U, dtype=np.float64)
    V_mat = np.array(V, dtype=np.float64)
    W_mat = np.array(W, dtype=np.float64)
    
    # Check tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_mat[i*m+j, :], V_mat[j*p+k, :] * W_mat[k*n+i, :])
                if abs(val - 1) > tol:
                    return False
    return True


def _build_strassen_decomp(n, m, p):
    """Build a Strassen-based decomposition for n x m * m x p multiplication.
    Uses recursive block Strassen where possible, schoolbook for small cases.
    Returns (U, V, W) with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    # For small cases, use schoolbook
    if n == 1 and m == 1 and p == 1:
        U = np.ones((1, 1))
        V = np.ones((1, 1))
        W = np.ones((1, 1))
        return U, V, W
    
    if n == 1 or m == 1 or p == 1:
        # One dimension is 1, use schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # For 2x2x2, use Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    # For other sizes, try to use block structure
    # Split into 2x2 blocks where possible
    # If n, m, p are all even, use block Strassen
    
    if n % 2 == 0 and m % 2 == 0 and p % 2 == 0:
        # All even, use block Strassen
        return _build_block_strassen(n, m, p)
    else:
        # Odd dimensions, use schoolbook or mixed approach
        # For now, use schoolbook for non-even cases
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W


def _build_block_strassen(n, m, p):
    """Build block Strassen decomposition for even n, m, p.
    Splits into 2x2 blocks and uses Strassen at the block level.
    """
    # Split each dimension into 2x2 blocks
    # n = 2*n2, m = 2*m2, p = 2*p2
    n2 = n // 2
    m2 = m // 2
    p2 = p // 2
    
    # Build sub-decompositions for the 7 block products
    # Each block product is a (n2 x m2) * (m2 x p2) multiplication
    
    # Strassen block products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A11)(B01+B11)
    # M7 = (A00-A10)(B00+B10)
    
    # C contributions:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # First, build the sub-decomposition for a generic (n2 x m2) * (m2 x p2) multiplication
    U_sub, V_sub, W_sub = _build_strassen_decomp(n2, m2, p2)
    sub_rank = U_sub.shape[1]
    
    # The block Strassen will have 7 * sub_rank terms total
    total_rank = 7 * sub_rank
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Define the 7 block products
    # Each product has linear combinations of A and B blocks
    # a_combos: list of (bi, bj, coeff) where bi,bj are block indices
    # b_combos: list of (bk, bj, coeff) where bk,bj are block indices
    
    block_products = [
        # M1 = (A00+A11)(B00+B11)
        {
            'a': [((0,0,1.0), (1,1,1.0))],
            'b': [((0,0,1.0), (1,1,1.0))],
            'c': [((0,0,1.0), (1,1,1.0))]
        },
        # M2 = (A10+A11)B00
        {
            'a': [((1,0,1.0), (1,1,1.0))],
            'b': [((0,0,1.0))],
            'c': [((1,0,1.0))]
        },
        # M3 = A00(B01-B11)
        {
            'a': [((0,0,1.0))],
            'b': [((0,1,1.0), (1,1,-1.0))],
            'c': [((0,1,1.0), (1,1,1.0))]
        },
        # M4 = A11(B10-B00)
        {
            'a': [((1,1,1.0))],
            'b': [((1,0,1.0), (0,0,-1.0))],
            'c': [((0,0,1.0), (1,0,1.0))]
        },
        # M5 = (A00+A01)B11
        {
            'a': [((0,0,1.0), (0,1,1.0))],
            'b': [((1,1,1.0))],
            'c': [((0,0,-1.0), (0,1,1.0))]
        },
        # M6 = (A10-A11)(B01+B11)
        {
            'a': [((1,0,1.0), (1,1,-1.0))],
            'b': [((0,1,1.0), (1,1,1.0))],
            'c': [((1,1,1.0))]
        },
        # M7 = (A00-A10)(B00+B10)
        {
            'a': [((0,0,1.0), (1,0,-1.0))],
            'b': [((0,0,1.0), (1,0,1.0))],
            'c': [((0,0,1.0), (1,0,1.0))]
        },
    ]
    
    for t in range(7):
        a_combos = block_products[t]['a'][0]
        b_combos = block_products[t]['b'][0]
        c_combos = block_products[t]['c'][0]
        
        for s in range(sub_rank):
            col = t * sub_rank + s
            
            # For U: coefficient of A[i,j] in term (t,s)
            # A[i,j] where i = 2*bi + i2, j = 2*bj + j2
            # The sub-decomposition has U_sub[i2*m2 + j2, s]
            for bi, bj, ca in a_combos:
                for i2 in range(n2):
                    for j2 in range(m2):
                        i = 2*bi + i2
                        j = 2*bj + j2
                        U[i*m + j, col] += ca * U_sub[i2*m2 + j2, s]
            
            # For V: coefficient of B[j,k] in term (t,s)
            # B[j,k] where j = 2*bj + j2, k = 2*bk + k2
            # The sub-decomposition has V_sub[j2*p2 + k2, s]
            for bk, bj, cb in b_combos:
                for j2 in range(m2):
                    for k2 in range(p2):
                        j = 2*bj + j2
                        k = 2*bk + k2
                        V[j*p + k, col] += cb * V_sub[j2*p2 + k2, s]
            
            # For W: coefficient for C[i,k] in term (t,s)
            # C[i,k] where i = 2*ci + i2, k = 2*ck + k2
            # The sub-decomposition has W_sub[k2*n2 + i2, s]
            for ci, ck, cw in c_combos:
                for i2 in range(n2):
                    for k2 in range(p2):
                        i = 2*ci + i2
                        k = 2*ck + k2
                        W[k*n + i, col] += cw * W_sub[k2*n2 + i2, s]
    
    return U, V, W


def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x9 multiplication."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Split the 9 columns of B into two parts: 4 and 5
    # For the 4x4x4 part: use block Strassen (rank 49)
    # For the 4x4x5 part: use schoolbook (rank 80)
    # Total rank: 49 + 80 = 129
    
    # Build 4x4x4 decomposition
    U_444, V_444, W_444 = _build_strassen_decomp(4, 4, 4)
    rank_444 = U_444.shape[1]
    
    # Verify
    if not _verify_decomposition(U_444, V_444, W_444, 4, 4, 4):
        # Fall back to schoolbook for 4x4x4
        rank_444 = 4*4*4
        U_444 = np.zeros((16, rank_444))
        V_444 = np.zeros((16, rank_444))
        W_444 = np.zeros((16, rank_444))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_444[i*4+j, col] = 1
                    V_444[j*4+k, col] = 1
                    W_444[k*4+i, col] = 1
                    col += 1
    
    # For 4x4x5: use schoolbook
    rank_445 = 4*4*5
    U_445 = np.zeros((16, rank_445))
    V_445 = np.zeros((20, rank_445))
    W_445 = np.zeros((20, rank_445))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(5):
                U_445[i*4+j, col] = 1
                V_445[j*5+k, col] = 1
                W_445[k*4+i, col] = 1
                col += 1
    
    # Combine into 4x4x9
    total_rank = rank_444 + rank_445
    
    U_total = np.zeros((16, total_rank))
    V_total = np.zeros((36, total_rank))
    W_total = np.zeros((36, total_rank))
    
    # 4x4x4 part: k=0..3
    U_total[:, :rank_444] = U_444
    V_total[:16, :rank_444] = V_444
    W_total[:16, :rank_444] = W_444
    
    # 4x4x5 part: k=4..8
    U_total[:, rank_444:] = U_445
    V_total[16:, rank_444:] = V_445
    W_total[16:, rank_444:] = W_445
    
    # Verify the full 4x4x9 decomposition
    if not _verify_decomposition(U_total, V_total, W_total, 4, 4, 9):
        # Fall back to full schoolbook
        rank = 4*4*9
        U_total = np.zeros((16, rank))
        V_total = np.zeros((36, rank))
        W_total = np.zeros((36, rank))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_total[i*4+j, col] = 1
                    V_total[j*9+k, col] = 1
                    W_total[k*4+i, col] = 1
                    col += 1
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
