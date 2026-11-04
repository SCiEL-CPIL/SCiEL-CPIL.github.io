"""Matrix multiplication 4,5,5 with Strassen-based decomposition."""
import time
import random

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_4x4():
    """
    Build U, V, W for 4x4x4 matrix multiplication using Strassen's algorithm.
    U has shape (16, 19), V has shape (16, 19), W has shape (16, 19).
    
    The tensor identity: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = delta for A[i,j]*B[j,k]
    
    Actually, the matrix multiplication tensor T[i,j,k] = 1 where (i,j,k) is a valid triple.
    A decomposition of rank r gives matrices U (n*m x r), V (m*p x r), W (p*n x r) such that:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    For 4x4x4: n=m=p=4.
    U: (16, r), V: (16, r), W: (16, r)
    """
    n = 4
    # Strassen's 7 products for 2x2x2, recursively applied to get 4x4x4.
    # Base case: 2x2x2 Strassen uses rank 7.
    # 4x4x4: split into 2x2x2 blocks, each using 7 products, so 7*7 = 49? 
    # Actually, the recursive Strassen for 2^k x 2^k x 2^k has rank 7^k.
    # For 4x4x4 = 2^2, rank = 7^2 = 49.
    # For 8x8x8 = 2^3, rank = 7^3 = 343.
    
    # But wait, we can do better. The known minimum rank for 4x4x4 is not 49.
    # Actually, the schoolbook rank for 4x4x4 is 64.
    # Strassen recursive gives 49.
    # There are known constructions with rank less than 49 for 4x4x4? 
    # Actually, I recall that the minimum rank of the 4x4x4 tensor is unknown but bounded.
    # Let me just use the standard recursive Strassen construction which gives rank 49.
    
    # Actually, let me reconsider. The standard way to build the rank-r decomposition 
    # for n x n x n using Strassen recursively:
    # For 2x2x2: rank 7
    # For 4x4x4: we have 8 sub-blocks of 2x2x2, but Strassen reduces the 8 products to 7.
    # Wait, no. Strassen's algorithm for 2x2 matrix multiplication uses 7 scalar multiplications 
    # instead of 8. When we view matrix multiplication as a bilinear form, the tensor rank 
    # of the 2x2x2 matrix multiplication tensor is 7.
    # 
    # For 4x4x4: we can think of 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # The product C = A*B where A, B, C are 4x4 can be written in terms of 8 block products 
    # of 2x2 matrices, but Strassen reduces this to 7 block products.
    # Each block product is a 2x2x2 tensor of rank 7.
    # So total rank = 7 * 7 = 49.
    
    # Let me construct the 2x2x2 rank-7 decomposition first.
    # For 2x2x2: n=m=p=2
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # 
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # In tensor decomposition form:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # where i,j,k in {0,1}
    
    # Let me define the 7 terms:
    # r=0: M1 = (A11 + A22)(B11 + B22)
    #   This contributes to C11 (via +1), C22 (via +1)
    #   U rows: (0,0)=A11, (1,1)=A22 -> U[0*2+0,0]=1, U[1*2+1,0]=1
    #   V rows: (0,0)=B11, (1,1)=B22 -> V[0*2+0,0]=1, V[1*2+1,0]=1
    #   W rows: C11=(0,0), C22=(1,1) -> W[0*2+0,0]=1, W[1*2+1,0]=1
    #
    # r=1: M2 = (A21 + A22)B11
    #   Contributes to C21 (+1), C22 (-1)
    #   U: A21=(1,0), A22=(1,1) -> U[1*2+0,1]=1, U[1*2+1,1]=1
    #   V: B11=(0,0) -> V[0*2+0,1]=1
    #   W: C21=(1,0) -> W[1*2+0,1]=1, C22=(1,1) -> W[1*2+1,1]=-1
    #
    # r=2: M3 = A11(B21 - B22)
    #   Contributes to C12 (+1), C22 (+1)
    #   U: A11=(0,0) -> U[0*2+0,2]=1
    #   V: B21=(1,0), B22=(1,1) -> V[1*2+0,2]=1, V[1*2+1,2]=-1
    #   W: C12=(0,1) -> W[0*2+1,2]=1, C22=(1,1) -> W[1*2+1,2]=1
    #
    # r=3: M4 = A22(B12 - B11)
    #   Contributes to C11 (+1), C21 (+1)
    #   U: A22=(1,1) -> U[1*2+1,3]=1
    #   V: B12=(0,1), B11=(0,0) -> V[0*2+1,3]=1, V[0*2+0,3]=-1
    #   W: C11=(0,0) -> W[0*2+0,3]=1, C21=(1,0) -> W[1*2+0,3]=1
    #
    # r=4: M5 = (A11 + A12)B22
    #   Contributes to C11 (-1), C12 (+1)
    #   U: A11=(0,0), A12=(0,1) -> U[0*2+0,4]=1, U[0*2+1,4]=1
    #   V: B22=(1,1) -> V[1*2+1,4]=1
    #   W: C11=(0,0) -> W[0*2+0,4]=-1, C12=(0,1) -> W[0*2+1,4]=1
    #
    # r=5: M6 = (A21 - A11)(B11 + B12)
    #   Contributes to C22 (+1)
    #   U: A21=(1,0), A11=(0,0) -> U[1*2+0,5]=1, U[0*2+0,5]=-1
    #   V: B11=(0,0), B12=(0,1) -> V[0*2+0,5]=1, V[0*2+1,5]=1
    #   W: C22=(1,1) -> W[1*2+1,5]=1
    #
    # r=6: M7 = (A12 - A22)(B21 + B22)
    #   Contributes to C11 (+1), C22 (-1)
    #   U: A12=(0,1), A22=(1,1) -> U[0*2+1,6]=1, U[1*2+1,6]=-1
    #   V: B21=(1,0), B22=(1,1) -> V[1*2+0,6]=1, V[1*2+1,6]=1
    #   W: C11=(0,0) -> W[0*2+0,6]=1, C22=(1,1) -> W[1*2+1,6]=-1
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    # r=1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    # r=2
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    # r=3
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    # r=4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    # r=5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    # r=6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = -1
    
    # Now build 4x4x4 from 2x2x2 using Strassen recursion.
    # The 4x4 matrices are viewed as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    # 
    # Using Strassen at the block level (7 block products):
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B10-B11)
    # P4 = A11*(B01-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    # 
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # Each Pi is a product of two 2x2 matrices, which has tensor rank 7.
    # So total rank = 7 * 7 = 49.
    
    # Let me build the 4x4x4 decomposition.
    # n = 4, so U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # For each block-level product Pi (i=0..6), we have a 2x2x2 decomposition with 7 terms.
    # For Pi, the A-factor is some combination of A-blocks, B-factor is some combination of B-blocks,
    # and the result contributes to C-blocks with certain coefficients.
    
    # Let me define the block-level coefficients:
    # For Pi: a_factors = list of (block_idx, coeff) for A
    #           b_factors = list of (block_idx, coeff) for B
    #           c_factors = list of (block_idx, coeff) for C
    
    block_decomps = [
        # P1 = (A00+A11)*(B00+B11), contributes to C00(+1), C11(+1)
        {
            'a': [(0, 1), (3, 1)],
            'b': [(0, 1), (3, 1)],
            'c': [(0, 1), (3, 1)]
        },
        # P2 = (A10+A11)*B00, contributes to C10(+1), C11(-1)
        {
            'a': [(2, 1), (3, 1)],
            'b': [(0, 1)],
            'c': [(2, 1), (3, -1)]
        },
        # P3 = A00*(B10-B11), contributes to C01(+1), C11(+1)
        {
            'a': [(0, 1)],
            'b': [(2, 1), (3, -1)],
            'c': [(1, 1), (3, 1)]
        },
        # P4 = A11*(B01-B00), contributes to C00(+1), C10(+1)
        {
            'a': [(3, 1)],
            'b': [(1, 1), (0, -1)],
            'c': [(0, 1), (2, 1)]
        },
        # P5 = (A00+A01)*B11, contributes to C00(-1), C01(+1)
        {
            'a': [(0, 1), (1, 1)],
            'b': [(3, 1)],
            'c': [(0, -1), (1, 1)]
        },
        # P6 = (A10-A00)*(B00+B01), contributes to C11(+1)
        {
            'a': [(2, 1), (0, -1)],
            'b': [(0, 1), (1, 1)],
            'c': [(3, 1)]
        },
        # P7 = (A01-A11)*(B10+B11), contributes to C00(+1), C11(-1)
        {
            'a': [(1, 1), (3, -1)],
            'b': [(2, 1), (3, 1)],
            'c': [(0, 1), (3, -1)]
        },
    ]
    
    # For each block product Pi and each 2x2x2 term r (0..6), we get a rank-1 term.
    # The U matrix for the 4x4 case:
    #   For A-block ba with coefficient ca, and within that block, 2x2 index (i2, j2) with 2x2x2 coefficient u2[i2*2+j2, r]
    #   The 4x4 index is i = i2*2 + i_block_row, j = j2*2 + j_block_col
    #   Wait, I need to be more careful about the indexing.
    
    # Let me think about this differently. 
    # In the 2x2x2 decomposition, U2[i*2+j, r] where i,j are 0 or 1.
    # When we embed this into a block, say A00 which occupies rows 0-1, cols 0-1 of the 4x4 matrix:
    #   The 4x4 row index for (i2, j2) in block (br, bc) is i2*2 + br*2, and j is j2*2 + bc*2.
    #   Wait, no. The 4x4 matrix A has entries A[i,j] for i,j in 0..3.
    #   Block A00 is A[0:2, 0:2], block A01 is A[0:2, 2:4], block A10 is A[2:4, 0:2], block A11 is A[2:4, 2:4].
    #   So block (br, bc) where br, bc in {0,1} corresponds to rows br*2..br*2+1 and cols bc*2..bc*2+1.
    #   The 4x4 index for 2x2 index (i2, j2) in block (br, bc) is (br*2 + i2, bc*2 + j2).
    #   So the row index in U (which is i*4+j) is (br*2 + i2)*4 + (bc*2 + j2).
    
    # Similarly for V: V[j*4+k] where j is the row of B (which is col of A) and k is the col of B.
    #   Block (br, bc) of B: rows br*2..br*2+1, cols bc*2..bc*2+1.
    #   2x2 index (j2, k2) in block (br, bc) -> 4x4 index (br*2 + j2, bc*2 + k2).
    #   V row index = (br*2 + j2)*4 + (bc*2 + k2).
    
    # For W: W[k*4+i] where k is the row of C (which is row of B... wait no).
    #   Actually, W[k*n+i] where k is the col index of B / row index of C, and i is the row index of A / col index of C.
    #   Wait, let me re-read the problem statement.
    #   "The output factor follows the official trace convention W[k*n+i,r]"
    #   So W has shape (p*n, rank) and the row index is k*n + i.
    #   For n=4: W[k*4+i, r]
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #   So i is the row of C (and A), k is the col of C (and B).
    #   Block (cr, cc) of C: rows cr*2..cr*2+1, cols cc*2..cc*2+1.
    #   2x2 index (i2, k2) in block (cr, cc) -> 4x4 index (cr*2 + i2, cc*2 + k2).
    #   W row index = (cc*2 + k2)*4 + (cr*2 + i2).
    
    # Now, for each block product Pi with a_factors, b_factors, c_factors:
    # For each 2x2x2 term r (0..6):
    #   U contribution: for each (ba, ca) in a_factors, for each (i2, j2) where U2[i2*2+j2, r] != 0:
    #     U[4x4_idx] += ca * U2[i2*2+j2, r]
    #   Similarly for V and W.
    
    # Let me build this:
    col = 0
    for pi_idx, block_decomp in enumerate(block_decomps):
        for r in range(7):
            # This is term col in the 49-term decomposition
            # U contribution
            for ba, ca in block_decomp['a']:
                br, bc = divmod(ba, 2)  # ba is block index 0..3, br=ba//2, bc=ba%2
                for i2 in range(2):
                    for j2 in range(2):
                        val = U2[i2*2 + j2][r]
                        if val != 0:
                            u_idx = (br*2 + i2)*4 + (bc*2 + j2)
                            U4[u_idx][col] += ca * val
            # V contribution
            for bb, cb in block_decomp['b']:
                br, bc = divmod(bb, 2)
                for j2 in range(2):
                    for k2 in range(2):
                        val = V2[j2*2 + k2][r]
                        if val != 0:
                            v_idx = (br*2 + j2)*4 + (bc*2 + k2)
                            V4[v_idx][col] += cb * val
            # W contribution
            for bc_block, cc in block_decomp['c']:
                br, bc = divmod(bc_block, 2)
                for i2 in range(2):
                    for k2 in range(2):
                        val = W2[k2*2 + i2][r]
                        if val != 0:
                            w_idx = (bc*2 + k2)*4 + (br*2 + i2)
                            W4[w_idx][col] += cc * val
            col += 1
    
    return U4, V4, W4

def _embed_4x4_into_4x5x5(U4, V4, W4):
    """
    Embed the 4x4x4 decomposition into 4x5x5.
    U4: (16, 49), V4: (16, 49), W4: (16, 49)
    Full: U: (20, rank), V: (25, rank), W: (20, rank)
    
    The 4x5 matrix A has rows i=0..3, cols j=0..4.
    The 5x5 matrix B has rows j=0..4, cols k=0..4.
    The 4x5 matrix C has rows i=0..3, cols k=0..4.
    
    C[i,k] = sum_{j=0}^{4} A[i,j] B[j,k]
    
    We can split this into:
    sum_{j=0}^{3} A[i,j] B[j,k] + A[i,4] B[4,k]
    
    The first part is a 4x4x4 multiplication (j=0..3, k=0..3... wait, k goes to 4).
    Actually, let me split more carefully.
    
    C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k] + A[i,4] B[4,k]
    
    For the first term: j=0..3, but k can be 0..4. So it's not a clean 4x4x4.
    
    Let me think differently. I'll use the schoolbook for the parts involving j=4 or k=4, 
    and Strassen for the 4x4x4 core (j=0..3, k=0..3).
    
    Actually, let me just do:
    - Core: i=0..3, j=0..3, k=0..3: use Strassen 4x4x4 (rank 49)
    - Strip 1: i=0..3, j=4, k=0..4: A[i,4] * B[4,k] for k=0..4, this is 4*5 = 20 terms, rank 20
    - Strip 2: i=0..3, j=0..3, k=4: A[i,j] * B[j,4] for j=0..3, this is 4*4 = 16 terms, rank 16
    
    Total rank: 49 + 20 + 16 = 85
    
    Wait, but the core covers k=0..3, and strip 2 covers k=4 with j=0..3.
    Strip 1 covers j=4 with k=0..4.
    So the term A[i,4]*B[4,4] is in strip 1, and the core doesn't include j=4 or k=4.
    This should be correct.
    
    U full: (20, 85)
    V full: (25, 85)  
    W full: (20, 85)
    
    For the core (4x4x4 with j=0..3, k=0..3):
    U4 has rows indexed by i*4+j for i=0..3, j=0..3.
    Full U has rows indexed by i*5+j for i=0..3, j=0..4.
    So U4 row (i*4+j) maps to full U row (i*5+j).
    
    V4 has rows indexed by j*4+k for j=0..3, k=0..3.
    Full V has rows indexed by j*5+k for j=0..4, k=0..4.
    So V4 row (j*4+k) maps to full V row (j*5+k).
    
    W4 has rows indexed by k*4+i for k=0..3, i=0..3.
    Full W has rows indexed by k*5+i for k=0..4, i=0..3.
    So W4 row (k*4+i) maps to full W row (k*5+i).
    
    For strip 1 (j=4, k=0..4, i=0..3):
    Each term is A[i,4] * B[4,k] * C[i,k]
    U: row i*5+4, col = 49 + i*5+k... wait, I need to assign columns.
    
    Let me assign:
    Columns 0..48: core (4x4x4 Strassen)
    Columns 49..68: strip 1 (j=4, i=0..3, k=0..4): 20 terms
    Columns 69..84: strip 2 (k=4, j=0..3, i=0..3): 16 terms
    
    Total: 85 columns.
    """
    n, m, p = 4, 5, 5
    rank = 85
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Core: columns 0..48
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U[i*5 + j][r] = U4[i*4 + j][r]
        for j in range(4):
            for k in range(4):
                V[j*5 + k][r] = V4[j*4 + k][r]
        for k in range(4):
            for i in range(4):
                W[k*5 + i][r] = W4[k*4 + i][r]
    
    # Strip 1: j=4, i=0..3, k=0..4: columns 49..68
    col = 49
    for i in range(4):
        for k in range(5):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*5 + i][col] = 1
            col += 1
    
    # Strip 2: k=4, j=0..3, i=0..3: columns 69..84
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*5 + i][col] = 1
            col += 1
    
    return U, V, W

def _verify(U, V, W, n=4, m=5, p=5):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(val - 1) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    
    U4, V4, W4 = _strassen_4x4()
    
    # Verify 4x4x4 decomposition
    if not _verify(U4, V4, W4, n=4, m=4, p=4):
        # Fall back to schoolbook for 4x4x4
        n4 = 4
        rank4 = n4 * n4 * n4
        U4 = [[0]*rank4 for _ in range(n4*n4)]
        V4 = [[0]*rank4 for _ in range(n4*n4)]
        W4 = [[0]*rank4 for _ in range(n4*n4)]
        col = 0
        for i in range(n4):
            for j in range(n4):
                for k in range(n4):
                    U4[i*n4 + j][col] = 1
                    V4[j*n4 + k][col] = 1
                    W4[k*n4 + i][col] = 1
                    col += 1
    
    U, V, W = _embed_4x4_into_4x5x5(U4, V4, W4)
    
    # Verify final decomposition
    if not _verify(U, V, W):
        # Fall back to full schoolbook
        n, m, p = 4, 5, 5
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    elapsed = time.time() - start_time
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
