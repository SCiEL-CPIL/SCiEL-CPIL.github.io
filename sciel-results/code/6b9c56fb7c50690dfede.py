"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
                if max_err > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2 multiplication.
    Returns U (4x7), V (4x7), W (4x7) with integer entries.
    """
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U: rows indexed by (i,j) of A, 0..3 -> [0,1,2,3] for [[0,0],[0,1],[1,0],[1,1]]
    # V: rows indexed by (j,k) of B, 0..3 -> [0,1,2,3] for [[0,0],[0,1],[1,0],[1,1]]
    # W: rows indexed by (k,i) of C, 0..3 -> [0,1,2,3] for [[0,0],[0,1],[1,0],[1,1]]
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1: A11+A22, B11+B22 -> C11, C22
    U[0, 0] = 1  # A[0,0]
    U[3, 0] = 1  # A[1,1]
    V[0, 0] = 1  # B[0,0]
    V[3, 0] = 1  # B[1,1]
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2: A21+A22, B11 -> C21, C22
    U[2, 1] = 1  # A[1,0]
    U[3, 1] = 1  # A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[2, 1] = 1  # C[1,0]
    W[3, 1] = -1  # C[1,1]
    
    # M3: A11, B12-B22 -> C12, C22
    U[0, 2] = 1  # A[0,0]
    V[1, 2] = 1  # B[0,1]
    V[3, 2] = -1  # B[1,1]
    W[1, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4: A22, B21-B11 -> C11, C21
    U[3, 3] = 1  # A[1,1]
    V[2, 3] = 1  # B[1,0]
    V[0, 3] = -1  # B[0,0]
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # M5: A11+A12, B22 -> C11, C12
    U[0, 4] = 1  # A[0,0]
    U[1, 4] = 1  # A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[0, 4] = -1  # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # M6: A21-A11, B11+B12 -> C22
    U[2, 5] = 1  # A[1,0]
    U[0, 5] = -1  # A[0,0]
    V[0, 5] = 1  # B[0,0]
    V[1, 5] = 1  # B[0,1]
    W[3, 5] = 1  # C[1,1]
    
    # M7: A12-A22, B21+B22 -> C11
    U[1, 6] = 1  # A[0,1]
    U[3, 6] = -1  # A[1,1]
    V[2, 6] = 1  # B[1,0]
    V[3, 6] = 1  # B[1,1]
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W

def _naive_2x2x3():
    """
    Naive decomposition for 2x2 * 2x3 -> 2x3 multiplication.
    Returns U (4x12), V (6x12), W (6x12) with integer entries.
    """
    U = np.zeros((4, 12), dtype=int)
    V = np.zeros((6, 12), dtype=int)
    W = np.zeros((6, 12), dtype=int)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i * 2 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 2 + i, col] = 1
                col += 1
    
    return U, V, W

def _strassen_4x4x5():
    """
    Construct Strassen-style decomposition for 4x4 * 4x5 -> 4x5.
    Split into 2x2 blocks.
    
    A is 4x4, split into four 2x2 blocks:
        A = [[A00, A01],
             [A10, A11]]
    
    B is 4x5. We can split columns into 2x2 and 2x1:
        B = [[B00, B01, B02],
             [B10, B11, B12]]
    where B00, B10 are 2x2, B01, B11 are 2x2, B02, B12 are 2x1.
    
    Actually, let's be more careful. B is 4x5.
    Rows: split into top 2 and bottom 2.
    Columns: split into first 2, next 2, last 1.
    
    So B has blocks:
        B = [[B00, B01, B02],
             [B10, B11, B12]]
    where B00, B01, B10, B11 are 2x2, and B02, B12 are 2x1.
    
    C = A * B is 4x5:
        C = [[C00, C01, C02],
             [C10, C11, C12]]
    where C00, C01, C10, C11 are 2x2, and C02, C12 are 2x1.
    
    For the 2x2 * 2x2 -> 2x2 multiplications, use Strassen (rank 7).
    For the 2x2 * 2x1 -> 2x1 multiplications, use naive (rank 4).
    
    Total rank = 7 * 4 (for the 2x2*2x2 blocks) + 4 * 2 (for the 2x2*2x1 blocks)
               = 28 + 8 = 36
    
    Wait, let me think about this more carefully.
    
    C00 = A00*B00 + A01*B10
    C01 = A00*B01 + A01*B11
    C02 = A00*B02 + A01*B12
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    C12 = A10*B02 + A11*B12
    
    Each 2x2*2x2 product needs rank 7 (Strassen).
    Each 2x2*2x1 product needs rank 4 (naive).
    
    But we can combine terms. Let's use the tensor product approach.
    
    Actually, let's just build it directly by combining Strassen for 2x2x2 
    and naive for 2x2x1.
    
    We'll have 4 Strassen instances (for the 2x2*2x2 blocks) and 
    2 naive instances (for the 2x2*2x1 blocks).
    
    Total rank = 4*7 + 2*4 = 28 + 8 = 36.
    """
    # We need to construct U (16 x rank), V (20 x rank), W (20 x rank)
    # A is 4x4: indices (i,j) for i in 0..3, j in 0..3
    # B is 4x5: indices (j,k) for j in 0..3, k in 0..4
    # C is 4x5: indices (k,i) for k in 0..4, i in 0..3
    
    # Block structure:
    # A blocks: A00=(0,0),(0,1),(1,0),(1,1); A01=(0,2),(0,3),(1,2),(1,3)
    #           A10=(2,0),(2,1),(3,0),(3,1); A11=(2,2),(2,3),(3,2),(3,3)
    # B blocks: B00=(0,0),(0,1),(1,0),(1,1); B01=(0,2),(0,3),(1,2),(1,3)
    #           B02=(0,4),(1,4)
    #           B10=(2,0),(2,1),(3,0),(3,1); B11=(2,2),(2,3),(3,2),(3,3)
    #           B12=(2,4),(3,4)
    # C blocks: C00=(0,0),(0,1),(1,0),(1,1); C01=(0,2),(0,3),(1,2),(1,3)
    #           C02=(0,4),(1,4)
    #           C10=(2,0),(2,1),(3,0),(3,1); C11=(2,2),(2,3),(3,2),(3,3)
    #           C12=(2,4),(3,4)
    
    # Get Strassen and naive factors
    S_U, S_V, S_W = _strassen_2x2x2()
    N_U, N_V, N_W = _naive_2x2x3()
    
    # We'll build the full decomposition by placing each sub-decomposition
    # at the appropriate rows/columns.
    
    # Total rank = 4 * 7 + 2 * 4 = 36
    total_rank = 4 * 7 + 2 * 4
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((20, total_rank), dtype=int)
    W = np.zeros((20, total_rank), dtype=int)
    
    # Define block row/column indices
    # A block (bi, bj) where bi in {0,1}, bj in {0,1}
    # Row index in A: bi*2 + local_i, Col index in A: bj*2 + local_j
    def a_idx(bi, bj, li, lj):
        return (bi * 2 + li) * 4 + (bj * 2 + lj)
    
    # B block (bi, bk) where bi in {0,1}, bk in {0,1,2}
    # For bk=0,1: 2x2 blocks; for bk=2: 2x1 block
    # Row index in B: bi*2 + local_j, Col index in B: depends on bk
    def b_idx_2x2(bi, bk, lj, lk):
        # bk=0: cols 0,1; bk=1: cols 2,3
        col = bk * 2 + lk
        return (bi * 2 + lj) * 5 + col
    
    def b_idx_2x1(bi, lk):
        # bk=2: col 4
        col = 4
        return (bi * 2 + lk) * 5 + col
    
    # C block (ck, ci) where ck in {0,1,2}, ci in {0,1}
    # For ck=0,1: 2x2 blocks; for ck=2: 2x1 block
    # Row index in C: depends on ck, Col index in C: ci*2 + local_i
    def c_idx_2x2(ck, ci, lk, li):
        # ck=0: rows 0,1; ck=1: rows 2,3
        row = ck * 2 + lk
        return row * 4 + (ci * 2 + li)
    
    def c_idx_2x1(ck, li):
        # ck=2: row 4
        row = 4
        return row * 4 + (ci * 2 + li) if False else row * 4 + li
    
    # Actually, W is indexed by (k*n + i) where n=4, so k in 0..4, i in 0..3
    # W index = k * 4 + i
    def w_idx(k, i):
        return k * 4 + i
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # For each product, we place the Strassen or naive factors.
    
    # Helper to place a 2x2x2 Strassen decomposition
    def place_strassen(a_block, b_block, c_block, offset):
        """Place Strassen decomposition for a_block * b_block -> c_block."""
        # a_block: (bi, bj)
        # b_block: (bi, bk)
        # c_block: (ck, ci)
        bi_a, bj_a = a_block
        bi_b, bk_b = b_block
        ck_c, ci_c = c_block
        
        for r in range(4):
            for c in range(7):
                # U
                if S_U[r, c] != 0:
                    U[a_idx(bi_a, bj_a, r // 2, r % 2), offset + c] += S_U[r, c]
                # V
                if S_V[r, c] != 0:
                    V[b_idx_2x2(bi_b, bk_b, r // 2, r % 2), offset + c] += S_V[r, c]
                # W
                if S_W[r, c] != 0:
                    W[w_idx(ck_c * 2 + r // 2, ci_c * 2 + r % 2), offset + c] += S_W[r, c]
    
    # Helper to place a 2x2x1 naive decomposition
    def place_naive_2x1(a_block, b_block, c_block, offset):
        """Place naive decomposition for a_block * b_block(2x1) -> c_block(2x1)."""
        bi_a, bj_a = a_block
        bi_b = b_block  # b_block is just the row block index for 2x1
        ci_c = c_block  # c_block is just the col block index for 2x1
        
        # Naive 2x2 * 2x1 -> 2x1: rank 4
        # A is 2x2, B is 2x1, C is 2x1
        # C[i] = sum_j A[i,j] * B[j]
        for i in range(2):
            for j in range(2):
                col = offset + i * 2 + j
                U[a_idx(bi_a, bj_a, i, j), col] = 1
                V[b_idx_2x1(bi_b, j), col] = 1
                W[w_idx(4, ci_c * 2 + i), col] = 1  # Wait, this is wrong
    
    # Let me redo this more carefully.
    
    # Reset
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((20, total_rank), dtype=int)
    W = np.zeros((20, total_rank), dtype=int)
    
    offset = 0
    
    # C00 = A00*B00 + A01*B10
    # First term: A00 * B00 -> C00
    for r in range(4):
        for c in range(7):
            if S_U[r, c] != 0:
                U[a_idx(0, 0, r // 2, r % 2), offset + c] += S_U[r, c]
            if S_V[r, c] != 0:
                V[b_idx_2x2(0, 0, r // 2, r % 2), offset + c] += S_V[r, c]
            if S_W[r, c] != 0:
                W[w_idx(0 * 2 + r // 2, 0 * 2 + r % 2), offset + c] += S_W[r, c]
    offset += 7
    
    # Second term: A01 * B10 -> C00
    for r in range(4):
        for c in range(7):
            if S_U[r, c] != 0:
                U[a_idx(0, 1, r // 2, r % 2), offset + c] += S_U[r, c]
            if S_V[r, c] != 0:
                V[b_idx_2x2(1, 0, r // 2, r % 2), offset + c] += S_V[r, c]
            if S_W[r, c] != 0:
                W[w_idx(0 * 2 + r // 2, 0 * 2 + r % 2), offset + c] += S_W[r, c]
    offset += 7
    
    # C01 = A00*B01 + A01*B11
    # First term: A00 * B01 -> C01
    for r in range(4):
        for c in range(7):
            if S_U[r, c] != 0:
                U[a_idx(0, 0, r // 2, r % 2), offset + c] += S_U[r, c]
            if S_V[r, c] != 0:
                V[b_idx_2x2(0, 1, r // 2, r % 2), offset + c] += S_V[r, c]
            if S_W[r, c] != 0:
                W[w_idx(0 * 2 + r // 2, 1 * 2 + r % 2), offset + c] += S_W[r, c]
    offset += 7
    
    # Second term: A01 * B11 -> C01
    for r in range(4):
        for c in range(7):
            if S_U[r, c] != 0:
                U[a_idx(0, 1, r // 2, r % 2), offset + c] += S_U[r, c]
            if S_V[r, c] != 0:
                V[b_idx_2x2(1, 1, r // 2, r % 2), offset + c] += S_V[r, c]
            if S_W[r, c] != 0:
                W[w_idx(0 * 2 + r // 2, 1 * 2 + r % 2), offset + c] += S_W[r, c]
    offset += 7
    
    # C02 = A00*B02 + A01*B12
    # Naive 2x2 * 2x1 -> 2x1, rank 4 each
    # First term: A00 * B02 -> C02
    for i in range(2):
        for j in range(2):
            col = offset + i * 2 + j
            U[a_idx(0, 0, i, j), col] = 1
            V[b_idx_2x1(0, j), col] = 1
            W[w_idx(4, i), col] = 1
    offset += 4
    
    # Second term: A01 * B12 -> C02
    for i in range(2):
        for j in range(2):
            col = offset + i * 2 + j
            U[a_idx(0, 1, i, j), col] = 1
            V[b_idx_2x1(1, j), col] = 1
            W[w_idx(4, i), col] = 1
    offset += 4
    
    # C10 = A10*B00 + A11*B10
    # First term: A10 * B00 -> C10
    for r in range(4):
        for c in range(7):
            if S_U[r, c] != 0:
                U[a_idx(1, 0, r // 2, r % 2), offset + c] += S_U[r, c]
            if S_V[r, c] != 0:
                V[b_idx_2x2(0, 0, r // 2, r % 2), offset + c] += S_V[r, c]
            if S_W[r, c] != 0:
                W[w_idx(1 * 2 + r // 2, 0 * 2 + r % 2), offset + c] += S_W[r, c]
    offset += 7
    
    # Second term: A11 * B10 -> C10
    for r in range(4):
        for c in range(7):
            if S_U[r, c] != 0:
                U[a_idx(1, 1, r // 2, r % 2), offset + c] += S_U[r, c]
            if S_V[r, c] != 0:
                V[b_idx_2x2(1, 0, r // 2, r % 2), offset + c] += S_V[r, c]
            if S_W[r, c] != 0:
                W[w_idx(1 * 2 + r // 2, 0 * 2 + r % 2), offset + c] += S_W[r, c]
    offset += 7
    
    # C11 = A10*B01 + A11*B11
    # First term: A10 * B01 -> C11
    for r in range(4):
        for c in range(7):
            if S_U[r, c] != 0:
                U[a_idx(1, 0, r // 2, r % 2), offset + c] += S_U[r, c]
            if S_V[r, c] != 0:
                V[b_idx_2x2(0, 1, r // 2, r % 2), offset + c] += S_V[r, c]
            if S_W[r, c] != 0:
                W[w_idx(1 * 2 + r // 2, 1 * 2 + r % 2), offset + c] += S_W[r, c]
    offset += 7
    
    # Second term: A11 * B11 -> C11
    for r in range(4):
        for c in range(7):
            if S_U[r, c] != 0:
                U[a_idx(1, 1, r // 2, r % 2), offset + c] += S_U[r, c]
            if S_V[r, c] != 0:
                V[b_idx_2x2(1, 1, r // 2, r % 2), offset + c] += S_V[r, c]
            if S_W[r, c] != 0:
                W[w_idx(1 * 2 + r // 2, 1 * 2 + r % 2), offset + c] += S_W[r, c]
    offset += 7
    
    # C12 = A10*B02 + A11*B12
    # First term: A10 * B02 -> C12
    for i in range(2):
        for j in range(2):
            col = offset + i * 2 + j
            U[a_idx(1, 0, i, j), col] = 1
            V[b_idx_2x1(0, j), col] = 1
            W[w_idx(4, i), col] = 1
    offset += 4
    
    # Second term: A11 * B12 -> C12
    for i in range(2):
        for j in range(2):
            col = offset + i * 2 + j
            U[a_idx(1, 1, i, j), col] = 1
            V[b_idx_2x1(1, j), col] = 1
            W[w_idx(4, i), col] = 1
    offset += 4
    
    assert offset == total_rank, f"Offset {offset} != total_rank {total_rank}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Ensure we have a valid baseline
    baseline = _schoolbook(n, m, p)
    
    # Try Strassen-style decomposition
    try:
        result = _strassen_4x4x5()
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-6):
            rank = len(result["U"][0])
            if rank < n * m * p:
                return result
    except Exception as e:
        print(f"Strassen failed: {e}")
    
    # Fallback to schoolbook
    return baseline
# EVOLVE-BLOCK-END
