"""Matrix multiplication 4x5x6 - rank-minimal exact decomposition."""
import time

DIMENSIONS = [4, 5, 6]
_DEADLINE = time.time() + 110.0

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Rank-7 Strassen for 2x2x2. U is 4x7 (A-index), V is 4x7 (B-index), W is 4x7 (C-index).
    Strassen intermediates:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U: rows indexed by A-position (a11,a12,a21,a22) as 0,1,2,3
    # V: rows indexed by B-position (b11,b12,b21,b22) as 0,1,2,3
    # W: rows indexed by C-position (c11,c12,c21,c22) as 0,1,2,3
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1: A11+A22, B11+B22 -> C11+=M1, C22+=M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2: A21+A22, B11 -> C21+=M2, C22-=M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3: A11, B21-B22 -> C12+=M3, C22+=M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4: A22, B12-B11 -> C11+=M4, C21+=M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5: A11+A12, B22 -> C11-=M5, C12+=M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6: A21-A11, B11+B12 -> C22+=M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7: A12-A22, B21+B22 -> C11+=M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


def _build_block(n, m, p, method):
    """Build a decomposition for n x m x p matrix multiplication.
    Returns (U, V, W) where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    method: 'strassen' (uses 2x2x2 Strassen blocks where possible), 'schoolbook'
    """
    # For this specific case, we handle recursively.
    # We'll build by splitting the problem into 2x2x2 Strassen blocks + schoolbook remainder.
    
    if method == 'schoolbook':
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
        return U, V, W
    
    # Recursive Strassen-based construction
    # For 2x2x2, use Strassen directly
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    
    # For larger cases, we'll use a block-based approach.
    # Since we only need specific cases (4x4x4, 4x4x2), we handle them specially.
    if n == 4 and m == 4 and p == 4:
        return _build_4x4x4()
    if n == 4 and m == 4 and p == 2:
        return _build_4x4x2()
    if n == 4 and m == 1 and p == 6:
        return _build_4x1x6()
    
    # Fallback: schoolbook
    return _build_block(n, m, p, 'schoolbook')


def _build_4x4x4():
    """4x4x4 using four 2x2x2 Strassen blocks. Rank 4*7 = 28... wait, that's not right.
    Actually, 4x4x4 can be done with rank 49 using the 4x4x4 Strassen-like decomposition.
    But the simplest is to tile with 2x2x2 Strassen: 2*2*2 = 8 blocks, each rank 7, total rank 56.
    However, a better 4x4x4 decomposition has rank 49 (Lader). Let's use the tiling approach for simplicity: rank 56.
    Actually wait - the standard 4x4x4 via 2x2x2 tiling is 8 blocks x 7 = 56.
    But we can do better. Let me use the known rank-49 decomposition for 4x4x4.
    For now, let's use 2x2x2 tiling (rank 56) and see if total still beats 120.
    56 + 28 + 24 = 108. Still better than 120.
    Let me try to get rank 49 for 4x4x4.
    """
    # Use 2x2x2 tiling: 8 Strassen blocks
    # A is 4x4, split into 4 2x2 blocks: A11,A12,A21,A22
    # B is 4x4, split into 4 2x2 blocks: B11,B12,B21,B22
    # C is 4x4, split into 4 2x2 blocks: C11,C12,C21,C22
    # C_ij = sum_k A_ik B_kj, where each product is 2x2x2
    # So C11 = A11*B11 + A12*B21 (each 2x2x2)
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Total 8 Strassen blocks, rank 56.
    
    # For each (i_block, j_block) pair of output blocks, we need sum over k_block.
    # Block (bi, bj) of C = sum_k A_block(bi,k) * B_block(k,bj)
    # Each A_block(bi,k) is 2x2, B_block(k,bj) is 2x2, product is 2x2x2 Strassen.
    
    su, sv, sw = _strassen_2x2x2()  # 4x7 each
    
    U = [[0]*56 for _ in range(16)]
    V = [[0]*56 for _ in range(16)]
    W = [[0]*56 for _ in range(16)]
    
    col = 0
    for bi in range(2):      # output row block
        for bj in range(2):  # output col block
            for bk in range(2):  # summation block
                # A_block(bi, bk): rows (bi*2, bi*2+1), cols (bk*2, bk*2+1)
                # B_block(bk, bj): rows (bk*2, bk*2+1), cols (bj*2, bj*2+1)
                # C_block(bi, bj): rows (bi*2, bi*2+1), cols (bj*2, bj*2+1)
                for a_pos in range(4):
                    ai = a_pos // 2
                    aj = a_pos % 2
                    U[(bi*2+ai)*4 + (bk*2+aj)][col + a_pos] = su[a_pos][0] if col % 7 == 0 else 0
                # Actually, I need to map properly. Let me redo this.
                # For each Strassen block at position (bi,bk,bj):
                # U rows: the 4 A-entries of block (bi,bk) -> global A indices
                # V rows: the 4 B-entries of block (bk,bj) -> global B indices
                # W rows: the 4 C-entries of block (bi,bj) -> global C indices
                # Each Strassen block contributes 7 columns.
                for t in range(4):
                    # A entry t in block (bi,bk): local (t//2, t%2)
                    # Global A index: row = bi*2 + t//2, col = bk*2 + t%2
                    a_row = bi*2 + t//2
                    a_col = bk*2 + t%2
                    for c in range(7):
                        U[a_row*4 + a_col][col + c] = su[t][c]
                    
                    # B entry t in block (bk,bj): local (t//2, t%2)
                    # Global B index: row = bk*2 + t//2, col = bj*2 + t%2
                    b_row = bk*2 + t//2
                    b_col = bj*2 + t%2
                    for c in range(7):
                        V[b_row*4 + b_col][col + c] = sv[t][c]
                    
                    # C entry t in block (bi,bj): local (t//2, t%2)
                    # Global C index: row = bi*2 + t//2, col = bj*2 + t%2
                    # W index: W[k*n + i] where k is col of C, i is row of C
                    # C entry t: row = t//2, col = t%2 (local)
                    # Global C: row = bi*2 + t//2, col = bj*2 + t%2
                    c_row = bi*2 + t//2
                    c_col = bj*2 + t//2  # wrong
                    c_col = bj*2 + t%2
                    for c in range(7):
                        W[c_col*4 + c_row][col + c] = sw[t][c]
                col += 7
    
    return U, V, W


def _build_4x4x2():
    """4x4x2: A is 4x4, B is 4x2, C is 4x2.
    Split B into two 4x1 columns, or use 2x2x2 blocks.
    C[i,j] = sum_k A[i,k]*B[k,j]
    Split into 2x2x2 blocks:
    A is 4x4 -> 2x2 blocks: A11,A12 (top row of blocks), A21,A22 (bottom)
    B is 4x2 -> 2x1 blocks... this doesn't tile nicely with 2x2x2.
    
    Alternative: split into (4x2x2) + (4x2x2) using the two columns of B separately?
    No, B is 4x2, so each column is 4x1.
    
    Let me think differently. 4x4x2:
    Split A into left 2 cols and right 2 cols: A = [A_L | A_R] where A_L is 4x2, A_R is 4x2.
    Split B into top 2 rows and bottom 2 rows: B = [B_T; B_B] where B_T is 2x2, B_B is 2x2.
    C = A_L * B_T + A_R * B_B, where each product is 4x2x2.
    
    Now 4x2x2: A is 4x2, B is 2x2, C is 4x2.
    Split A into top 2 rows and bottom 2 rows: A = [A_T; A_B], each 2x2.
    C = A_T * B + A_B * B, each 2x2x2.
    So 4x2x2 = 2 * (2x2x2), rank 14.
    And 4x4x2 = 2 * (4x2x2) = 4 * (2x2x2), rank 28.
    """
    su, sv, sw = _strassen_2x2x2()
    
    U = [[0]*28 for _ in range(16)]  # A is 4x4
    V = [[0]*28 for _ in range(8)]   # B is 4x2
    W = [[0]*28 for _ in range(8)]   # C is 4x2, W is p*n = 2*4 = 8
    
    col = 0
    # 4x4x2 = sum over (block_row, block_col) of 2x2x2
    # A blocks: (bi, bj) for bi in 0,1 (row), bj in 0,1 (col) -> 4 blocks of 2x2
    # B blocks: (bk, bl) for bk in 0,1 (row), bl in 0,1 (col) -> but B is 4x2, so 2 row blocks of 2x1?
    # No, let me restructure.
    
    # C[i,j] = sum_{k=0}^{3} A[i,k] * B[k,j]
    # Group k into two pairs: k in {0,1} and k in {2,3}
    # C[i,j] = sum_{k=0}^{1} A[i,k]*B[k,j] + sum_{k=2}^{3} A[i,k]*B[k,j]
    # First term: A[:, 0:2] (4x2) * B[0:2, :] (2x2) -> 4x2
    # Second term: A[:, 2:4] (4x2) * B[2:4, :] (2x2) -> 4x2
    
    # For 4x2x2 product: A_sub is 4x2, B_sub is 2x2, C_sub is 4x2
    # Split A_sub into top 2 rows and bottom 2 rows (each 2x2)
    # C_sub = A_top(2x2)*B_sub(2x2) + A_bot(2x2)*B_sub(2x2)
    # Each is 2x2x2 Strassen.
    
    for k_block in range(2):  # k_block 0: k=0,1; k_block 1: k=2,3
        for a_block in range(2):  # a_block 0: rows 0,1 of A_sub; a_block 1: rows 2,3
            # This is a 2x2x2 Strassen block
            for t in range(4):
                # A entry: row = a_block*2 + t//2, col = k_block*2 + t%2
                a_row = a_block*2 + t//2
                a_col = k_block*2 + t%2
                for c in range(7):
                    U[a_row*4 + a_col][col + c] = su[t][c]
                
                # B entry: row = k_block*2 + t//2, col = t%2
                b_row = k_block*2 + t//2
                b_col = t%2
                for c in range(7):
                    V[b_row*2 + b_col][col + c] = sv[t][c]
                
                # C entry: row = a_block*2 + t//2, col = t%2
                # W index: W[k*n + i] = W[col_of_C * 4 + row_of_C]
                c_row = a_block*2 + t//2
                c_col = t%2
                for c in range(7):
                    W[c_col*4 + c_row][col + c] = sw[t][c]
            col += 7
    
    return U, V, W


def _build_4x1x6():
    """4x1x6: A is 4x1, B is 1x6, C is 4x6.
    C[i,j] = A[i,0] * B[0,j]
    Schoolbook: rank = 4*1*6 = 24.
    """
    rank = 24
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(24)]
    col = 0
    for i in range(4):
        for j in range(6):
            U[i][col] = 1
            V[j][col] = 1
            W[j*4 + i][col] = 1
            col += 1
    return U, V, W


def _merge(U_list, V_list, W_list):
    """Merge multiple decompositions by concatenating columns."""
    total_rank = sum(len(U[0]) for U in U_list)
    n_m = len(U_list[0])
    m_p = len(V_list[0])
    p_n = len(W_list[0])
    
    U = [[0]*total_rank for _ in range(n_m)]
    V = [[0]*total_rank for _ in range(m_p)]
    W = [[0]*total_rank for _ in range(p_n)]
    
    col = 0
    for Ui, Vi, Wi in zip(U_list, V_list, W_list):
        r = len(Ui[0])
        for i in range(n_m):
            for c in range(r):
                U[i][col+c] = Ui[i][c]
        for i in range(m_p):
            for c in range(r):
                V[i][col+c] = Vi[i][c]
        for i in range(p_n):
            for c in range(r):
                W[i][col+c] = Wi[i][c]
        col += r
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    C[i,k] = sum_j A[i,j]*B[j,k] = sum_c U[i*m+j... wait.
    
    The tensor is T[i,j,k] = delta_{i,k}... no.
    The matrix multiplication tensor: T[i, j, k] where the trilinear form is
    sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[k,i] (with appropriate indices).
    
    Actually, the standard formulation:
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    So the tensor T has T[i,j,k] = 1 if we think of it as the coefficient.
    Wait, the tensor for matrix multiplication is T_{i,j,k} such that:
    C_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The bilinear map: given A (n x m) and B (m x p), produce C (n x p) where C_{i,k} = sum_j A_{i,j} B_{j,k}.
    
    As a tensor decomposition of rank r:
    C_{i,k} = sum_{c=1}^r (sum_j U_{i*m+j, c} A_{i,j}) (sum_k V_{j*p+k, c} B_{j,k}) ... 
    
    Hmm, let me re-read the problem. U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    
    The decomposition says:
    sum_c U[a, c] * V[b, c] * W[d, c] = T[a, b, d]
    where a indexes A (n*m entries), b indexes B (m*p entries), d indexes C (p*n entries).
    
    T[a, b, d] = 1 if a, b, d are consistent with matrix multiplication, 0 otherwise.
    Specifically: a = i*m + j, b = j'*p + k, d = k'*n + i'.
    T[a, b, d] = 1 if i' = i, j' = j, k' = k, else 0.
    
    So: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = 1 if i'=i, j'=j, k'=k, else 0.
    
    Let me verify this.
    """
    import numpy as np
    U_np = np.array(U, dtype=object)
    V_np = np.array(V, dtype=object)
    W_np = np.array(W, dtype=object)
    rank = U_np.shape[1]
    
    # Compute the decomposed tensor
    # T_decomp[i,j,j',k,k',i'] = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    # This is expensive for large rank. Let's use einsum.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    # T[i, j, j', k, k', i'] = sum_c U[i,j,c] * V[j',k,c] * W[k',i',c]
    # = np.einsum('ijc,jkc,kic->ijjkkci', U_reshaped, V_reshaped, W_reshaped)  # too many dims
    # Actually: T[i,j,j',k,k',i'] where i in 0..n-1, j in 0..m-1, j' in 0..m-1, k in 0..p-1, k' in 0..p-1, i' in 0..n-1
    # This is n*m*m*p*p*n which is huge. Let me just check a few entries.
    
    # Better: check that for all (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # And for all (i, j, j', k, k', i') with not (i'=i and j'=j and k'=k): sum = 0
    
    # Check the diagonal (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    
    # Check off-diagonal (should be 0) - sample a few
    # For efficiency, check all cases where exactly one index differs
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # i' != i
                for i2 in range(n):
                    if i2 == i:
                        continue
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i2][c]
                    if s != 0:
                        return False
                # j' != j
                for j2 in range(m):
                    if j2 == j:
                        continue
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j2*p+k][c] * W[k*n+i][c]
                    if s != 0:
                        return False
                # k' != k
                for k2 in range(p):
                    if k2 == k:
                        continue
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k2][c] * W[k2*n+i][c]
                    if s != 0:
                        return False
    
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _build_4x4x6():
    """4x4x6: A is 4x4, B is 4x6, C is 4x6.
    Split B into two 4x3 matrices: B = [B1 | B2]
    C = A*B1 + A*B2, each 4x4x3.
    
    4x4x3: split B into 4x2 + 4x1.
    A*B_{4x2}: 4x4x2, rank 28 (from above).
    A*B_{4x1}: 4x4x1, rank 16 (schoolbook).
    So 4x4x3 = 28 + 16 = 44.
    4x4x6 = 2 * 44 = 88.
    
    Alternatively, split B into three 4x2 matrices:
    4x4x6 = 3 * (4x4x2) = 3 * 28 = 84.
    """
    # Use 3 blocks of 4x4x2
    U_list = []
    V_list = []
    W_list = []
    for col_block in range(3):
        # 4x4x2: A is 4x4, B_sub is 4x2 (columns 2*col_block, 2*col_block+1), C_sub is 4x2
        su, sv, sw = _strassen_2x2x2()
        U_b = [[0]*28 for _ in range(16)]
        V_b = [[0]*28 for _ in range(8)]  # B is 4x2 for this sub-problem
        W_b = [[0]*28 for _ in range(8)]  # C is 4x2, W is 2*4=8
        
        col = 0
        for k_block in range(2):
            for a_block in range(2):
                for t in range(4):
                    a_row = a_block*2 + t//2
                    a_col = k_block*2 + t%2
                    for c in range(7):
                        U_b[a_row*4 + a_col][col + c] = su[t][c]
                    
                    b_row = k_block*2 + t//2
                    b_col = t%2
                    for c in range(7):
                        V_b[b_row*2 + b_col][col + c] = sv[t][c]
                    
                    c_row = a_block*2 + t//2
                    c_col = t%2
                    for c in range(7):
                        W_b[c_col*4 + c_row][col + c] = sw[t][c]
                col += 7
        
        # Now map V_b (4x2 local) to global B (4x6) and W_b (4x2 local) to global C (4x6)
        V_global = [[0]*28 for _ in range(24)]  # B is 4x6
        W_global = [[0]*28 for _ in range(24)]  # C is 4x6, W is 6*4=24
        
        for r in range(4):
            for cc in range(2):
                V_global[r*6 + (2*col_block + cc)] = V_b[r*2 + cc]
        
        for r in range(4):
            for cc in range(2):
                W_global[(2*col_block + cc)*4 + r] = W_b[cc*4 + r]
        
        U_list.append(U_b)
        V_list.append(V_global)
        W_list.append(W_global)
    
    return _merge(U_list, V_list, W_list)


def _build_full_decomposition():
    """Build the full 4x5x6 decomposition.
    Split A (4x5) into A_top (4x4) and A_bot (4x1).
    Split B (5x6) into B_top (4x6) and B_bot (1x6).
    C = A_top * B_top + A_bot * B_bot, each 4x4x6 and 4x1x6.
    
    4x4x6: rank 84 (3 blocks of 4x4x2).
    4x1x6: rank 24.
    Total: 108.
    """
    U_top, V_top, W_top = _build_4x4x6()  # 16x84, 24x84, 24x84
    U_bot, V_bot, W_bot = _build_4x1x6()  # 4x24, 6x24, 24x24
    
    # Full U: 20 rows (4x5), V: 30 rows (5x6), W: 24 rows (6x4)
    total_rank = 84 + 24
    U = [[0]*total_rank for _ in range(20)]
    V = [[0]*total_rank for _ in range(30)]
    W = [[0]*total_rank for _ in range(24)]
    
    # Top part: A[0:4, 0:4] * B[0:4, 0:6] -> C[0:4, 0:6]
    for i in range(16):
        for c in range(84):
            U[i][c] = U_top[i][c]
    for i in range(24):
        for c in range(84):
            V[i][c] = V_top[i][c]
    for i in range(24):
        for c in range(84):
            W[i][c] = W_top[i][c]
    
    # Bottom part: A[0:4, 4] * B[4, 0:6] -> C[0:4, 0:6]
    # A[0:4, 4] is 4x1, indexed as U rows 16..19 (i*5+4 for i in 0..3)
    # B[4, 0:6] is 1x6, indexed as V rows 24..29 (4*6+j for j in 0..5)
    # C[0:4, 0:6] is 4x6, W indexed as k*4+i for k in 0..5, i in 0..3
    for i in range(4):
        for c in range(24):
            U[16 + i][84 + c] = U_bot[i][c]
    for j in range(6):
        for c in range(24):
            V[24 + j][84 + c] = V_bot[j][c]
    for k in range(6):
        for i in range(4):
            for c in range(24):
                W[k*4 + i][84 + c] = W_bot[k*4 + i][c]
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    try:
        U, V, W = _build_full_decomposition()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback: schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
