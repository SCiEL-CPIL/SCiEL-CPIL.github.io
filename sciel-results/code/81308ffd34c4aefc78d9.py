"""Strassen-based hierarchical decomposition for 4x5x6 matrix multiplication."""
import time
import sys
import random

DIMENSIONS = [4, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0


def remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W, "rank": rank}


def _verify(U, V, W, rank):
    import random
    n, m, p = DIMENSIONS
    for _ in range(5):
        if remaining() <= 0:
            return False
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                acc = 0
                for j in range(m):
                    for c in range(rank):
                        u = U[i * m + j][c]
                        if u == 0:
                            continue
                        acc += u * W[k * n + i][c] * V[j * p + k][c]
                if acc != C[i][k]:
                    return False
    return True


def _strassen_2x2():
    """Return 7 rank-1 terms for 2x2x2 matmul.
    Each term: (u_vec, v_vec, w_vec) as flat length-4 vectors.
    A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    Standard Strassen:
      M1 = (a+d)(e+h)
      M2 = (c+d)e
      M3 = a(f-h)
      M4 = d(g-e)
      M5 = (a+b)g
      M6 = (c-a)(e+f)
      M7 = (b-d)(h+f)
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M4 + M5 + M7
    C10 = M2 + M4 - M6 + M7
    C11 = M1 - M2 + M6 - M7
    """
    terms = []
    # M1 = (a+d)(e+h)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))  # +M1: C00,C11
    # M2 = (c+d)e
    terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))  # +M2: C10, -M2: C11
    # M3 = a(f-h)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))  # +M3: C01
    # M4 = d(g-e)
    terms.append(([0,0,0,1], [-1,0,1,0], [-1,1,1,0]))  # -M4:C00, +M4:C01,C10
    # M5 = (a+b)g
    terms.append(([1,1,0,0], [0,0,1,0], [1,1,0,0]))  # +M5: C00,C01
    # M6 = (c-a)(e+f)
    terms.append([[-1,0,1,0], [1,1,0,0], [0,0,-1,1]])  # -M6:C10, +M6:C11
    # M7 = (b-d)(h+f)
    terms.append(([0,1,0,-1], [0,1,0,1], [-1,1,1,0]))  # -M7:C00, +M7:C01,C10
    return terms


def _strassen_4x4():
    """Build 4x4x4 matmul using Strassen recursion. Returns list of (u,v,w) flat vectors."""
    # Split 4x4 into 2x2 blocks. Use Strassen at top level (7 sub-problems of 2x2x2).
    # A = [[A11,A12],[A21,A22]], B = [[B11,B12],[B21,B22]]
    # M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B22-B12)
    # M4=A22(B21-B11), M5=(A11+A12)B21, M6=(A21-A11)(B11+B12), M7=(A12-A22)(B22+B21)
    # C11=M1-M4+M5-M7, C12=M3+M4+M5+M7, C21=M2+M4-M6+M7, C22=M1-M2+M6-M7
    # Each Mi is a 2x2x2 matmul, which we do with rank 7.
    # Total rank: 7 * 7 = 49
    
    base = _strassen_2x2()
    n = 4
    terms_U = []
    terms_V = []
    terms_W = []
    
    # For each of the 7 Strassen products, we need a 2x2x2 decomposition
    # The 7 Strassen products are:
    # P1 = (A11+A22)(B11+B22) -> A part: rows 0-1,2-3 all cols 0-1,2-3 with +1; B part similar
    # P2 = (A21+A22)B11 -> A: rows 2-3, cols 0-1 and rows 2-3, cols 2-3; B: rows 0-1, cols 0-1
    # P3 = A11(B22-B12) -> A: rows 0-1, cols 0-1; B: rows 2-3, cols 2-3 minus rows 0-1, cols 2-3
    # P4 = A22(B21-B11) -> A: rows 2-3, cols 2-3; B: rows 2-3, cols 0-1 minus rows 0-1, cols 0-1
    # P5 = (A11+A12)B21 -> A: rows 0-1, cols 0-1 and rows 0-1, cols 2-3; B: rows 2-3, cols 0-1
    # P6 = (A21-A11)(B11+B12) -> A: rows 2-3, cols 0-1 minus rows 0-1, cols 0-1; B: rows 0-1, cols 0-1 and rows 0-1, cols 2-3
    # P7 = (A12-A22)(B22+B21) -> A: rows 0-1, cols 2-3 minus rows 2-3, cols 2-3; B: rows 2-3, cols 2-3 and rows 2-3, cols 0-1
    
    # Define the 7 Strassen combinations as (A_mask, B_mask, C_mask)
    # where each mask is a 2x2x2x2 tensor (block row, block col, block row, block col) with values
    # Actually, let's think of it differently.
    
    # For Strassen, we have 7 intermediate products. Each product involves:
    # - A linear combination of 2x2 blocks of A
    # - B linear combination of 2x2 blocks of B
    # - Result contributes to 2x2 blocks of C with signs
    
    # Let's define for each of the 7 products:
    # a_blocks: list of (block_row, block_col, coeff) for A
    # b_blocks: list of (block_row, block_col, coeff) for B  
    # c_blocks: list of (block_row, block_col, coeff) for C contribution
    
    strassen_defs = [
        # P1 = (A11+A22)(B11+B22)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # P2 = (A21+A22)B11
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # P3 = A11(B22-B12)
        ([(0,0,1)], [(1,1,1),(0,1,-1)], [(0,1,1)]),
        # P4 = A22(B21-B11)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,-1),(0,1,1),(1,0,1)]),
        # P5 = (A11+A12)B21
        ([(0,0,1),(0,1,1)], [(1,0,1)], [(0,0,1),(0,1,1)]),
        # P6 = (A21-A11)(B11+B12)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,-1),(1,1,1)]),
        # P7 = (A12-A22)(B22+B21)
        ([(0,1,1),(1,1,-1)], [(1,1,1),(1,0,1)], [(0,0,-1),(0,1,1),(1,0,1)]),
    ]
    
    for a_blocks, b_blocks, c_blocks in strassen_defs:
        # For each of the 7 base Strassen terms within this 2x2x2 sub-problem
        for (u_base, v_base, w_base) in base:
            u_full = [0] * (n * n)
            v_full = [0] * (n * n)
            w_full = [0] * (n * n)
            
            # Apply A block transformations
            for (br, bc, coeff) in a_blocks:
                for i in range(2):
                    for j in range(2):
                        u_idx = (br * 2 + i) * n + (bc * 2 + j)
                        u_full[u_idx] += coeff * u_base[i * 2 + j]
            
            # Apply B block transformations
            for (br, bc, coeff) in b_blocks:
                for i in range(2):
                    for j in range(2):
                        v_idx = (br * 2 + i) * n + (bc * 2 + j)
                        v_full[v_idx] += coeff * v_base[i * 2 + j]
            
            # Apply C block transformations
            for (br, bc, coeff) in c_blocks:
                for i in range(2):
                    for j in range(2):
                        w_idx = (br * 2 + i) * n + (bc * 2 + j)
                        w_full[w_idx] += coeff * w_base[i * 2 + j]
            
            terms_U.append(u_full)
            terms_V.append(v_full)
            terms_W.append(w_full)
    
    return list(zip(terms_U, terms_V, terms_W))


def _strassen_4x4x2():
    """Build 4x4x2 matmul. Split into two 4x2x1 and use efficient decomposition.
    Actually, let's use a block approach: 4x4x2 = process 4x4 with p=2.
    We can view this as multiplying 4x4 A by 4x2 B.
    Split A into 2x2 blocks (4 blocks), B into 2x1 blocks (2 blocks).
    Use Strassen-like approach.
    
    Alternative: just use schoolbook for the 4x4x2 part = 32 terms, 
    but we can do better.
    
    Let's use: split into two 2x4x2 sub-problems (top half and bottom half of A).
    Each 2x4x2 can be done with rank... let's think.
    
    Actually, simplest efficient approach: 4x4x2 using Strassen on the 4x4 part
    but only 2 columns of B. 
    
    Let me just use a direct approach: treat as 4x4 matrix mult but B has only 2 cols.
    Rank = 4*4*2 = 32 schoolbook. Can we do better?
    
    For now, use schoolbook for 4x4x2 = 32 terms.
    """
    n, m, p = 4, 4, 2
    terms_U = []
    terms_V = []
    terms_W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                terms_U.append(u)
                terms_V.append(v)
                terms_W.append(w)
    return list(zip(terms_U, terms_V, terms_W))


def _build_4x5x6():
    """Build 4x5x6 decomposition using Strassen blocks.
    
    Strategy:
    - Split 4x5x6 into blocks
    - Use Strassen 4x4x4 for the 4x4x4 part (rank 49)
    - Handle remainders: j=4 (4x1x6) and k=4,5 (4x4x2)
    
    4x5x6 = 4x4x4 + 4x1x6 + 4x4x2 (but need to be careful about overlap)
    
    Actually: A is 4x5, B is 5x6.
    Split A = [A_left (4x4) | A_right (4x1)]
    Split B = [[B_top (5x4); B_bottom (5x2)]]  -- no, B is 5x6
    
    Better: 
    A = [A1 (4x4) | A2 (4x1)]
    B = [B1 (4x6); B2 (1x6)]  -- B is 5x6, so B1 is 4x6, B2 is 1x6
    
    C = A1*B1 + A2*B2
    
    A1*B1: 4x4 times 4x6 = 4x6. Use Strassen for 4x4x4 part (cols 0-3) + 4x4x2 (cols 4-5)
    A2*B2: 4x1 times 1x6 = 4x6. This is rank 4*1*6 = 24 schoolbook, or we can do better.
    
    4x1x6: For each i,k, C[i,k] = A[i,4]*B[4,k]. Rank = 4*6 = 24 (each term is one i,k pair).
    Actually this is just outer product structure. Rank = 24.
    
    Total: 49 (4x4x4) + 32 (4x4x2) + 24 (4x1x6) = 105
    
    But wait, can we do better on 4x4x2? Let me think...
    4x4x2: We're computing C = A*B where A is 4x4, B is 4x2.
    This is equivalent to computing two 4x4x1 products.
    Rank of 4x4x1 is at least 4 (trivially), and schoolbook is 16.
    
    Hmm, let me just go with the straightforward approach for now.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    all_terms_U = []
    all_terms_V = []
    all_terms_W = []
    
    # Part 1: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4] (4x4x4)
    s44 = _strassen_4x4()
    for (u, v, w) in s44:
        u_full = [0] * (n * m)
        v_full = [0] * (m * p)
        w_full = [0] * (p * n)
        for idx in range(16):
            i = idx // 4
            j = idx % 4
            u_full[i * m + j] = u[idx]
            v_full[j * p + idx] = v[idx]  # j in 0..3, k in 0..3
        for idx in range(16):
            i = idx // 4
            k = idx % 4
            w_full[k * n + i] = w[idx]
        all_terms_U.append(u_full)
        all_terms_V.append(v_full)
        all_terms_W.append(w_full)
    
    # Part 2: A[0:4, 0:4] * B[0:4, 4:6] -> C[0:4, 4:6] (4x4x2)
    s442 = _strassen_4x4x2()
    for (u, v, w) in s442:
        u_full = [0] * (n * m)
        v_full = [0] * (m * p)
        w_full = [0] * (p * n)
        for idx in range(16):
            i = idx // 4
            j = idx % 4
            u_full[i * m + j] = u[idx]
        for idx in range(8):
            j = idx // 2
            k = idx % 2
            v_full[j * p + 4 + k] = v[idx]  # j in 0..3, k in 4..5
        for idx in range(8):
            i = idx // 2
            k = idx % 2
            w_full[(4 + k) * n + i] = w[idx]  # k in 4..5, i in 0..3
        all_terms_U.append(u_full)
        all_terms_V.append(v_full)
        all_terms_W.append(w_full)
    
    # Part 3: A[0:4, 4] * B[4, 0:6] -> C[0:4, 0:6] (4x1x6)
    # C[i,k] += A[i,4] * B[4,k]
    # Rank-1 terms: for each (i,k), term with U[i*5+4]=1, V[4*6+k]=1, W[k*4+i]=1
    for i in range(n):
        for k in range(p):
            u_full = [0] * (n * m)
            v_full = [0] * (m * p)
            w_full = [0] * (p * n)
            u_full[i * m + 4] = 1  # A[i,4]
            v_full[4 * p + k] = 1  # B[4,k]
            w_full[k * n + i] = 1  # C[i,k]
            all_terms_U.append(u_full)
            all_terms_V.append(v_full)
            all_terms_W.append(w_full)
    
    rank = len(all_terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(zip(all_terms_U, all_terms_V, all_terms_W)):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    return {"U": U, "V": V, "W": W, "rank": rank}


def _build_4x5x6_v2():
    """Alternative: use 2x2 Strassen blocks more aggressively.
    
    Split 4x5x6 into 2x2x2 blocks where possible.
    i: 0-3 (2 blocks of 2), j: 0-4 (2 blocks of 2 + 1), k: 0-5 (3 blocks of 2)
    
    Full 2x2x2 blocks: 2*2*3 = 12 blocks, each rank 7 = 84
    Remainder: j=4 column: 4*1*6 = 24 terms
    Total: 84 + 24 = 108
    
    Or: j: 0-4 as (2,2,1), k: 0-5 as (2,2,2)
    Blocks: 2 * 3 * 3 = 18 blocks of 2x2x2 = 126, plus 4*1*6=24... no that's worse.
    
    Let me try: i:(2,2), j:(2,2,1), k:(2,2,2)
    2x2x2 blocks: 2*2*3 = 12, rank 7 each = 84
    2x1x2 blocks: 2*1*3 = 6, rank 2 each = 12 (schoolbook for 2x1x2)
    Total: 84 + 12 = 96
    
    Wait, 2x1x2: A is 2x1, B is 1x2, C is 2x2.
    C[i,k] = A[i,0]*B[0,k]. Rank = 2*1*2 = 4 schoolbook, but actually rank 2 since
    it's an outer product of a 2-vector and a 2-vector... no, it's 4 independent entries.
    Actually C = a * b^T where a is 2x1 and b is 1x2, so rank 1!
    No wait: C[i,k] = A[i,0] * B[0,k]. This is outer product of column A[:,0] and row B[0,:].
    So rank 1! Each 2x1x2 block has rank 1.
    
    So: 12 * 7 + 6 * 1 = 84 + 6 = 90
    
    Let me verify this decomposition carefully.
    """
    n, m, p = DIMENSIONS  # 4, 5, 6
    all_terms_U = []
    all_terms_V = []
    all_terms_W = []
    
    base = _strassen_2x2()
    
    # Block decomposition: i: (0,1),(2,3); j: (0,1),(2,3),(4); k: (0,1),(2,3),(4,5)
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    for i_start, i_size in i_blocks:
        for j_start, j_size in j_blocks:
            for k_start, k_size in k_blocks:
                if i_size == 2 and j_size == 2 and k_size == 2:
                    # 2x2x2 block, use Strassen rank 7
                    for (u, v, w) in base:
                        u_full = [0] * (n * m)
                        v_full = [0] * (m * p)
                        w_full = [0] * (p * n)
                        for i in range(2):
                            for j in range(2):
                                u_full[(i_start + i) * m + (j_start + j)] = u[i * 2 + j]
                        for j in range(2):
                            for k in range(2):
                                v_full[(j_start + j) * p + (k_start + k)] = v[j * 2 + k]
                        for k in range(2):
                            for i in range(2):
                                w_full[(k_start + k) * n + (i_start + i)] = w[k * 2 + i]
                        all_terms_U.append(u_full)
                        all_terms_V.append(v_full)
                        all_terms_W.append(w_full)
                elif i_size == 2 and j_size == 1 and k_size == 2:
                    # 2x1x2 block, rank 1 (outer product)
                    u_full = [0] * (n * m)
                    v_full = [0] * (m * p)
                    w_full = [0] * (p * n)
                    for i in range(2):
                        u_full[(i_start + i) * m + j_start] = 1
                    for k in range(2):
                        v_full[j_start * p + (k_start + k)] = 1
                    for k in range(2):
                        for i in range(2):
                            w_full[(k_start + k) * n + (i_start + i)] = 1
                    all_terms_U.append(u_full)
                    all_terms_V.append(v_full)
                    all_terms_W.append(w_full)
                else:
                    # Should not happen with our block sizes
                    raise ValueError(f"Unexpected block sizes: {i_size}x{j_size}x{k_size}")
    
    rank = len(all_terms_U)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, (u, v, w) in enumerate(zip(all_terms_U, all_terms_V, all_terms_W)):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    return {"U": U, "V": V, "W": W, "rank": rank}


def _build_4x5x6_v3():
    """Try yet another blocking: i:(2,2), j:(2,2,1), k:(2,2,2) but handle
    the 2x1x2 blocks more carefully, and see if we can merge some terms.
    
    Actually, let's try: i:(2,2), j:(2,2,1), k:(2,2,2)
    2x2x2: 2*2*3=12 blocks * rank 7 = 84
    2x1x2: 2*1*3=6 blocks * rank 1 = 6
    Total: 90
    
    But can we do even better? What if we group the j=4 column differently?
    
    Alternative: i:(2,2), j:(2,2,1), k:(2,2,2) is what we have.
    
    Let's try i:(4), j:(2,2,1), k:(2,2,2) -- no, that doesn't help.
    
    What about using 4x4x4 Strassen for part of it?
    i:(4), j:(4), k:(4) -> 4x4x4 rank 49
    i:(4), j:(1), k:(4) -> 4x1x4 rank 16
    i:(4), j:(4), k:(2) -> 4x4x2 rank 32 (or better?)
    i:(4), j:(1), k:(2) -> 4x1x2 rank 8
    Total: 49+16+32+8 = 105
    
    The 2x2x2 approach gives 90, which is better.
    
    Can we improve the 2x1x2 blocks? They're already rank 1, which is minimal.
    
    What if we use a different 2x2x2 decomposition? The minimum rank for 2x2x2
    matrix multiplication is 7 (over most fields), so we can't improve that.
    
    Let's stick with v2 which gives rank 90.
    """
    return _build_4x5x6_v2()


def _iterative_refinement(U, V, W, rank):
    """Try to remove columns that are linear combinations or zero.
    Also try to find if any column can be dropped by checking if the tensor
    still decomposes without it.
    """
    n, m, p = DIMENSIONS
    # First, remove zero columns
    keep = []
    for c in range(rank):
        col_u = [U[r][c] for r in range(n * m)]
        col_v = [V[r][c] for r in range(m * p)]
        col_w = [W[r][c] for r in range(p * n)]
        if all(x == 0 for x in col_u + col_v + col_w):
            continue
        keep.append(c)
    
    if len(keep) == rank:
        return U, V, W, rank
    
    # Rebuild with only kept columns
    new_rank = len(keep)
    new_U = [[U[r][c] for c in keep] for r in range(n * m)]
    new_V = [[V[r][c] for c in keep] for r in range(m * p)]
    new_W = [[W[r][c] for c in keep] for r in range(p * n)]
    
    return new_U, new_V, new_W, new_rank


def construct():
    n, m, p = DIMENSIONS
    best = None
    best_rank = float('inf')
    
    # Try the 2x2x2 block decomposition (expected rank 90)
    try:
        if remaining() > 10:
            cand = _build_4x5x6_v2()
            if remaining() > 5:
                U, V, W, rank = _iterative_refinement(cand["U"], cand["V"], cand["W"], cand["rank"])
                if _verify(U, V, W, rank):
                    best = {"U": U, "V": V, "W": W, "rank": rank}
                    best_rank = rank
    except Exception as e:
        pass
    
    # Try the Strassen 4x4 based decomposition (expected rank 105)
    if best_rank > 95 and remaining() > 10:
        try:
            cand = _build_4x5x6()
            if remaining() > 5:
                U, V, W, rank = _iterative_refinement(cand["U"], cand["V"], cand["W"], cand["rank"])
                if _verify(U, V, W, rank) and rank < best_rank:
                    best = {"U": U, "V": V, "W": W, "rank": rank}
                    best_rank = rank
        except Exception:
            pass
    
    # Fallback to schoolbook (rank 120)
    if best is None:
        best = _schoolbook()
    
    del best["rank"]
    return best


if __name__ == "__main__":
    r = construct()
    print("U shape:", len(r["U"]), "x", len(r["U"][0]))
    print("rank:", len(r["U"][0]))
