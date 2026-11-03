"""Matrix multiplication 3x4x7 with reduced rank via block decomposition."""
import time

DIMENSIONS = [3, 4, 7]

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2.
    Returns U, V, W as lists of lists.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: U[row, col], V[row, col], W[row, col]
    Row indices: 0,1 for i=0; 2,3 for i=1 (U: i*m+j, m=2)
                0,1 for j=0; 2,3 for j=1 (V: j*p+k, p=2)
                0,1 for k=0; 2,3 for k=1 (W: k*n+i, n=2)
    """
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    # U matrix: rows correspond to A elements (i*m+j), columns to M1..M7
    # A00 is row 0, A01 is row 1, A10 is row 2, A11 is row 3
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1],  # A11
    ]

    # V matrix: rows correspond to B elements (j*p+k), columns to M1..M7
    # B00 is row 0, B01 is row 1, B10 is row 2, B11 is row 3
    V = [
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [0, 0, 0, 0, 0, 1, 0],   # B01
        [0, 0, 1, -1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1],  # B11
    ]

    # W matrix: rows correspond to C elements (k*n+i), columns to M1..M7
    # C00 is row 0, C01 is row 1, C10 is row 2, C11 is row 3
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0],  # C11
    ]

    return U, V, W

def _schoolbook_2x2x1():
    """
    Schoolbook decomposition for 2x2 * 2x1.
    Returns U, V, W as lists of lists.
    U: shape (4, 4), V: shape (2, 4), W: shape (2, 4)
    """
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # 4 terms
    U = [
        [1, 0, 0, 0],  # A00
        [0, 1, 0, 0],  # A01
        [0, 0, 1, 0],  # A10
        [0, 0, 0, 1],  # A11
    ]
    V = [
        [1, 0, 1, 0],  # B0
        [0, 1, 0, 1],  # B1
    ]
    W = [
        [1, 1, 0, 0],  # C0
        [0, 0, 1, 1],  # C1
    ]
    return U, V, W

def _schoolbook_1x4x7():
    """
    Schoolbook decomposition for 1x4 * 4x7.
    Returns U, V, W as lists of lists.
    U: shape (4, 28), V: shape (28, 28), W: shape (7, 28)
    """
    n, m, p = 1, 4, 7
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _concat_decompositions(decomps):
    """
    Concatenate multiple decompositions along the rank dimension.
    Each decomp is (U, V, W) with same row shapes.
    Returns combined (U, V, W).
    """
    if not decomps:
        return None
    U = []
    V = []
    W = []
    for u, v, w in decomps:
        for row in u:
            U.append(row[:])
        for row in v:
            V.append(row[:])
        for row in w:
            W.append(row[:])
    return U, V, W

def _embed_decomposition(sub_U, sub_V, sub_W, sub_n, sub_m, sub_p, 
                         offset_i, offset_j, offset_k, 
                         full_n, full_m, full_p):
    """
    Embed a sub-decomposition into a larger decomposition.
    The sub-decomposition is for (sub_n x sub_m) * (sub_m x sub_p) -> (sub_n x sub_p).
    It is placed at offset (offset_i, offset_j, offset_k) in the full problem.
    """
    sub_rank = len(sub_U[0])
    full_rank = full_n * full_m * full_p  # This is just for sizing; we'll use actual rank from concatenation
    
    # We need to create U, V, W with the correct full dimensions
    # U has shape (full_n * full_m, sub_rank)
    # V has shape (full_m * full_p, sub_rank)
    # W has shape (full_p * full_n, sub_rank)
    
    U = [[0] * sub_rank for _ in range(full_n * full_m)]
    V = [[0] * sub_rank for _ in range(full_m * full_p)]
    W = [[0] * sub_rank for _ in range(full_p * full_n)]
    
    # Map sub indices to full indices
    # Sub U row: i_sub * sub_m + j_sub -> Full U row: (i_sub + offset_i) * full_m + (j_sub + offset_j)
    # Sub V row: j_sub * sub_p + k_sub -> Full V row: (j_sub + offset_j) * full_p + (k_sub + offset_k)
    # Sub W row: k_sub * sub_n + i_sub -> Full W row: (k_sub + offset_k) * full_n + (i_sub + offset_i)
    
    for i_sub in range(sub_n):
        for j_sub in range(sub_m):
            full_row_u = (i_sub + offset_i) * full_m + (j_sub + offset_j)
            for col in range(sub_rank):
                U[full_row_u][col] = sub_U[i_sub * sub_m + j_sub][col]
    
    for j_sub in range(sub_m):
        for k_sub in range(sub_p):
            full_row_v = (j_sub + offset_j) * full_p + (k_sub + offset_k)
            for col in range(sub_rank):
                V[full_row_v][col] = sub_V[j_sub * sub_p + k_sub][col]
    
    for k_sub in range(sub_p):
        for i_sub in range(sub_n):
            full_row_w = (k_sub + offset_k) * full_n + (i_sub + offset_i)
            for col in range(sub_rank):
                W[full_row_w][col] = sub_W[k_sub * sub_n + i_sub][col]
    
    return U, V, W

def _combine_decompositions(decomps, full_n, full_m, full_p):
    """
    Combine multiple embedded decompositions by summing (concatenating along rank).
    Each decomp is (U, V, W) with shapes (full_n*full_m, r), (full_m*full_p, r), (full_p*full_n, r).
    """
    if not decomps:
        return None
    total_rank = sum(len(u[0]) for u, v, w in decomps)
    U = [[0] * total_rank for _ in range(full_n * full_m)]
    V = [[0] * total_rank for _ in range(full_m * full_p)]
    W = [[0] * total_rank for _ in range(full_p * full_n)]
    
    col_offset = 0
    for u, v, w in decomps:
        r = len(u[0])
        for i in range(full_n * full_m):
            for c in range(r):
                U[i][col_offset + c] = u[i][c]
        for i in range(full_m * full_p):
            for c in range(r):
                V[i][col_offset + c] = v[i][c]
        for i in range(full_p * full_n):
            for c in range(r):
                W[i][col_offset + c] = w[i][c]
        col_offset += r
    
    return U, V, W

def construct():
    """
    Returns a tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 blocks.
    """
    start_time = time.time()
    deadline = 145.0  # seconds, safety margin
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Build 2x4x7 decomposition
    # Split n=2 into two rows of 1, but we want to use Strassen for 2x2 blocks
    # Better: split the 2x4x7 problem into four 2x2x? blocks? No, m=4, p=7.
    # Let's split m=4 into 2+2, p=7 into 3+4? Or use different approach.
    
    # Approach: 
    # 1x4x7: schoolbook, rank 28
    # 2x4x7: decompose as four 2x2x? blocks won't work cleanly.
    
    # Alternative: Use the decomposition:
    # 3x4x7 = 1x4x7 + 2x4x7
    # 2x4x7: split m=4 into 2+2, so we have 2x2x7 and 2x2x7
    # Each 2x2x7: split p=7 into 2+2+2+1? Or 2+5?
    # 2x2x7: use Strassen for 2x2x2 (rank 7) three times + 2x2x1 (rank 4) = 7*3+4 = 25
    # So 2x4x7 = 25 + 25 = 50
    # Total: 28 + 50 = 78. That's not better than 84 by much.
    
    # Better: 2x2x7: split p=7 into 3+4
    # 2x2x3: Strassen 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11
    # 2x2x4: Strassen 2x2x2 (rank 7) + 2x2x2 (rank 7) = 14
    # 2x2x7: 11 + 14 = 25
    # Same result.
    
    # Let me try: 3x4x7
    # Split n=3 into 1+2
    # 1x4x7: rank 28
    # 2x4x7: split m=4 into 2+2
    #   2x2x7 + 2x2x7
    #   Each 2x2x7: split p=7 into 2+2+2+1
    #   2x2x2: rank 7 (Strassen)
    #   2x2x1: rank 4 (schoolbook)
    #   2x2x7: 7+7+7+4 = 25
    #   2x4x7: 25+25 = 50
    # Total: 28+50 = 78
    
    # Can we do better? 
    # What if we split 3x4x7 differently?
    # Split m=4 into 1+3?
    # 3x1x7: rank 21
    # 3x3x7: ?
    # 3x3x7: split n=3 into 1+2
    #   1x3x7: rank 21
    #   2x3x7: split m=3 into 1+2
    #     2x1x7: rank 14
    #     2x2x7: rank 25
    #   2x3x7: 14+25 = 39
    #   3x3x7: 21+39 = 60
    # Total: 21+60 = 81. Worse.
    
    # Let's go with 78.
    
    # Build components
    # 1x4x7 schoolbook
    U_1x4x7, V_1x4x7, W_1x4x7 = _schoolbook_1x4x7()
    
    # 2x2x2 Strassen
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # 2x2x1 schoolbook
    U_221, V_221, W_221 = _schoolbook_2x2x1()
    
    # Build 2x2x7 by concatenating three 2x2x2 and one 2x2x1
    # Each needs to be embedded into 2x2x7
    decomp_2x2x7 = []
    # First 2x2x2: offset_k=0
    decomp_2x2x7.append(_embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 0, 0, 0, 2, 2, 7))
    # Second 2x2x2: offset_k=2
    decomp_2x2x7.append(_embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 0, 0, 2, 2, 2, 7))
    # Third 2x2x2: offset_k=4
    decomp_2x2x7.append(_embed_decomposition(U_s, V_s, W_s, 2, 2, 2, 0, 0, 4, 2, 2, 7))
    # 2x2x1: offset_k=6
    decomp_2x2x7.append(_embed_decomposition(U_221, V_221, W_221, 2, 2, 1, 0, 0, 6, 2, 2, 7))
    
    U_227, V_227, W_227 = _combine_decompositions(decomp_2x2x7, 2, 2, 7)
    
    # Build 2x4x7 by combining two 2x2x7 blocks
    decomp_2x4x7 = []
    # First 2x2x7: offset_j=0
    decomp_2x4x7.append(_embed_decomposition(U_227, V_227, W_227, 2, 2, 7, 0, 0, 0, 2, 4, 7))
    # Second 2x2x7: offset_j=2
    decomp_2x4x7.append(_embed_decomposition(U_227, V_227, W_227, 2, 2, 7, 0, 2, 0, 2, 4, 7))
    
    U_247, V_247, W_247 = _combine_decompositions(decomp_2x4x7, 2, 4, 7)
    
    # Build 3x4x7 by combining 1x4x7 and 2x4x7
    decomp_3x4x7 = []
    # 1x4x7: offset_i=0
    decomp_3x4x7.append(_embed_decomposition(U_1x4x7, V_1x4x7, W_1x4x7, 1, 4, 7, 0, 0, 0, 3, 4, 7))
    # 2x4x7: offset_i=1
    decomp_3x4x7.append(_embed_decomposition(U_247, V_247, W_247, 2, 4, 7, 1, 0, 0, 3, 4, 7))
    
    U, V, W = _combine_decompositions(decomp_3x4x7, 3, 4, 7)
    
    rank = len(U[0])
    
    if time.time() - start_time > deadline:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
