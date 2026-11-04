"""
Matrix multiplication 5x5x5 tensor decomposition.

Approach: Block decomposition with 5 = 2 + 2 + 1 partition.
- Use Strassen's algorithm (rank 7) for the 2x2x2 blocks
- Use schoolbook (rank n*m*p) for all other blocks

Total rank: 117
"""

DIMENSIONS = [5, 5, 5]


def _schoolbook_factors(n, m, p):
    """Return U, V, W for schoolbook n x m x p matrix multiplication."""
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


def _strassen_factors():
    """
    Return U, V, W for Strassen's 2x2x2 algorithm.
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[a*(e+f) - (d-c)*(g+f), b*(e+f)],
         [(a+c)*h - (e+d)*(h-g), (a+c)*(g+h)]]
    
    Seven products:
    M1 = a*(e+f)
    M2 = (a+b)*h
    M3 = c*(g+h)
    M4 = d*(e+f)
    M5 = (a+d)*(e+h-f)
    M6 = (b+c)*(g+e-h)
    M7 = (a+c)*(g+h)
    
    C00 = M1 - M4 + M5
    C01 = M2 - M5 + M7
    C10 = M3 + M4 - M6
    C11 = M1 - M3 + M6
    """
    rank = 7
    # U: shape (n*m, rank) = (4, 7), indexed by [i*m+j]
    # A = [[a, b], [c, d]] -> indices: a=0, b=1, c=2, d=3
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # a: M1, M5
        [0, 1, 0, 0, 0, 1, 0],  # b: M2, M6
        [0, 0, 1, 0, 0, 1, 1],  # c: M3, M6, M7
        [0, 0, 0, 1, 1, 0, 0],  # d: M4, M5
    ]
    # V: shape (m*p, rank) = (4, 7), indexed by [j*p+k]
    # B = [[e, f], [g, h]] -> indices: e=0, f=1, g=2, h=3
    V = [
        [1, 0, 0, 1, 1, 1, 0],  # e: M1, M4, M5, M6
        [1, 0, 0, 1, -1, 0, 0], # f: M1, M4, M5
        [0, 0, 1, 0, 0, 0, 1],  # g: M3, M7
        [0, 1, 1, 0, 1, -1, 1], # h: M2, M3, M5, M6, M7
    ]
    # W: shape (p*n, rank) = (4, 7), indexed by [k*n+i]
    # C = [[C00, C01], [C10, C11]] -> indices: C00=0, C10=1, C01=2, C11=3
    W = [
        [1, 0, 0, -1, 1, 0, 0],   # C00 = M1 - M4 + M5
        [0, 1, 0, 0, -1, 0, 1],   # C01 = M2 - M5 + M7
        [0, 0, 1, 1, 0, -1, 0],   # C10 = M3 + M4 - M6
        [1, 0, -1, 0, 0, 1, 0],   # C11 = M1 - M3 + M6
    ]
    return U, V, W


def _block_decompose_5x5x5():
    """
    Decompose 5x5x5 matrix mult into blocks using 5 = 2 + 2 + 1.
    
    Partitions: [0,1], [2,3], [4]
    
    For each block (i_block, j_block, k_block):
    - If all dimensions are 2, use Strassen (rank 7)
    - Otherwise, use schoolbook (rank = ni*nj*nk)
    """
    n, m, p = 5, 5, 5
    partitions = [[0, 1], [2, 3], [4]]
    
    # Precompute all factor sets
    schoolbook_cache = {}
    strassen_U, strassen_V, strassen_W = _strassen_factors()
    
    def get_factors(ni, nj, nk):
        if ni == 2 and nj == 2 and nk == 2:
            return strassen_U, strassen_V, strassen_W
        key = (ni, nj, nk)
        if key not in schoolbook_cache:
            schoolbook_cache[key] = _schoolbook_factors(ni, nj, nk)
        return schoolbook_cache[key]
    
    # Initialize full factor matrices with zeros
    total_rank = 0
    # First pass: compute total rank
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                i_start = partitions[i_block][0]
                i_end = partitions[i_block][-1] + 1
                j_start = partitions[j_block][0]
                j_end = partitions[j_block][-1] + 1
                k_start = partitions[k_block][0]
                k_end = partitions[k_block][-1] + 1
                
                ni = i_end - i_start
                nj = j_end - j_start
                nk = k_end - k_start
                
                U_sub, V_sub, W_sub = get_factors(ni, nj, nk)
                total_rank += len(U_sub[0])
    
    # Initialize full factor matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Second pass: fill in factors
    col_offset = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                i_start = partitions[i_block][0]
                i_end = partitions[i_block][-1] + 1
                j_start = partitions[j_block][0]
                j_end = partitions[j_block][-1] + 1
                k_start = partitions[k_block][0]
                k_end = partitions[k_block][-1] + 1
                
                ni = i_end - i_start
                nj = j_end - j_start
                nk = k_end - k_start
                
                U_sub, V_sub, W_sub = get_factors(ni, nj, nk)
                rank_sub = len(U_sub[0])
                
                # Fill U
                for i_loc in range(ni):
                    for j_loc in range(nj):
                        u_idx_local = i_loc * nj + j_loc
                        u_idx_global = (i_start + i_loc) * m + (j_start + j_loc)
                        for r in range(rank_sub):
                            U[u_idx_global][col_offset + r] = U_sub[u_idx_local][r]
                
                # Fill V
                for j_loc in range(nj):
                    for k_loc in range(nk):
                        v_idx_local = j_loc * nk + k_loc
                        v_idx_global = (j_start + j_loc) * p + (k_start + k_loc)
                        for r in range(rank_sub):
                            V[v_idx_global][col_offset + r] = V_sub[v_idx_local][r]
                
                # Fill W
                for k_loc in range(nk):
                    for i_loc in range(ni):
                        w_idx_local = k_loc * ni + i_loc
                        w_idx_global = (k_start + k_loc) * n + (i_start + i_loc)
                        for r in range(rank_sub):
                            W[w_idx_global][col_offset + r] = W_sub[w_idx_local][r]
                
                col_offset += rank_sub
    
    return U, V, W


def construct():
    U, V, W = _block_decompose_5x5x5()
    return {"U": U, "V": V, "W": W}
