"""
Matrix multiplication 5x5x5 tensor decomposition.

Approach: Block decomposition with 5 = 2 + 2 + 1 partition.
- Use Strassen's algorithm (rank 7) for 2x2x2 blocks
- Use schoolbook (rank n*m*p) for all other blocks

Total rank: 
- 8 blocks of 2x2x2 use Strassen (rank 7 each) -> 56
- 19 blocks of mixed sizes use Schoolbook.
  Total schoolbook rank for full 5x5x5 is 125.
  The 8 2x2x2 blocks would have contributed 8*8=64 to the schoolbook sum.
  So the remaining 19 blocks contribute 125 - 64 = 61.
  Total rank = 56 + 61 = 117.
"""

DIMENSIONS = [5, 5, 5]


def _strassen_factors():
    """
    Return U, V, W for Strassen's 2x2x2 algorithm.
    """
    rank = 7
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # a: M1, M5
        [0, 1, 0, 0, 0, 1, 0],  # b: M2, M6
        [0, 0, 1, 0, 0, 1, 1],  # c: M3, M6, M7
        [0, 0, 0, 1, 1, 0, 0],  # d: M4, M5
    ]
    V = [
        [1, 0, 0, 1, 1, 1, 0],  # e: M1, M4, M5, M6
        [1, 0, 0, 1, -1, 0, 0], # f: M1, M4, M5
        [0, 0, 1, 0, 0, 0, 1],  # g: M3, M7
        [0, 1, 1, 0, 1, -1, 1], # h: M2, M3, M5, M6, M7
    ]
    W = [
        [1, 0, 0, -1, 1, 0, 0],   # C00 = M1 - M4 + M5
        [0, 1, 0, 0, -1, 0, 1],   # C01 = M2 - M5 + M7
        [0, 0, 1, 1, 0, -1, 0],   # C10 = M3 + M4 - M6
        [1, 0, -1, 0, 0, 1, 0],   # C11 = M1 - M3 + M6
    ]
    return U, V, W


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


def _block_decompose_5x5x5():
    """
    Decompose 5x5x5 matrix mult into blocks using 5 = 2 + 2 + 1.
    """
    n, m, p = 5, 5, 5
    # Block sizes
    sizes = [2, 2, 1]
    # Offsets for rows/cols
    # A rows, C rows: 0, 2, 4
    # A cols, B rows: 0, 2, 4
    # B cols, C cols: 0, 2, 4
    row_offsets = [0, 2, 4]
    col_offsets = [0, 2, 4]
    
    # Calculate total rank
    total_rank = 0
    block_ranks = {}
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = sizes[i]
                mj = sizes[j]
                pk = sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    r = 7
                else:
                    r = ni * mj * pk
                total_rank += r
                block_ranks[(i, j, k)] = r

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = sizes[i]
                mj = sizes[j]
                pk = sizes[k]
                
                # Get local factors
                if ni == 2 and mj == 2 and pk == 2:
                    u_loc, v_loc, w_loc = _strassen_factors()
                else:
                    u_loc, v_loc, w_loc = _schoolbook_factors(ni, mj, pk)
                
                local_rank = block_ranks[(i, j, k)]
                
                # Map local indices to global indices
                # U: local row u*mj+v -> global row (row_offsets[i]+u)*m + (col_offsets[j]+v)
                for lu in range(ni):
                    for lv in range(mj):
                        local_idx = lu * mj + lv
                        global_idx = (row_offsets[i] + lu) * m + (col_offsets[j] + lv)
                        for lr in range(local_rank):
                            U[global_idx][col_offset + lr] = u_loc[local_idx][lr]
                
                # V: local row v*pk+w -> global row (col_offsets[j]+v)*p + (col_offsets[k]+w)
                for lv in range(mj):
                    for lw in range(pk):
                        local_idx = lv * pk + lw
                        global_idx = (col_offsets[j] + lv) * p + (col_offsets[k] + lw)
                        for lr in range(local_rank):
                            V[global_idx][col_offset + lr] = v_loc[local_idx][lr]
                
                # W: local row w*ni+u -> global row (col_offsets[k]+w)*n + (row_offsets[i]+u)
                for lw in range(pk):
                    for lu in range(ni):
                        local_idx = lw * ni + lu
                        global_idx = (col_offsets[k] + lw) * n + (row_offsets[i] + lu)
                        for lr in range(local_rank):
                            W[global_idx][col_offset + lr] = w_loc[local_idx][lr]
                
                col_offset += local_rank

    return U, V, W


def construct():
    U, V, W = _block_decompose_5x5x5()
    return {"U": U, "V": V, "W": W}
