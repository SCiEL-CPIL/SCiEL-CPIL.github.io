"""Program using Laderman's 19-rank 4x4x4 decomposition + schoolbook for remaining 3 columns."""

# Laderman's 19 bilinear forms for 4x4 * 4x4
# Indexing: row r in 0..3, col c in 0..3
# Each bilinear form is (row_expr, col_expr, output_positions)
# row_expr: list of (coeff, (r, c)) for A
# col_expr: list of (coeff, (r, c)) for B
# output_positions: list of (coeff, (out_r, out_c)) for C

LADERMAN_19 = [
    # 1
    ([(1, (0,0))], [(1, (0,0))], [(1, (0,0))]),
    # 2
    ([(1, (1,0)), (1, (0,0))], [(1, (0,1))], [(1, (1,1))]),
    # 3
    ([(1, (1,0))], [(1, (0,0)), (-1, (0,1))], [(-1, (1,0))]),
    # 4
    ([(1, (0,0)), (-1, (1,0))], [(1, (1,0))], [(1, (0,1))]),
    # 5
    ([(1, (0,0)), (-1, (1,0))], [(1, (1,1))], [(1, (0,2))]),
    # 6
    ([(1, (0,0)), (1, (1,0))], [(1, (1,1)), (-1, (0,1))], [(1, (1,2))]),
    # 7
    ([(1, (0,1))], [(1, (0,1)), (1, (1,1))], [(1, (0,3))]),
    # 8
    ([(1, (0,1)), (1, (1,1))], [(1, (0,0)), (-1, (0,1))], [(1, (1,3))]),
    # 9
    ([(1, (0,0)), (-1, (1,0))], [(1, (0,1)), (1, (1,1))], [(-1, (0,0))]),
    # 10
    ([(1, (0,0))], [(1, (0,1)), (-1, (1,1))], [(1, (0,1))]),
    # 11
    ([(1, (1,0))], [(1, (0,0)), (1, (1,0))], [(1, (1,0))]),
    # 12
    ([(1, (1,0)), (-1, (0,0))], [(1, (1,1))], [(1, (1,1))]),
    # 13
    ([(1, (1,0))], [(1, (0,1))], [(-1, (0,0))]),
    # 14
    ([(1, (0,1))], [(1, (0,0)), (1, (1,0))], [(1, (0,1))]),
    # 15
    ([(1, (1,1))], [(1, (0,0)), (-1, (1,0))], [(1, (1,1))]),
    # 16
    ([(1, (1,1))], [(1, (0,1))], [(1, (0,2))]),
    # 17
    ([(1, (0,1)), (1, (1,1))], [(1, (0,0))], [(-1, (0,2))]),
    # 18
    ([(1, (0,1))], [(1, (1,1))], [(1, (1,2))]),
    # 19
    ([(1, (1,1)), (-1, (0,1))], [(1, (1,0))], [(-1, (1,3))]),
]

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    rank_laderman = 19
    rank_schoolbook = n * m * (p - 4)  # 4*4*3 = 48
    total_rank = rank_laderman + rank_schoolbook
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Laderman part for first 4 columns (indices 0..3)
    for form_idx, (row_expr, col_expr, out_expr) in enumerate(LADERMAN_19):
        col = form_idx
        for coeff, (r, c) in row_expr:
            U[r * m + c][col] += coeff
        for coeff, (r, c) in col_expr:
            V[r * p + c][col] += coeff
        for coeff, (out_r, out_c) in out_expr:
            W[out_c * n + out_r][col] += coeff
    
    # Schoolbook part for columns 4, 5, 6
    col_offset = rank_laderman
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col_offset] = 1
                V[j * p + k][col_offset] = 1
                W[k * n + i][col_offset] = 1
                col_offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
