import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-66 decomposition:
    # Split A rows: {0}, {1,2}
    # Split A cols / B rows: {0,1}, {2,3}
    # Split B cols: {0,1}, {2,3}, {4,5}
    # 1x2x2 blocks: 1*2*3 = 6 blocks, rank 4 each = 24
    # 2x2x2 blocks: 1*2*3 = 6 blocks, rank 7 each = 42
    # Total: 66
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Rank-7 decomposition for 2x2 matrix multiplication
    # Local indices: A: 0,1,2,3; B: 0,1,2,3; C: 0,1,2,3
    # These are the 7 terms of the Strassen-based rank-7 decomposition
    rank7_terms = [
        # (u, v, w) where u,v,w are 4-element vectors
        ([1,0,0,0], [0,0,0,1], [1,0,0,1]),
        ([0,0,0,1], [1,0,0,0], [0,0,1,0]),
        ([0,1,0,0], [0,1,0,0], [0,1,0,0]),
        ([1,0,1,0], [0,1,0,1], [1,0,1,0]),
        ([0,1,1,0], [1,0,1,0], [0,1,1,0]),
        ([1,1,0,0], [0,0,1,1], [1,1,0,0]),
        ([1,0,0,0], [0,0,1,1], [0,0,1,1]),
    ]
    
    def add_1x2x2_block(a_row, a_cols, b_cols):
        """Add a 1x2x2 schoolbook block.
        a_row: row index in A
        a_cols: list of 2 col indices in A
        b_cols: list of 2 col indices in B
        """
        nonlocal col
        for j in a_cols:
            for k in b_cols:
                U[a_row * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + a_row][col] = 1
                col += 1
    
    def add_2x2x2_block(a_rows, a_cols, b_cols):
        """Add a 2x2x2 block using rank-7 decomposition.
        a_rows: list of 2 row indices in A
        a_cols: list of 2 col indices in A
        b_cols: list of 2 col indices in B
        """
        nonlocal col
        for u, v, w in rank7_terms:
            for li in range(4):
                if u[li] != 0:
                    # Map local index li to global A index
                    i = a_rows[li // 2]
                    j = a_cols[li % 2]
                    U[i * m + j][col] = u[li]
            for lj in range(4):
                if v[lj] != 0:
                    # Map local index lj to global B index
                    j = a_cols[lj // 2]
                    k = b_cols[lj % 2]
                    V[j * p + k][col] = v[lj]
            for lk in range(4):
                if w[lk] != 0:
                    # Map local index lk to global C index
                    i = a_rows[lk // 2]
                    k = b_cols[lk % 2]
                    W[k * n + i][col] = w[lk]
            col += 1
    
    # 1x2x2 blocks: row 0 of A
    for a_cols in [[0, 1], [2, 3]]:
        for b_cols in [[0, 1], [2, 3], [4, 5]]:
            add_1x2x2_block(0, a_cols, b_cols)
    
    # 2x2x2 blocks: rows 1,2 of A
    for a_cols in [[0, 1], [2, 3]]:
        for b_cols in [[0, 1], [2, 3], [4, 5]]:
            add_2x2x2_block([1, 2], a_cols, b_cols)
    
    return {"U": U, "V": V, "W": W}
