import time
import numpy as np

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition:
    # 2 * Strassen (2x2x2) + 1 * Schoolbook (2x2x1)
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a block decomposition
    # u_offsets, v_offsets, w_offsets are the starting indices in the global matrices
    # for A, B, C respectively.
    def add_strassen(u_off, v_off, w_off, col_start):
        # Strassen coefficients for 2x2x2
        # A indices: 0,1,2,3 (i*m+j)
        # B indices: 0,1,2,3 (j*p+k) -> local 0,1,2,3
        # C indices: 0,1,2,3 (k*n+i) -> local 0,1,2,3
        
        u_vecs = [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, -1, 0],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        w_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, -1],
            [0, 0, 0, 1],
            [0, 1, 0, 0]
        ]
        
        for s in range(7):
            c = col_start + s
            for i in range(4):
                U[u_off + i][c] += u_vecs[s][i]
                V[v_off + i][c] += v_vecs[s][i]
                W[w_off + i][c] += w_vecs[s][i]

    def add_schoolbook_2x2x1(u_off, v_off, w_off, col_start):
        # Schoolbook for 2x2x1
        # A: 0,1,2,3
        # B: 0,1 (j*p+k, k=0) -> local 0,1
        # C: 0,1 (k*n+i, k=0) -> local 0,1
        
        # M1: A00 * B0 -> C0
        # M2: A01 * B1 -> C0
        # M3: A10 * B0 -> C1
        # M4: A11 * B1 -> C1
        
        u_vecs = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        v_vecs = [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1]
        ]
        w_vecs = [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1]
        ]
        
        for s in range(4):
            c = col_start + s
            for i in range(4):
                U[u_off + i][c] += u_vecs[s][i]
                # V local size 2
                for j in range(2):
                    V[v_off + j][c] += v_vecs[s][j]
                # W local size 2
                for k in range(2):
                    W[w_off + k][c] += w_vecs[s][k]

    # Block 1: Strassen for B cols 0,1 and C rows 0,1
    # A: 0-3
    # B: 0-3 (j*5+k, j=0,1, k=0,1 -> 0,1,5,6)
    #    Local 0,1,2,3 map to Global 0,1,5,6
    #    Wait, my add_strassen assumes contiguous local indices.
    #    I need to map local indices to global indices explicitly.
    
    # Let's rewrite the addition to handle non-contiguous global indices.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_block(u_map, v_map, w_map, u_vecs, v_vecs, w_vecs, col_start):
        for s in range(len(u_vecs)):
            c = col_start + s
            for local_i, global_i in u_map:
                U[global_i][c] += u_vecs[s][local_i]
            for local_j, global_j in v_map:
                V[global_j][c] += v_vecs[s][local_j]
            for local_k, global_k in w_map:
                W[global_k][c] += w_vecs[s][local_k]

    # Strassen vectors
    s_u = [
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 1, -1, 0],
        [0, 1, 0, -1]
    ]
    s_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    s_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, -1],
        [0, 0, 0, 1],
        [0, 1, 0, 0]
    ]
    
    # Schoolbook 2x2x1 vectors
    sb_u = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    sb_v = [
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1]
    ]
    sb_w = [
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ]

    # Block 1: Strassen
    # A: 0,1,2,3
    # B: 0,1,5,6 (local 0,1,2,3)
    # C: 0,1,2,3 (local 0,1,2,3)
    u_map1 = [(i, i) for i in range(4)]
    v_map1 = [(0, 0), (1, 1), (2, 5), (3, 6)]
    w_map1 = [(i, i) for i in range(4)]
    add_block(u_map1, v_map1, w_map1, s_u, s_v, s_w, 0)
    
    # Block 2: Strassen
    # A: 0,1,2,3
    # B: 2,3,7,8 (local 0,1,2,3)
    # C: 4,5,6,7 (local 0,1,2,3)
    u_map2 = [(i, i) for i in range(4)]
    v_map2 = [(0, 2), (1, 3), (2, 7), (3, 8)]
    w_map2 = [(0, 4), (1, 5), (2, 6), (3, 7)]
    add_block(u_map2, v_map2, w_map2, s_u, s_v, s_w, 7)
    
    # Block 3: Schoolbook
    # A: 0,1,2,3
    # B: 4,9 (local 0,1)
    # C: 8,9 (local 0,1)
    u_map3 = [(i, i) for i in range(4)]
    v_map3 = [(0, 4), (1, 9)]
    w_map3 = [(0, 8), (1, 9)]
    add_block(u_map3, v_map3, w_map3, sb_u, sb_v, sb_w, 14)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
