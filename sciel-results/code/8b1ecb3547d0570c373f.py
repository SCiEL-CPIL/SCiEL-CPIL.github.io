import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 64 solution:
    # Two 3x3x3 blocks (Laderman, rank 23 each) and one 3x3x2 block (Schoolbook, rank 18).
    rank = 23 + 23 + 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman coefficients for 3x3x3
    # Format: (u, v, w) where u, v, w are lists of 9 integers.
    # Indices in u/v: i*3+j (row i, col j of A/B)
    # Indices in w: k*3+i (col k, row i of C) -> Wait, W[k*n+i, r]. n=3. So index is k*3+i.
    laderman_terms = [
        ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]),
        ([0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]),
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]),
        ([1,0,0,1,0,0,0,0,0], [1,0,0,1,0,0,0,0,0], [1,0,1,1,1,0,0,0,0]),
        ([0,0,0,1,0,0,0,0,1], [0,0,0,1,0,0,0,0,1], [0,0,0,0,1,1,1,1,0]),
        ([1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,1], [1,0,0,0,0,0,1,0,1]),
        ([0,1,0,0,1,0,0,0,0], [0,1,0,0,1,0,0,0,0], [0,1,0,1,0,0,0,0,0]),
        ([0,0,1,0,0,1,0,0,0], [0,0,1,0,0,1,0,0,0], [0,0,1,0,0,1,0,0,0]),
        ([0,1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0,0]),
        ([0,0,1,0,1,0,0,0,0], [0,0,1,0,1,0,0,0,0], [0,0,1,0,1,0,0,0,0]),
        ([0,0,0,0,1,0,1,0,0], [0,0,0,0,1,0,1,0,0], [0,0,0,0,1,0,0,0,1]),
        ([0,0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1,0]),
        ([0,0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1,0]),
        ([0,0,0,0,1,0,0,0,1], [0,0,0,0,1,0,0,0,1], [0,0,0,0,1,0,0,0,1]),
        ([0,0,0,0,0,1,1,0,0], [0,0,0,0,0,1,1,0,0], [0,0,0,0,0,1,1,0,0]),
        ([0,0,0,1,0,0,0,0,1], [0,0,0,1,0,0,0,0,1], [0,0,0,1,0,0,0,0,1]),
        ([0,0,0,0,1,0,1,0,0], [0,0,0,0,1,0,1,0,0], [0,0,0,0,1,0,1,0,0]),
        ([0,0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1,0]),
        ([0,1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0,0]),
        ([0,0,1,0,1,0,0,0,0], [0,0,1,0,1,0,0,0,0], [0,0,1,0,1,0,0,0,0]),
        ([0,1,0,0,1,0,0,0,0], [0,1,0,0,1,0,0,0,0], [0,1,0,0,1,0,0,0,0]),
        ([0,0,1,0,0,1,0,0,0], [0,0,1,0,0,1,0,0,0], [0,0,1,0,0,1,0,0,0]),
        ([0,0,0,1,0,0,0,0,1], [0,0,0,1,0,0,0,0,1], [0,0,0,1,0,0,0,0,1]),
    ]
    
    def add_block(u_offset, v_offset, w_offset, terms, rank_start):
        for idx, (u, v, w) in enumerate(terms):
            r = rank_start + idx
            for i in range(9):
                if u[i] != 0:
                    U[u_offset + i][r] += u[i]
                if v[i] != 0:
                    V[v_offset + i][r] += v[i]
                if w[i] != 0:
                    W[w_offset + i][r] += w[i]

    # Block 1: k=0,1,2 (3x3x3)
    # A indices: 0..8 (i*3+j)
    # B indices: 0..8 (j*3+k) -> For k=0,1,2, B indices are 0..8
    # C indices: 0..8 (k*3+i) -> For k=0,1,2, C indices are 0..8
    add_block(0, 0, 0, laderman_terms, 0)
    
    # Block 2: k=3,4,5 (3x3x3)
    # A indices: 0..8
    # B indices: 9..17 (j*3 + (3+k')) -> j=0,1,2. k'=0,1,2.
    # B index = j*3 + k. k in 3..5.
    # So B offset is 9? No.
    # B is 3x8. Index = j*8 + k.
    # For k=3,4,5:
    # j=0: 3,4,5
    # j=1: 11,12,13
    # j=2: 19,20,21
    # This is not contiguous in the flattened B matrix if we just shift.
    # We need to map the 3x3 B block to the correct positions in V.
    
    # Let's handle mapping explicitly.
    
    def add_block_mapped(u_map, v_map, w_map, terms, rank_start):
        for idx, (u, v, w) in enumerate(terms):
            r = rank_start + idx
            for i in range(9):
                if u[i] != 0:
                    U[u_map[i]][r] += u[i]
                if v[i] != 0:
                    V[v_map[i]][r] += v[i]
                if w[i] != 0:
                    W[w_map[i]][r] += w[i]

    # Map for Block 1 (k=0,1,2)
    u_map1 = list(range(9))
    v_map1 = [j*8 + k for j in range(3) for k in range(3)] # 0,1,2, 8,9,10, 16,17,18
    w_map1 = [k*3 + i for k in range(3) for i in range(3)] # 0,1,2, 3,4,5, 6,7,8
    
    # Map for Block 2 (k=3,4,5)
    u_map2 = list(range(9))
    v_map2 = [j*8 + (3+k) for j in range(3) for k in range(3)] # 3,4,5, 11,12,13, 19,20,21
    w_map2 = [(3+k)*3 + i for k in range(3) for i in range(3)] # 9,10,11, 12,13,14, 15,16,17

    # Map for Block 3 (k=6,7) - Schoolbook 3x3x2
    # Terms are (i, j, k) with 1s.
    # u_map3, v_map3, w_map3
    terms3 = []
    u_map3 = []
    v_map3 = []
    w_map3 = []
    for i in range(3):
        for j in range(3):
            for k in range(2): # k=6,7
                terms3.append(([1]*9, [1]*9, [1]*9)) # Placeholder, we will set specific indices
                u_map3.append(i*3 + j)
                v_map3.append(j*8 + (6+k))
                w_map3.append((6+k)*3 + i)
                
    # Re-construct terms3 properly for schoolbook
    schoolbook_terms = []
    for i in range(3):
        for j in range(3):
            for k in range(2):
                u = [0]*9; u[i*3+j] = 1
                v = [0]*9; v[j*3+k] = 1 # Local k 0,1
                w = [0]*9; w[k*3+i] = 1 # Local k 0,1
                schoolbook_terms.append((u, v, w))
    
    # Apply Block 1
    add_block_mapped(u_map1, v_map1, w_map1, laderman_terms, 0)
    
    # Apply Block 2
    add_block_mapped(u_map2, v_map2, w_map2, laderman_terms, 23)
    
    # Apply Block 3
    add_block_mapped(u_map3, v_map3, w_map3, schoolbook_terms, 46)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
