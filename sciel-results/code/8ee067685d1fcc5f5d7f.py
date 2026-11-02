import time

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=7
    # Rank 25 decomposition: 3 blocks of 2x2x2 (Strassen, rank 7) + 1 block of 2x2x1 (Schoolbook, rank 4)
    # 7+7+7+4 = 25
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)] # 4 x 25
    V = [[0] * rank for _ in range(m * p)] # 14 x 25
    W = [[0] * rank for _ in range(p * n)] # 14 x 25
    
    # Strassen factors for 2x2x2
    # U rows: 00, 01, 10, 11
    # V rows: 00, 01, 10, 11
    # W rows: 00, 01, 10, 11
    
    strassen_U = [
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, 0], # A10
        [1, 1, 0, 1, 0, -1, -1] # A11
    ]
    # Check M1: (A00+A11)(B00+B11) -> U: 1,0,0,1. V: 1,0,0,1.
    # My list above is columns?
    # Let's define columns explicitly to be safe.
    
    # Col 1: M1
    sU1 = [1, 0, 0, 1]
    sV1 = [1, 0, 0, 1]
    sW1 = [1, 0, 0, 1] # C00, C11
    
    # Col 2: M2
    sU2 = [0, 0, 1, 1]
    sV2 = [1, 0, 0, 0]
    sW2 = [0, 0, 1, -1] # C10, C11
    
    # Col 3: M3
    sU3 = [1, 0, 0, 0]
    sV3 = [0, 1, 0, -1]
    sW3 = [0, 1, 0, 1] # C01, C11
    
    # Col 4: M4
    sU4 = [0, 0, 0, 1]
    sV4 = [0, 0, 1, -1]
    sW4 = [1, 0, 1, 0] # C00, C10
    
    # Col 5: M5
    sU5 = [1, 1, 0, 0]
    sV5 = [0, 0, 0, 1]
    sW5 = [-1, 1, 0, 0] # C00, C01
    
    # Col 6: M6
    sU6 = [0, 0, 1, -1]
    sV6 = [1, 1, 0, 0]
    sW6 = [0, 0, 0, 1] # C11
    
    # Col 7: M7
    sU7 = [0, 1, 0, -1]
    sV7 = [0, 0, 1, 1]
    sW7 = [1, 0, 0, 0] # C00
    
    strassen_blocks = [
        (sU1, sV1, sW1),
        (sU2, sV2, sW2),
        (sU3, sV3, sW3),
        (sU4, sV4, sW4),
        (sU5, sV5, sW5),
        (sU6, sV6, sW6),
        (sU7, sV7, sW7)
    ]
    
    # Schoolbook factors for 2x2x1
    # U rows: 00, 01, 10, 11
    # V rows: 0, 1 (corresponding to B0, B1)
    # W rows: 0, 1 (corresponding to C0, C1)
    
    # Col 1: A00 * B0 -> C0
    sbU1 = [1, 0, 0, 0]
    sbV1 = [1, 0]
    sbW1 = [1, 0]
    
    # Col 2: A01 * B1 -> C0
    sbU2 = [0, 1, 0, 0]
    sbV2 = [0, 1]
    sbW2 = [1, 0]
    
    # Col 3: A10 * B0 -> C1
    sbU3 = [0, 0, 1, 0]
    sbV3 = [1, 0]
    sbW3 = [0, 1]
    
    # Col 4: A11 * B1 -> C1
    sbU4 = [0, 0, 0, 1]
    sbV4 = [0, 1]
    sbW4 = [0, 1]
    
    schoolbook_blocks = [
        (sbU1, sbV1, sbW1),
        (sbU2, sbV2, sbW2),
        (sbU3, sbV3, sbW3),
        (sbU4, sbV4, sbW4)
    ]
    
    col_idx = 0
    
    # Block 1: cols 0, 1
    # V local indices 0, 1 -> global k=0, 1
    # W local indices 0,1,2,3 -> global k,i
    # C00 (0) -> k=0, i=0 -> 0
    # C01 (1) -> k=1, i=0 -> 2
    # C10 (2) -> k=0, i=1 -> 1
    # C11 (3) -> k=1, i=1 -> 3
    w_map_1 = [0, 2, 1, 3]
    v_map_1 = [6, 7] # j=0,k=0 -> 6; j=0,k=1 -> 7? No.
    # V index is j*p + k.
    # Local B index 0 is j=0, k=0 -> 0*7+0 = 0.
    # Local B index 1 is j=0, k=1 -> 0*7+1 = 1.
    # Wait, the Strassen V vector has 4 entries corresponding to B00, B01, B10, B11.
    # Local index 0: B00 -> j=0, k=0 -> 0
    # Local index 1: B01 -> j=0, k=1 -> 1
    # Local index 2: B10 -> j=1, k=0 -> 7
    # Local index 3: B11 -> j=1, k=1 -> 8
    v_map_1 = [0, 1, 7, 8]
    
    for u_vec, v_vec, w_vec in strassen_blocks:
        for i in range(4):
            U[i][col_idx] = u_vec[i]
        for l in range(4):
            V[v_map_1[l]][col_idx] = v_vec[l]
        for l in range(4):
            W[w_map_1[l]][col_idx] = w_vec[l]
        col_idx += 1
        
    # Block 2: cols 2, 3
    # V local 0: B00 -> j=0, k=2 -> 2
    # V local 1: B01 -> j=0, k=3 -> 3
    # V local 2: B10 -> j=1, k=2 -> 9
    # V local 3: B11 -> j=1, k=3 -> 10
    v_map_2 = [2, 3, 9, 10]
    
    # W local 0: C00 -> k=2, i=0 -> 4
    # W local 1: C01 -> k=3, i=0 -> 6
    # W local 2: C10 -> k=2, i=1 -> 5
    # W local 3: C11 -> k=3, i=1 -> 7
    w_map_2 = [4, 6, 5, 7]
    
    for u_vec, v_vec, w_vec in strassen_blocks:
        for i in range(4):
            U[i][col_idx] = u_vec[i]
        for l in range(4):
            V[v_map_2[l]][col_idx] = v_vec[l]
        for l in range(4):
            W[w_map_2[l]][col_idx] = w_vec[l]
        col_idx += 1
        
    # Block 3: cols 4, 5
    # V local 0: B00 -> j=0, k=4 -> 4
    # V local 1: B01 -> j=0, k=5 -> 5
    # V local 2: B10 -> j=1, k=4 -> 11
    # V local 3: B11 -> j=1, k=5 -> 12
    v_map_3 = [4, 5, 11, 12]
    
    # W local 0: C00 -> k=4, i=0 -> 8
    # W local 1: C01 -> k=5, i=0 -> 10
    # W local 2: C10 -> k=4, i=1 -> 9
    # W local 3: C11 -> k=5, i=1 -> 11
    w_map_3 = [8, 10, 9, 11]
    
    for u_vec, v_vec, w_vec in strassen_blocks:
        for i in range(4):
            U[i][col_idx] = u_vec[i]
        for l in range(4):
            V[v_map_3[l]][col_idx] = v_vec[l]
        for l in range(4):
            W[w_map_3[l]][col_idx] = w_vec[l]
        col_idx += 1
        
    # Block 4: col 6
    # Schoolbook
    # V local 0: B0 -> j=0, k=6 -> 6
    # V local 1: B1 -> j=1, k=6 -> 13
    v_map_4 = [6, 13]
    
    # W local 0: C0 -> i=0, k=6 -> 12
    # W local 1: C1 -> i=1, k=6 -> 13
    w_map_4 = [12, 13]
    
    for u_vec, v_vec, w_vec in schoolbook_blocks:
        for i in range(4):
            U[i][col_idx] = u_vec[i]
        for l in range(2):
            V[v_map_4[l]][col_idx] = v_vec[l]
        for l in range(2):
            W[w_map_4[l]][col_idx] = w_vec[l]
        col_idx += 1
        
    return {"U": U, "V": V, "W": W}

