import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (8, rank)
    
    # Rank 28 using Strassen-like decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen terms for 2x2x2 multiplication
    # Each term is a tuple (U_vec, V_vec, W_vec)
    # U_vec: coefficients for A (2x2 -> 4 elements)
    # V_vec: coefficients for B (2x2 -> 4 elements)
    # W_vec: coefficients for C (2x2 -> 4 elements)
    
    # Local indices: A[0,0], A[0,1], A[1,0], A[1,1]
    # B[0,0], B[0,1], B[1,0], B[1,1]
    # C[0,0], C[0,1], C[1,0], C[1,1]
    
    terms = [
        # Term 1: (A00+A11)(B00+B11) -> C00+, C11+
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # Term 2: (A10+A11)B00 -> C10+, C11-
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # Term 3: A00(B01-B11) -> C01+, C11+
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # Term 4: A11(B10-B00) -> C00+, C10+
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # Term 5: (A00+A01)B11 -> C00-, C01+
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # Term 6: (A10-A00)(B00+B01) -> C11+
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # Term 7: (A01-A11)(B10+B11) -> C00+
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # 4 blocks of 2x2x2
    # Block 0: A cols 0-1, B rows 0-1 cols 0-1, C cols 0-1
    # Block 1: A cols 2-3, B rows 2-3 cols 0-1, C cols 0-1
    # Block 2: A cols 0-1, B rows 0-1 cols 2-3, C cols 2-3
    # Block 3: A cols 2-3, B rows 2-3 cols 2-3, C cols 2-3
    
    blocks = [
        {'a_j_off': 0, 'b_j_off': 0, 'c_k_off': 0},
        {'a_j_off': 2, 'b_j_off': 2, 'c_k_off': 0},
        {'a_j_off': 0, 'b_j_off': 0, 'c_k_off': 2},
        {'a_j_off': 2, 'b_j_off': 2, 'c_k_off': 2},
    ]
    
    col_idx = 0
    for block in blocks:
        a_j_off = block['a_j_off']
        b_j_off = block['b_j_off']
        c_k_off = block['c_k_off']
        
        for u_loc, v_loc, w_loc in terms:
            # Map U (A)
            # Local A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global A indices: i, j = a_j_off + lb
            # U row: i * m + j
            for la in range(2):
                for lb in range(2):
                    local_idx = la * 2 + lb
                    val = u_loc[local_idx]
                    if val != 0:
                        global_i = la
                        global_j = a_j_off + lb
                        u_row = global_i * m + global_j
                        U[u_row][col_idx] += val
                        
            # Map V (B)
            # Local B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global B indices: j = b_j_off + lb, k = c_k_off + lc
            # V row: j * p + k
            for lb in range(2):
                for lc in range(2):
                    local_idx = lb * 2 + lc
                    val = v_loc[local_idx]
                    if val != 0:
                        global_j = b_j_off + lb
                        global_k = c_k_off + lc
                        v_row = global_j * p + global_k
                        V[v_row][col_idx] += val
                        
            # Map W (C)
            # Local C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Global C indices: i = la, k = c_k_off + lc
            # W row: k * n + i
            for la in range(2):
                for lc in range(2):
                    local_idx = la * 2 + lc
                    val = w_loc[local_idx]
                    if val != 0:
                        global_i = la
                        global_k = c_k_off + lc
                        w_row = global_k * n + global_i
                        W[w_row][col_idx] += val
                        
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
