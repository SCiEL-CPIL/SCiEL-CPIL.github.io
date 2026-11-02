"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 7
    rank = 0
    U = []
    V = []
    W = []
    
    def add_block(block_U, block_V, block_W, i_start, j_start, k_start):
        nonlocal rank, U, V, W
        r_u = len(block_U[0]) if block_U else 0
        r_v = len(block_V[0]) if block_V else 0
        r_w = len(block_W[0]) if block_W else 0
        r = max(r_u, r_v, r_w)
        if r == 0:
            return
            
        # Pad to square if necessary (should not be needed if consistent)
        # But just in case
        block_U = [row + [0] * (r - len(row)) for row in block_U]
        block_V = [row + [0] * (r - len(row)) for row in block_V]
        block_W = [row + [0] * (r - len(row)) for row in block_W]
        
        # Get row counts
        n_u = len(block_U)
        n_v = len(block_V)
        n_w = len(block_W)
        
        # Determine local sizes
        # i_start to i_start + n_i
        # j_start to j_start + n_j
        # k_start to k_start + n_k
        # n_i = n_u / n_j ? No.
        # block_U has rows for (i, j) pairs.
        # block_V has rows for (j, k) pairs.
        # block_W has rows for (k, i) pairs.
        
        # We need to know the dimensions of the sub-block.
        # The caller must pass dimensions or we infer them.
        # Let's pass n_i, n_j, n_k explicitly to avoid confusion.
        
        # Re-do function signature
        pass
        
    def get_decomp_2x2x2():
        # Returns U (4x7), V (4x7), W (4x7)
        U = [
            [1, 0, 0, 0, 1, 0, 0], # A00
            [0, 0, 0, 0, 1, 0, 1], # A01
            [0, 0, 1, 0, 0, 1, 0], # A10
            [0, 1, 0, 1, 0, 0, 0], # A11
            [0, 0, 0, 0, 0, 0, 0], # Dummy? No, 4 rows.
        ]
        # Correcting U for Strassen
        # M1: A00+A11 -> [1, 0, 0, 1]
        # M2: A10+A11 -> [0, 0, 1, 1]
        # M3: A00 -> [1, 0, 0, 0]
        # M4: A11 -> [0, 0, 0, 1]
        # M5: A00+A01 -> [1, 1, 0, 0]
        # M6: A10-A00 -> [-1, 0, 1, 0]
        # M7: A01-A11 -> [0, 1, 0, -1]
        
        U = [
            [1, 0, 1, 0, 1, -1, 0], # A00
            [0, 0, 0, 0, 1, 0, 1],  # A01
            [0, 1, 0, 0, 0, 1, 0],  # A10
            [1, 1, 0, 1, 0, 0, -1]  # A11
        ]
        
        # M1: B00+B11 -> [1, 0, 0, 1]
        # M2: B00 -> [1, 0, 0, 0]
        # M3: B10-B11 -> [0, 0, 1, -1]
        # M4: B01-B00 -> [-1, 1, 0, 0]
        # M5: B11 -> [0, 0, 0, 1]
        # M6: B00+B01 -> [1, 1, 0, 0]
        # M7: B10+B11 -> [0, 0, 1, 1]
        
        V = [
            [1, 1, 0, -1, 0, 1, 0], # B00
            [0, 0, 0, 1, 0, 1, 0],  # B01
            [0, 0, 1, 0, 0, 0, 1],  # B10
            [1, 0, -1, 0, 1, 0, 1]  # B11
        ]
        
        # M1: C00+C11 -> [1, 0, 0, 1]
        # M2: C10-C11? No. C10 (idx 2) +1, C11 (idx 3) -1. [0, 0, 1, -1]
        # M3: C01+C11? C01 (idx 1) +1, C11 (idx 3) +1. [0, 1, 0, 1]
        # M4: C00+C10? C00 (idx 0) +1, C10 (idx 2) +1. [1, 0, 1, 0]
        # M5: C00-C01? C00 (idx 0) -1, C01 (idx 1) +1. [-1, 1, 0, 0]
        # M6: C11? C11 (idx 3) +1. [0, 0, 0, 1]
        # M7: C00? C00 (idx 0) +1. [1, 0, 0, 0]
        
        W = [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 1, 1, 0, 1, 0, 0],  # C01
            [0, 1, 0, 1, 0, 0, 0],  # C10
            [1, -1, 1, 0, 0, 1, 0]  # C11
        ]
        
        return U, V, W

    def get_decomp_2x2x1():
        # A 2x2, B 2x1, C 2x1. Rank 4.
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        U = [
            [1, 0, 0, 0], # A00
            [0, 1, 0, 0], # A01
            [0, 0, 1, 0], # A10
            [0, 0, 0, 1]  # A11
        ]
        # B rows: B00, B10
        V = [
            [1, 0, 1, 0], # B00
            [0, 1, 0, 1]  # B10
        ]
        # W rows: C00, C10
        W = [
            [1, 1, 0, 0], # C00
            [0, 0, 1, 1]  # C10
        ]
        return U, V, W

    def get_decomp_1x2x2():
        # A 1x2, B 2x2, C 1x2. Rank 4.
        # C00 = A00 B00 + A01 B10
        # C01 = A00 B01 + A01 B11
        U = [
            [1, 0, 1, 0], # A00
            [0, 1, 0, 1]  # A01
        ]
        # B rows: B00, B01, B10, B11
        V = [
            [1, 0, 0, 0], # B00
            [0, 1, 0, 0], # B01
            [0, 0, 1, 0], # B10
            [0, 0, 0, 1]  # B11
        ]
        # W rows: C00, C01
        W = [
            [1, 0, 1, 0], # C00
            [0, 1, 0, 1]  # C01
        ]
        return U, V, W

    def get_decomp_1x2x1():
        # A 1x2, B 2x1, C 1x1. Rank 2.
        # C00 = A00 B00 + A01 B10
        U = [
            [1, 0], # A00
            [0, 1]  # A01
        ]
        # B rows: B00, B10
        V = [
            [1, 0], # B00
            [0, 1]  # B10
        ]
        # W rows: C00
        W = [
            [1, 1]  # C00
        ]
        return U, V, W

    def add_block(block_U, block_V, block_W, i_start, j_start, k_start, n_i, n_j, n_k):
        nonlocal U, V, W
        r = len(block_U[0]) if block_U else 0
        if r == 0: return
        
        # block_U size: (n_i * n_j) x r
        # block_V size: (n_j * n_k) x r
        # block_W size: (n_k * n_i) x r
        
        # Expand global U, V, W
        # We append columns to U, V, W
        for c in range(r):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Fill U
            for local_i in range(n_i):
                for local_j in range(n_j):
                    global_i = i_start + local_i
                    global_j = j_start + local_j
                    global_u_idx = global_i * m + global_j
                    local_u_idx = local_i * n_j + local_j
                    u_col[global_u_idx] = block_U[local_u_idx][c]
            
            # Fill V
            for local_j in range(n_j):
                for local_k in range(n_k):
                    global_j = j_start + local_j
                    global_k = k_start + local_k
                    global_v_idx = global_j * p + global_k
                    local_v_idx = local_j * n_k + local_k
                    v_col[global_v_idx] = block_V[local_v_idx][c]
            
            # Fill W
            for local_k in range(n_k):
                for local_i in range(n_i):
                    global_k = k_start + local_k
                    global_i = i_start + local_i
                    global_w_idx = global_k * n + global_i
                    local_w_idx = local_k * n_i + local_i
                    w_col[global_w_idx] = block_W[local_w_idx][c]
            
            if not U:
                U = [u_col]
                V = [v_col]
                W = [w_col]
            else:
                for i in range(len(U)):
                    U[i].append(u_col[i])
                    V[i].append(v_col[i])
                    W[i].append(w_col[i])

    # Partitions:
    # I: [0, 2), [2, 3) -> sizes 2, 1
    # J: [0, 2), [2, 4) -> sizes 2, 2
    # K: [0, 2), [2, 4), [4, 6), [6, 7) -> sizes 2, 2, 2, 1
    
    i_parts = [(0, 2), (2, 1)]
    j_parts = [(0, 2), (2, 2)]
    k_parts = [(0, 2), (2, 2), (4, 2), (6, 1)]
    
    for i_start, n_i in i_parts:
        for j_start, n_j in j_parts:
            for k_start, n_k in k_parts:
                if n_i == 2 and n_j == 2 and n_k == 2:
                    bU, bV, bW = get_decomp_2x2x2()
                elif n_i == 2 and n_j == 2 and n_k == 1:
                    bU, bV, bW = get_decomp_2x2x1()
                elif n_i == 1 and n_j == 2 and n_k == 2:
                    bU, bV, bW = get_decomp_1x2x2()
                elif n_i == 1 and n_j == 2 and n_k == 1:
                    bU, bV, bW = get_decomp_1x2x1()
                else:
                    # Should not happen with these partitions
                    continue
                add_block(bU, bV, bW, i_start, j_start, k_start, n_i, n_j, n_k)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
