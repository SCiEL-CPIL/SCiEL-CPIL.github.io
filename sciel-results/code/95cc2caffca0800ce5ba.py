import numpy as np

def strassen_factors(n):
    if n == 1:
        return [[1]], [[1]], [[1]]
    
    if n == 2:
        # Hardcoded factors for 2x2x2
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: 00, 01, 10, 11
        U = [
            [1, 0, 1, 0, 1, 0, 0], # A00
            [0, 0, 0, 0, 1, 0, 0], # A01
            [0, 1, 0, 0, 0, 1, 0], # A10
            [1, 1, 0, 1, 0, 0, 0]  # A11
        ]
        V = [
            [1, 1, 0, -1, 0, 1, 0], # B00
            [0, 0, 1, 0, 0, 1, 0], # B01
            [0, 0, 0, 1, 0, 0, 1], # B10
            [1, 0, -1, 0, 1, 0, 1]  # B11
        ]
        W = [
            [1, 0, 0, 1, -1, 0, 1], # C00
            [0, 0, 1, 0, 1, 0, 0], # C01
            [0, 1, 0, 1, 0, 0, 0], # C10
            [1, -1, 1, 0, 0, 1, 0]  # C11
        ]
        return U, V, W

    n_sub = n // 2
    U_sub, V_sub, W_sub = strassen_factors(n_sub)
    R_sub = len(U_sub[0])
    R = 7 * R_sub
    
    U = [[0] * R for _ in range(n * n)]
    V = [[0] * R for _ in range(n * n)]
    W = [[0] * R for _ in range(n * n)]
    
    # Strassen coefficients
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    
    a_coeffs = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: 1}, # M2
        {0: 1},       # M3
        {3: 1},       # M4
        {0: 1, 1: 1}, # M5
        {2: 1, 0: -1},# M6
        {1: 1, 3: -1} # M7
    ]
    
    b_coeffs = [
        {0: 1, 3: 1}, # M1
        {0: 1},       # M2
        {1: 1, 3: -1},# M3
        {2: 1, 0: -1},# M4
        {3: 1},       # M5
        {0: 1, 1: 1}, # M6
        {2: 1, 3: 1}  # M7
    ]
    
    c_coeffs = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: -1},# M2
        {1: 1, 3: 1}, # M3
        {0: 1, 2: 1}, # M4
        {0: -1, 1: 1},# M5
        {3: 1},       # M6
        {0: 1}        # M7
    ]
    
    for k in range(7):
        col_offset = k * R_sub
        # Fill U
        for block_idx, coeff in a_coeffs[k].items():
            # Block block_idx is at row block_idx // 2, col block_idx % 2
            r_block = block_idx // 2
            c_block = block_idx % 2
            for i in range(n_sub):
                for j in range(n_sub):
                    global_i = r_block * n_sub + i
                    global_j = c_block * n_sub + j
                    u_idx = global_i * n + global_j
                    for r in range(R_sub):
                        U[u_idx][col_offset + r] += coeff * U_sub[i * n_sub + j][r]
                        
        # Fill V
        for block_idx, coeff in b_coeffs[k].items():
            r_block = block_idx // 2
            c_block = block_idx % 2
            for i in range(n_sub):
                for j in range(n_sub):
                    global_i = r_block * n_sub + i
                    global_j = c_block * n_sub + j
                    v_idx = global_i * n + global_j
                    for r in range(R_sub):
                        V[v_idx][col_offset + r] += coeff * V_sub[i * n_sub + j][r]
                        
        # Fill W
        for block_idx, coeff in c_coeffs[k].items():
            r_block = block_idx // 2
            c_block = block_idx % 2
            for i in range(n_sub):
                for j in range(n_sub):
                    global_i = r_block * n_sub + i
                    global_j = c_block * n_sub + j
                    w_idx = global_i * n + global_j
                    for r in range(R_sub):
                        W[w_idx][col_offset + r] += coeff * W_sub[i * n_sub + j][r]
                        
    return U, V, W

def schoolbook_factors(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r_idx] = 1
                V[v_idx][r_idx] = 1
                W[w_idx][r_idx] = 1
                r_idx += 1
    return U, V, W

def construct():
    n, m, p = 4, 4, 6
    
    # Part 1: 4x4 x 4x4
    U1, V1, W1 = strassen_factors(4)
    rank1 = len(U1[0])
    
    # Part 2: 4x4 x 4x2
    U2, V2, W2 = schoolbook_factors(4, 4, 2)
    rank2 = len(U2[0])
    
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Copy U1
    for i in range(n * m):
        for r in range(rank1):
            U[i][r] = U1[i][r]
            
    # Copy U2 (rows 0..15 of U2 map to rows 0..15 of U)
    for i in range(n * m):
        for r in range(rank2):
            U[i][rank1 + r] = U2[i][r]
            
    # Copy V1 (rows 0..15 of V1 map to rows 0..15 of V)
    for i in range(16):
        for r in range(rank1):
            V[i][r] = V1[i][r]
            
    # Copy V2 (rows 0..7 of V2 map to rows 16..23 of V)
    # V2 is 4x2 = 8 rows.
    # B is 4x6. Rows 0..3 correspond to B[:, 0:4]. Rows 4..7 correspond to B[:, 4:6]?
    # No, B index is j * p + k.
    # j=0, k=0..5 -> 0..5
    # j=1, k=0..5 -> 6..11
    # j=2, k=0..5 -> 12..17
    # j=3, k=0..5 -> 18..23
    # V1 uses B[:, 0:4]. Indices:
    # j=0, k=0..3 -> 0..3
    # j=1, k=0..3 -> 6..9
    # j=2, k=0..3 -> 12..15
    # j=3, k=0..3 -> 18..21
    # Wait, V1 is 16 rows.
    # The indices in V1 are $j \cdot 4 + k$ for $k \in 0..3$.
    # In the global V, the index is $j \cdot 6 + k$.
    # So we need to map V1 row $i$ to global row $j \cdot 6 + k$.
    
    # Let's re-map V1 and V2 carefully.
    
    # Clear V and W
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Map V1 (16 rows) to V (24 rows)
    # V1 row index $i = j \cdot 4 + k$.
    # Global V row index $g = j \cdot 6 + k$.
    for i in range(16):
        j = i // 4
        k = i % 4
        g = j * 6 + k
        for r in range(rank1):
            V[g][r] = V1[i][r]
            
    # Map V2 (8 rows) to V (24 rows)
    # V2 is for 4x2. Rows $i = j \cdot 2 + k'$ where $k' \in 0..1$.
    # This corresponds to B[:, 4:6]. So $k = 4 + k'$.
    # Global V row index $g = j \cdot 6 + (4 + k')$.
    for i in range(8):
        j = i // 2
        k_prime = i % 2
        k = 4 + k_prime
        g = j * 6 + k
        for r in range(rank2):
            V[g][rank1 + r] = V2[i][r]
            
    # Map W1 (16 rows) to W (24 rows)
    # W1 row index $i = k \cdot 4 + i'$ where $k \in 0..3, i' \in 0..3$.
    # Global W row index $g = k \cdot 4 + i'$.
    # Since $p=6$, $k$ in global is $0..5$.
    # W1 corresponds to $k \in 0..3$.
    for i in range(16):
        k = i // 4
        i_prime = i % 4
        g = k * 4 + i_prime
        for r in range(rank1):
            W[g][r] = W1[i][r]
            
    # Map W2 (8 rows) to W (24 rows)
    # W2 is for 4x2. Rows $i = k' \cdot 4 + i'$ where $k' \in 0..1, i' \in 0..3$.
    # This corresponds to $k = 4 + k'$.
    # Global W row index $g = (4 + k') \cdot 4 + i'$.
    for i in range(8):
        k_prime = i // 4
        i_prime = i % 4
        k = 4 + k_prime
        g = k * 4 + i_prime
        for r in range(rank2):
            W[g][rank1 + r] = W2[i][r]
            
    return {"U": U, "V": V, "W": W}
