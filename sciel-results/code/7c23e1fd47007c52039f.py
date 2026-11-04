"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    m1, m2 = 2, 1
    p_blocks = 5
    block_size = 2
    
    # Strassen decomposition for 2x2x2
    # Coefficients for U (A), V (B), W (C)
    # U index: i*m1+j
    # V index: j*block_size+k_offset
    # W index: k_offset*n+i
    
    S = [
        # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00 -> C10+=1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        # M3: A00(B11-B01) -> C01+=1, C11+=1
        ([1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 1, 1]),
        # M4: A11(B10-B00) -> C00+=1, C10+=1
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        # M5: (A00+A01)B11 -> C00-=1, C01+=1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6: (A10-A00)(B00+B01) -> C11+=1
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A01-A11)(B10+B11) -> C00+=1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    rank_part1 = p_blocks * 7
    rank_part2 = n * p
    total_rank = rank_part1 + rank_part2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for b in range(p_blocks):
        for s in S:
            u_coeffs, v_coeffs, w_coeffs = s
            
            # Fill U
            for k in range(4):
                i = k // m1
                j = k % m1
                U[i * m + j][col] = u_coeffs[k]
            
            # Fill V
            for k in range(4):
                j = k // block_size
                k_offset = k % block_size
                v_idx = j * p + (b * block_size + k_offset)
                V[v_idx][col] = v_coeffs[k]
            
            # Fill W
            for k in range(4):
                k_offset = k // n
                i = k % n
                w_idx = (b * block_size + k_offset) * n + i
                W[w_idx][col] = w_coeffs[k]
                
            col += 1
            
    # Fill Part 2: A2 (n x m2) * B2 (m2 x p)
    # m2=1, j=2
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
