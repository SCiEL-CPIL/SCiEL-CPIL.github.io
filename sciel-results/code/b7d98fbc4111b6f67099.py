"""Strassen-based decomposition for 2x2 * 2x15 matrix multiplication."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # We'll use Strassen's algorithm on pairs of columns
    # 7 pairs of columns (14 columns) use 7 multiplications each
    # 1 remaining column uses 4 schoolbook multiplications
    # Total rank = 7*7 + 4 = 53
    
    num_pairs = p // 2  # 7 pairs
    remainder = p % 2   # 1 remaining column
    
    rank = num_pairs * 7 + remainder * 4  # 49 + 4 = 53
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53
    
    col = 0
    
    # Strassen's 7 products for a 2x2 block
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # Products:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # W reconstruction (output indices):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each pair of columns (k_a, k_b):
    # Local B block:
    #   b00 = B[0, k_a]  (row 0, col k_a) -> index 0*15 + k_a
    #   b01 = B[0, k_b]  (row 0, col k_b) -> index 0*15 + k_b
    #   b10 = B[1, k_a]  (row 1, col k_a) -> index 1*15 + k_a
    #   b11 = B[1, k_b]  (row 1, col k_b) -> index 1*15 + k_b
    
    # Output C block (indices in W):
    #   C[0, k_a] -> index k_a * 2 + 0
    #   C[0, k_b] -> index k_b * 2 + 0
    #   C[1, k_a] -> index k_a * 2 + 1
    #   C[1, k_b] -> index k_b * 2 + 1
    
    for pair_idx in range(num_pairs):
        k_a = pair_idx * 2
        k_b = pair_idx * 2 + 1
        
        # Process 7 Strassen multiplications for this pair
        # Each multiplication r gets a column in U, V, W
        
        # M1 = (a00 + a11) * (b00 + b11)
        r = col
        U[0][r] = 1   # a00
        U[3][r] = 1   # a11
        V[0 * 15 + k_a][r] = 1   # b00
        V[1 * 15 + k_b][r] = 1   # b11
        W[k_a * 2 + 0][r] = 1   # C[0, k_a]
        W[k_b * 2 + 1][r] = 1   # C[1, k_b]
        col += 1
        
        # M2 = (a10 + a11) * b00
        r = col
        U[2][r] = 1   # a10
        U[3][r] = 1   # a11
        V[0 * 15 + k_a][r] = 1   # b00
        W[k_a * 2 + 1][r] = 1   # C[1, k_a]
        W[k_b * 2 + 1][r] = -1  # C[1, k_b]
        col += 1
        
        # M3 = a00 * (b01 - b11)
        r = col
        U[0][r] = 1   # a00
        V[0 * 15 + k_b][r] = 1   # b01
        V[1 * 15 + k_b][r] = -1  # b11
        W[k_b * 2 + 0][r] = 1   # C[0, k_b]
        W[k_b * 2 + 1][r] = 1   # C[1, k_b]
        col += 1
        
        # M4 = a11 * (b10 - b00)
        r = col
        U[3][r] = 1   # a11
        V[1 * 15 + k_a][r] = 1   # b10
        V[0 * 15 + k_a][r] = -1  # b00
        W[k_a * 2 + 0][r] = 1   # C[0, k_a]
        W[k_a * 2 + 1][r] = 1   # C[1, k_a]
        col += 1
        
        # M5 = (a00 + a01) * b11
        r = col
        U[0][r] = 1   # a00
        U[1][r] = 1   # a01
        V[1 * 15 + k_b][r] = 1   # b11
        W[k_a * 2 + 0][r] = -1  # C[0, k_a]
        W[k_b * 2 + 0][r] = 1   # C[0, k_b]
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        r = col
        U[2][r] = 1   # a10
        U[0][r] = -1  # a00
        V[0 * 15 + k_a][r] = 1   # b00
        V[0 * 15 + k_b][r] = 1   # b01
        W[k_b * 2 + 1][r] = 1   # C[1, k_b]
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        r = col
        U[1][r] = 1   # a01
        U[3][r] = -1  # a11
        V[1 * 15 + k_a][r] = 1   # b10
        V[1 * 15 + k_b][r] = 1   # b11
        W[k_a * 2 + 0][r] = 1   # C[0, k_a]
        col += 1
    
    # Handle the remaining single column (k = 14)
    if remainder > 0:
        k = 2 * num_pairs  # k = 14
        
        # Schoolbook for single column: 4 products
        # C[0, k] = a00 * B[0, k] + a01 * B[1, k]
        # C[1, k] = a10 * B[0, k] + a11 * B[1, k]
        
        # Product 1: a00 * B[0, k] -> contributes to C[0, k]
        r = col
        U[0][r] = 1
        V[0 * 15 + k][r] = 1
        W[k * 2 + 0][r] = 1
        col += 1
        
        # Product 2: a01 * B[1, k] -> contributes to C[0, k]
        r = col
        U[1][r] = 1
        V[1 * 15 + k][r] = 1
        W[k * 2 + 0][r] = 1
        col += 1
        
        # Product 3: a10 * B[0, k] -> contributes to C[1, k]
        r = col
        U[2][r] = 1
        V[0 * 15 + k][r] = 1
        W[k * 2 + 1][r] = 1
        col += 1
        
        # Product 4: a11 * B[1, k] -> contributes to C[1, k]
        r = col
        U[3][r] = 1
        V[1 * 15 + k][r] = 1
        W[k * 2 + 1][r] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
