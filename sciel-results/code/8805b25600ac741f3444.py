"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 decomposition: 7 (Strassen 2x2x4) + 7 (Strassen 2x2x4) + 4 (Schoolbook 2x2x1)
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen 2x2x2 block
    def add_strassen(offset_u, offset_v, offset_w, col_start):
        # Strassen 7 multiplications
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Indices in U: A11, A12, A21, A22
        # u_0 = A11, u_1 = A12, u_2 = A21, u_3 = A22
        
        # Indices in V: B11, B12, B21, B22
        # v_0 = B11, v_1 = B12, v_2 = B21, v_3 = B22
        
        # M1
        U[offset_u + 0][col_start] += 1
        U[offset_u + 3][col_start] += 1
        V[offset_v + 0][col_start] += 1
        V[offset_v + 3][col_start] += 1
        # W: C11 += M1 - M4 - M5 + M7, C12 += M1 + M3 - M5 - M7
        # C11 is W[0], C12 is W[1] (if p=2)
        # Here we are adding to a specific part of W
        W[offset_w + 0][col_start] += 1
        W[offset_w + 1][col_start] += 1
        
        # M2
        U[offset_u + 2][col_start+1] += 1
        U[offset_u + 3][col_start+1] += 1
        V[offset_v + 0][col_start+1] += 1
        W[offset_w + 0][col_start+1] += 1
        W[offset_w + 1][col_start+1] += 1
        
        # M3
        U[offset_u + 0][col_start+2] += 1
        V[offset_v + 2][col_start+2] += 1
        V[offset_v + 3][col_start+2] -= 1
        W[offset_w + 1][col_start+2] += 1
        
        # M4
        U[offset_u + 3][col_start+3] += 1
        V[offset_v + 1][col_start+3] += 1
        V[offset_v + 0][col_start+3] -= 1
        W[offset_w + 0][col_start+3] -= 1
        
        # M5
        U[offset_u + 0][col_start+4] += 1
        U[offset_u + 1][col_start+4] += 1
        V[offset_v + 3][col_start+4] += 1
        W[offset_w + 0][col_start+4] -= 1
        W[offset_w + 1][col_start+4] -= 1
        
        # M6
        U[offset_u + 2][col_start+5] += 1
        U[offset_u + 0][col_start+5] -= 1
        V[offset_v + 0][col_start+5] += 1
        V[offset_v + 1][col_start+5] += 1
        W[offset_w + 1][col_start+5] += 1
        
        # M7
        U[offset_u + 1][col_start+6] += 1
        U[offset_u + 3][col_start+6] -= 1
        V[offset_v + 2][col_start+6] += 1
        V[offset_v + 3][col_start+6] += 1
        W[offset_w + 0][col_start+6] += 1
        W[offset_w + 1][col_start+6] -= 1

    # Block 1: 2x2x4 (cols 0-3 of B)
    # Split 2x2x4 into two 2x2x2?
    # Strassen works on 2x2x2.
    # For 2x2x4, we can use two Strassen blocks?
    # Rank 7 for 2x2x4 is known.
    # But implementing 2x2x4 Strassen directly is complex.
    # Let's use 2x2x2 (rank 7) + 2x2x2 (rank 7) = 14 for 2x2x4?
    # No, 2x2x4 rank is 7.
    # But if I use two 2x2x2 blocks, I get rank 14.
    # 14 + 14 + 4 = 32.
    # This is the schoolbook-like Strassen (32).
    # Can I do better?
    # If I use rank 7 for 2x2x4, I need the specific decomposition.
    # Since I don't have it memorized, I will use the rank 32 decomposition (2x2x2 x 4 blocks + 2x2x1).
    # 4 * 7 + 4 = 32.
    
    # Reset rank
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4 blocks of 2x2x2
    for b in range(4):
        offset_v = b * 2
        offset_w = b * 2
        add_strassen(0, offset_v, offset_w, col)
        col += 7
        
    # 1 block of 2x2x1 (cols 8 of B)
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # 4 multiplications
    U[0][col] = 1
    V[8][col] = 1
    W[8*2 + 0][col] = 1 # C1, col 8
    col += 1
    
    U[1][col] = 1
    V[9][col] = 1
    W[8*2 + 0][col] = 1 # C1, col 8
    col += 1
    
    U[2][col] = 1
    V[8][col] = 1
    W[8*2 + 1][col] = 1 # C2, col 8
    col += 1
    
    U[3][col] = 1
    V[9][col] = 1
    W[8*2 + 1][col] = 1 # C2, col 8
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
