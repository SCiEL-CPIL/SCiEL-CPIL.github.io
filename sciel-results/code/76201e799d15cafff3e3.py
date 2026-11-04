"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build a decomposition of rank 70.
    # Part 1: 4x4x4 core (Rank 49)
    # Part 2: 4x4x1 term (Rank 16)
    # Part 3: 4x1x4 term (Rank 1)
    # Part 4: 4x1x1 term (Rank 4)
    
    total_rank = 49 + 16 + 1 + 4
    U = [[0] * total_rank for _ in range(n * m)] # 20 x 70
    V = [[0] * total_rank for _ in range(m * p)] # 25 x 70
    W = [[0] * total_rank for _ in range(p * n)] # 20 x 70
    
    rank_offset = 0
    
    # --- Part 1: 4x4x4 using Hierarchical Strassen ---
    # Indices for A_L (4x4) in U: rows 0..15 (4*4)
    # Indices for B_LL (4x4) in V: rows 0..15 (4*4)
    # Indices for C_L (4x4) in W: rows 0..15 (4*4) -- Note: W index is k*n+i.
    # C is 4x5. C_L is 4x4.
    # C[k, i] for i in 0..3, k in 0..3.
    # W index: k*4 + i.
    
    def add_strassen(n_size, u_base, v_base, w_base, offset):
        """
        Recursively add Strassen decomposition for n_size x n_size x n_size.
        u_base: starting row index in U for this block
        v_base: starting row index in V for this block
        w_base: starting row index in W for this block
        offset: current column index in U, V, W
        """
        if n_size == 1:
            U[u_base][offset] = 1
            V[v_base][offset] = 1
            W[w_base][offset] = 1
            return offset + 1
        
        half = n_size // 2
        # Strassen 7 products
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to map the linear combinations to U and V rows.
        # This is complex to do recursively without expanding the vectors.
        # Instead, we can just expand the base case.
        # For n_size=2, we add 7 columns.
        # For n_size=4, we add 7 * (rank of 2x2x2) columns.
        
        # Let's just hardcode the 49 terms for 4x4x4 to avoid recursion complexity in the flat structure.
        # Or, simpler: Use the fact that I can generate the 49 terms by applying Strassen logic.
        
        # Actually, let's just use Schoolbook for the 4x4x4 part? No, that's 64.
        # Let's use the Rank 49 algorithm.
        # I will generate the 49 terms programmatically.
        
        # This function is a placeholder for the recursive logic.
        # Due to complexity, I will implement a simpler Rank 64 (2x2 Strassen on 2x2 blocks of 4x4)
        # Wait, 2x2 Strassen on 2x2 blocks of 4x4 is exactly Rank 49.
        
        # Let's implement the recursion properly.
        # We need to update U, V, W with the linear combinations.
        
        # For M1: (A11+A22)(B11+B22)
        # U rows for A11+A22:
        #   A11 is at u_base..u_base+3
        #   A22 is at u_base+4..u_base+7
        #   We need to add 1 to both sets of rows in the new U column?
        #   No, U is a matrix. We can have fractional entries.
        #   So U[u_base+i][offset] += 1 for i in 0..3
        #   U[u_base+4+i][offset] += 1 for i in 0..3
        #   Similarly for V.
        
        # This works!
        
        # M1
        for i in range(half * half):
            U[u_base + i][offset] += 1
            U[u_base + half * half + i][offset] += 1
            V[v_base + i][offset] += 1
            V[v_base + half * half + i][offset] += 1
        # W: M1 contributes to C11 and C22
        # C11 is at w_base..w_base+3
        # C22 is at w_base+4..w_base+7
        for i in range(half * half):
            W[w_base + i][offset] += 1
            W[w_base + half * half + i][offset] += 1
        offset += 1
        
        # M2: (A21+A22)B11
        # A21 is at u_base+2..5? No.
        # A = [[A11, A12], [A21, A22]]
        # A11: 0..3, A12: 4..7, A21: 8..11, A22: 12..15 (for n=4)
        # For general n_size:
        # A11: 0..h2-1
        # A12: h2..2h2-1
        # A21: 2h2..3h2-1
        # A22: 3h2..4h2-1
        # where h2 = half * half
        
        h2 = half * half
        
        # M2: (A21+A22)B11
        for i in range(h2):
            U[u_base + 2*h2 + i][offset] += 1 # A21
            U[u_base + 3*h2 + i][offset] += 1 # A22
            V[v_base + i][offset] += 1 # B11
        # W: M2 contributes to C21
        # C21 is at w_base + 2*h2 .. 3*h2 - 1
        for i in range(h2):
            W[w_base + 2*h2 + i][offset] += 1
        offset += 1
        
        # M3: A11(B12-B22)
        for i in range(h2):
            U[u_base + i][offset] += 1 # A11
            V[v_base + h2 + i][offset] += 1 # B12
            V[v_base + 3*h2 + i][offset] -= 1 # B22
        # W: M3 contributes to C12
        # C12 is at w_base + h2 .. 2*h2 - 1
        for i in range(h2):
            W[w_base + h2 + i][offset] += 1
        offset += 1
        
        # M4: A22(B21-B11)
        for i in range(h2):
            U[u_base + 3*h2 + i][offset] += 1 # A22
            V[v_base + 2*h2 + i][offset] += 1 # B21
            V[v_base + i][offset] -= 1 # B11
        # W: M4 contributes to C11 and C21
        for i in range(h2):
            W[w_base + i][offset] += 1 # C11
            W[w_base + 2*h2 + i][offset] += 1 # C21
        offset += 1
        
        # M5: (A11+A12)B22
        for i in range(h2):
            U[u_base + i][offset] += 1 # A11
            U[u_base + h2 + i][offset] += 1 # A12
            V[v_base + 3*h2 + i][offset] += 1 # B22
        # W: M5 contributes to C11 (-) and C12 (+)
        for i in range(h2):
            W[w_base + i][offset] -= 1 # C11
            W[w_base + h2 + i][offset] += 1 # C12
        offset += 1
        
        # M6: (A21-A11)(B11+B12)
        for i in range(h2):
            U[u_base + 2*h2 + i][offset] += 1 # A21
            U[u_base + i][offset] -= 1 # A11
            V[v_base + i][offset] += 1 # B11
            V[v_base + h2 + i][offset] += 1 # B12
        # W: M6 contributes to C22
        for i in range(h2):
            W[w_base + 3*h2 + i][offset] += 1 # C22
        offset += 1
        
        # M7: (A12-A22)(B21+B22)
        for i in range(h2):
            U[u_base + h2 + i][offset] += 1 # A12
            U[u_base + 3*h2 + i][offset] -= 1 # A22
            V[v_base + 2*h2 + i][offset] += 1 # B21
            V[v_base + 3*h2 + i][offset] += 1 # B22
        # W: M7 contributes to C11 and C22
        for i in range(h2):
            W[w_base + i][offset] += 1 # C11
            W[w_base + 3*h2 + i][offset] += 1 # C22
        offset += 1
        
        return offset

    # Call for 4x4x4
    # A_L is rows 0..15 of U
    # B_LL is rows 0..15 of V
    # C_L is rows 0..15 of W (indices k*4+i for k,i in 0..3)
    rank_offset = add_strassen(4, 0, 0, 0, rank_offset)
    
    # --- Part 2: A_L B_LR (4x4 * 4x1) ---
    # A_L: U rows 0..15
    # B_LR: V rows 16..19 (4x1 block of B)
    # C_R: W rows 16..19 (4x1 block of C, indices k*4+4 for k in 0..3)
    # Schoolbook: 16 multiplications
    for i in range(4):
        for j in range(4):
            U[i * 4 + j][rank_offset] = 1
            V[16 + j][rank_offset] = 1
            W[16 + i][rank_offset] = 1
            rank_offset += 1
            
    # --- Part 3: A_R B_RL (4x1 * 1x4) ---
    # A_R: U rows 16..19
    # B_RL: V rows 20..23
    # C_L: W rows 0..15
    # Outer product: 1 multiplication
    for i in range(4):
        U[16 + i][rank_offset] = 1
    for j in range(4):
        V[20 + j][rank_offset] = 1
    for i in range(4):
        for j in range(4):
            W[j * 4 + i][rank_offset] = 1
    rank_offset += 1
    
    # --- Part 4: A_R B_RR (4x1 * 1x1) ---
    # A_R: U rows 16..19
    # B_RR: V row 24
    # C_R: W rows 16..19
    # 4 multiplications
    for i in range(4):
        U[16 + i][rank_offset] = 1
        V[24][rank_offset] = 1
        W[16 + i][rank_offset] = 1
        rank_offset += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
