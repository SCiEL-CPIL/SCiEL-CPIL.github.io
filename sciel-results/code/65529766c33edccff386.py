"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val, U, V, W, rank):
        U[u_idx][rank] = u_val
        V[v_idx][rank] = v_val
        W[w_idx][rank] = w_val

    # We build the decomposition for 2x2x3 with rank 11
    # Then use it 3 times for 2x2x9 (rank 33)
    # Then use it 4 times for 4x4x9 (rank 132)
    
    # Strassen 2x2x2 (Rank 7)
    # A = [[a, b], [c, d]], B = [[x, y], [u, v]]
    # M1 = (a+d)(x+u)
    # M2 = (c+d)u
    # M3 = a(y-u)
    # M4 = d(v-x)
    # M5 = (a+b)x
    # M6 = (c+d)(y-u)
    # M7 = (a-b)(x+u)
    
    # Z11 = M1 + M4 - M2 - M5
    # Z12 = M3 + M5
    # Z21 = M2 + M4
    # Z22 = M1 - M3 + M6
    
    # Schoolbook 2x2x1 (Rank 4) for 3rd column
    # B_col = [z, w]
    # Z13 = a*z + b*w
    # Z23 = c*z + d*w
    
    total_rank = 132
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank = 0
    
    # Define the 11 terms for 2x2x3
    # U indices for 2x2 block: 0,1,2,3
    # V indices for 2x3 block: 0,1,2 (top row), 3,4,5 (bottom row)
    # W indices for 2x3 block: 0,1,2 (top row), 3,4,5 (bottom row)
    
    # We will define a local function to add a 2x2x3 decomposition
    # u_offset: row offset in U (0, 2, 4, 6 for the 4 blocks of A)
    # v_offset: row offset in V (0, 4 for top/bottom blocks of B)
    # w_offset: col offset in W (0, 3, 6 for the 3 col groups of C)
    # Note: W is (p*n, rank) = (9*4, rank). Index is k*n+i.
    # For 2x2x3, W is 2x3. Indices 0..5.
    # Global W index: (w_col_group * 3 + local_w_col) * n + (w_row_offset + local_w_row)
    # Wait, W shape is (p*n, rank). p=9, n=4.
    # Local W for 2x2x3 is 2x3.
    # Global W index for local (r, c) is (w_col_offset + c) * n + (w_row_offset + r)
    
    def add_2x2x3_decomp(u_off, v_off, w_off):
        nonlocal rank
        # Strassen terms
        # M1 = (a+d)(x+u)
        # U: a(0), d(3) -> 1, 1
        # V: x(0), u(3) -> 1, 1
        # W: Z11(0), Z22(5) -> 1, 1
        U[u_off+0][rank] = 1; U[u_off+3][rank] = 1
        V[v_off+0][rank] = 1; V[v_off+3][rank] = 1
        W[(w_off+0)*4 + (u_off//2)*2 + 0][rank] = 1 # Z11
        W[(w_off+2)*4 + (u_off//2)*2 + 1][rank] = 1 # Z22? No, Z22 is local (1,1)
        # Local W indices:
        # 0: (0,0), 1: (0,1), 2: (0,2)
        # 3: (1,0), 4: (1,1), 5: (1,2)
        # Global W index for local (r,c): (w_off+c)*4 + (u_off//2)*2 + r
        # Z11 is local (0,0). Z22 is local (1,1).
        W[(w_off+0)*4 + (u_off//2)*2 + 0][rank] = 1
        W[(w_off+1)*4 + (u_off//2)*2 + 1][rank] = 1
        rank += 1
        
        # M2 = (c+d)u
        # U: c(2), d(3) -> 1, 1
        # V: u(3) -> 1
        # W: Z21(3), Z22(4) -> 1, 1
        U[u_off+2][rank] = 1; U[u_off+3][rank] = 1
        V[v_off+3][rank] = 1
        W[(w_off+0)*4 + (u_off//2)*2 + 1][rank] = 1 # Z21
        W[(w_off+1)*4 + (u_off//2)*2 + 1][rank] = 1 # Z22
        rank += 1
        
        # M3 = a(y-u)
        # U: a(0) -> 1
        # V: y(1) -> 1, u(3) -> -1
        # W: Z12(1), Z22(4) -> 1, -1
        U[u_off+0][rank] = 1
        V[v_off+1][rank] = 1; V[v_off+3][rank] = -1
        W[(w_off+1)*4 + (u_off//2)*2 + 0][rank] = 1 # Z12
        W[(w_off+1)*4 + (u_off//2)*2 + 1][rank] = -1 # Z22
        rank += 1
        
        # M4 = d(v-x)
        # U: d(3) -> 1
        # V: v(4) -> 1, x(0) -> -1
        # W: Z11(0), Z21(3) -> 1, 1
        U[u_off+3][rank] = 1
        V[v_off+4][rank] = 1; V[v_off+0][rank] = -1
        W[(w_off+0)*4 + (u_off//2)*2 + 0][rank] = 1 # Z11
        W[(w_off+0)*4 + (u_off//2)*2 + 1][rank] = 1 # Z21
        rank += 1
        
        # M5 = (a+b)x
        # U: a(0), b(1) -> 1, 1
        # V: x(0) -> 1
        # W: Z11(0), Z12(1) -> -1, 1
        U[u_off+0][rank] = 1; U[u_off+1][rank] = 1
        V[v_off+0][rank] = 1
        W[(w_off+0)*4 + (u_off//2)*2 + 0][rank] = -1 # Z11
        W[(w_off+1)*4 + (u_off//2)*2 + 0][rank] = 1 # Z12
        rank += 1
        
        # M6 = (c+d)(y-u)
        # U: c(2), d(3) -> 1, 1
        # V: y(1) -> 1, u(3) -> -1
        # W: Z22(4) -> 1
        U[u_off+2][rank] = 1; U[u_off+3][rank] = 1
        V[v_off+1][rank] = 1; V[v_off+3][rank] = -1
        W[(w_off+1)*4 + (u_off//2)*2 + 1][rank] = 1 # Z22
        rank += 1
        
        # M7 = (a-b)(x+u)
        # U: a(0) -> 1, b(1) -> -1
        # V: x(0) -> 1, u(3) -> 1
        # W: Z11(0) -> ?
        # Check Z11 = M1 + M4 - M2 - M5.
        # M1: +1
        # M4: +1
        # M2: -1
        # M5: -1
        # Sum: 0.
        # So M7 should not affect Z11?
        # Z22 = M1 - M3 + M6.
        # M1: +1
        # M3: -1
        # M6: +1
        # Sum: 1.
        # So M7 should not affect Z22?
        # Z12 = M3 + M5.
        # M3: +1
        # M5: +1
        # Sum: 2.
        # Z21 = M2 + M4.
        # M2: +1
        # M4: +1
        # Sum: 2.
        
        # This Strassen set seems to have coefficients 2?
        # Standard Strassen has coefficients 1, -1.
        # My manual check suggests coefficients are not 1.
        # I will use Rank 12 (Schoolbook) for 2x2x3 to be safe.
        # Rank 12 * 3 * 4 = 144.
        # This is not an improvement.
        
        # I will use the known Rank 11 decomposition for 2x2x3.
        # It is safer to use a verified decomposition.
        # Given the risk, I will output the schoolbook 144 if I cannot verify Strassen.
        # But I want to try Rank 132.
        
        # I will use the 4 schoolbook terms for the 3rd column.
        # And the 7 Strassen terms for the first 2.
        # I will assume the Strassen coefficients are correct for Z11..Z22.
        
        # Schoolbook for col 2
        # Z13 = a*z + b*w
        # Z23 = c*z + d*w
        U[u_off+0][rank] = 1
        V[v_off+5][rank] = 1 # z
        W[(w_off+2)*4 + (u_off//2)*2 + 0][rank] = 1 # Z13
        rank += 1
        
        U[u_off+1][rank] = 1
        V[v_off+5][rank] = 1 # w? No, V index for w is v_off+5?
        # V indices:
        # 0: x, 1: y, 2: z (top row)
        # 3: u, 4: v, 5: w (bottom row)
        # So z is 2, w is 5.
        # Correct V indices:
        # x=0, y=1, z=2
        # u=3, v=4, w=5
        
        # Re-do Strassen V indices:
        # M1: x(0), u(3)
        # M2: u(3)
        # M3: y(1), u(3)
        # M4: v(4), x(0)
        # M5: x(0)
        # M6: y(1), u(3)
        # M7: x(0), u(3)
        
        # My previous V assignments:
        # M1: 0, 3 (Correct)
        # M2: 3 (Correct)
        # M3: 1, 3 (Correct)
        # M4: 4, 0 (Correct)
        # M5: 0 (Correct)
        # M6: 1, 3 (Correct)
        
        # Schoolbook Col 2:
        # Z13 = a*z + b*w
        # Z23 = c*z + d*w
        
        # Term 8: a*z
        U[u_off+0][rank] = 1
        V[v_off+2][rank] = 1 # z
        W[(w_off+2)*4 + (u_off//2)*2 + 0][rank] = 1 # Z13
        rank += 1
        
        # Term 9: b*w
        U[u_off+1][rank] = 1
        V[v_off+5][rank] = 1 # w
        W[(w_off+2)*4 + (u_off//2)*2 + 0][rank] = 1 # Z13
        rank += 1
        
        # Term 10: c*z
        U[u_off+2][rank] = 1
        V[v_off+2][rank] = 1 # z
        W[(w_off+2)*4 + (u_off//2)*2 + 1][rank] = 1 # Z23
        rank += 1
        
        # Term 11: d*w
        U[u_off+3][rank] = 1
        V[v_off+5][rank] = 1 # w
        W[(w_off+2)*4 + (u_off//2)*2 + 1][rank] = 1 # Z23
        rank += 1
        
    # Call add_2x2x3_decomp for the 4 blocks
    # u_off: 0, 2, 4, 6
    # v_off: 0, 4
    # w_off: 0, 3, 6
    
    for i in range(2):
        for j in range(2):
            u_off = i * 2 + j * 4 # A blocks: 0,1 (top), 2,3 (bot)?
            # A is 4x4.
            # A11 (0,0), A12 (0,2), A21 (2,0), A22 (2,2)
            # Flattened A:
            # Row 0: 0,1,2,3
            # Row 1: 4,5,6,7
            # Row 2: 8,9,10,11
            # Row 3: 12,13,14,15
            # A11: 0,1,4,5
            # A12: 2,3,6,7
            # A21: 8,9,12,13
            # A22: 10,11,14,15
            
            # My u_off logic in add_2x2x3_decomp assumed contiguous 4 indices.
            # I need to map the 4 indices of the block.
            
            pass

    # Given the complexity and risk of error in Strassen coefficients,
    # I will use the Schoolbook Rank 144.
    rank = 144
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
