"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    # Rank 28: Strassen for 2x2x2 blocks (cols 0-6, 11-17) + schoolbook for rest
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
    
    # Helper for Strassen 2x2x2 block
    # A rows: a0, a1 (indices in U: a0*3+j, a1*3+j)
    # B cols: b0, b1 (indices in V: j*5+b0, j*5+b1)
    # C output: (a0,b0), (a0,b1), (a1,b0), (a1,b1) -> W indices: b*2+a
    def strassen_2x2x2(base_col, a_rows, b_cols):
        # a_rows: list of U base indices (i*3) for the two rows
        # b_cols: list of V col offsets (0-4) for the two cols
        # W base: b_col * n + a_row
        
        # M1 = A00 * B00
        for j in range(m):
            pass  # This is getting complex with the mixed dimensions
        
    # Let me just hardcode all 28 columns directly
    # 
    # Part 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, 7 terms, cols 0-6)
    # Part 2a: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2] (schoolbook, 2 terms, cols 7-8)
    # Part 2b: A[0:2, 2] * B[0:2, 0:2] -> C[0:2, 0:2] (schoolbook, 2 terms, cols 9-10)
    # Wait, let me re-think the decomposition.
    #
    # A is 2x3, B is 3x5. C = A*B is 2x5.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Decomposition strategy:
    # Split A into [A_left (2x2), A_right (2x1)]
    # Split B into [B_left (3x2), B_mid (3x1), B_right (3x2)]
    #
    # C = A_left*B_left + A_left*B_mid + A_left*B_right + A_right*B_left + A_right*B_mid + A_right*B_right
    #
    # To use Strassen:
    # A_left*B_left: 2x2x2 -> 7 terms (cols 0-6)
    # A_right*B_right: (2x1)*(3x2) = 2x2 output -> but this is not square...
    #
    # Actually, let me think differently.
    # 
    # The previous best found rank 28. Let me just construct it carefully.
    #
    # Strategy: 
    # Block 1: A[0:2,0:2] * B[0:2,0:2] = C[0:2,0:2] via Strassen (7)
    # Block 2: A[0:2,0:2] * B[2,0:2] = C[0:2,0:2] via schoolbook (2)
    # Block 3: A[0:2,2] * B[0:2,0:2] = C[0:2,0:2] via schoolbook (2)
    # Block 4: A[0:2,0:2] * B[0:2,2] = C[0:2,2] via schoolbook (2)
    # Block 5: A[0:2,2] * B[2,2] = C[0:2,2] via schoolbook (1)
    # Block 6: A[0:2,0:2] * B[0:2,3:5] = C[0:2,3:5] via Strassen-like? 
    #
    # This is getting complicated. Let me just use the known rank-28 result.
    # 
    # Known approach: 
    # - Use Strassen for the 2x2 subproblem twice
    # - Use schoolbook for the remaining parts
    #
    # Let me define it as:
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2]*B[2,0:2]
    # C[0:2, 2] = A[0:2,0:2]*B[0:2,2] + A[0:2,2]*B[2,2]
    # C[0:2, 3:5] = A[0:2,0:2]*B[0:2,3:5] + A[0:2,2]*B[2,3:5]
    #
    # For C[0:2, 0:2]: Strassen(A[0:2,0:2], B[0:2,0:2]) + 2 schoolbook terms
    # For C[0:2, 2]: 2+1=3 schoolbook terms
    # For C[0:2, 3:5]: This is 2x2, can use Strassen? But A part is 2x2, B part is 2x2 (cols 3,4 of B[0:2] and B[2])
    # Actually B[0:2, 3:5] is 2x2, and we need A[0:2,0:2]*B[0:2,3:5] + A[0:2,2]*B[2,3:5]
    # The second term is (2x1)*(2x2) -> not square for Strassen
    #
    # Total: 7 + 2 + 3 + ? 
    # 
    # Let me just go with a clean schoolbook + one Strassen:
    # Use Strassen for A[0:2,0:2]*B[0:2,0:2] -> 7
    # Everything else schoolbook:
    # A[0:2,2]*B[2,0:5] -> 2*5 = 10 terms
    # A[0:2,0:2]*B[0:2,2:5] -> 2*3 = 6 terms  
    # A[0:2,0:2]*B[2,2:5] -> 2*3 = 6 terms
    # Total: 7 + 10 + 6 + 6 = 29. That's rank 29.
    #
    # To get 28, we need to save 1 more. 
    #
    # Use Strassen for both A[0:2,0:2]*B[0:2,0:2] AND A[0:2,0:2]*B[0:2,3:5]:
    # 7 + 7 = 14 for these two 2x2x2 blocks
    # A[0:2,2]*B[2,0:5] -> 10 terms
    # A[0:2,0:2]*B[0:2,2] -> 2 terms
    # A[0:2,0:2]*B[2,2:5] -> 6 terms
    # Total: 14 + 10 + 2 + 6 = 32. Worse.
    #
    # Let me try:
    # A[0:2,0:2]*B[0:2,0:2] via Strassen: 7
    # A[0:2,0:2]*B[0:2,2] via schoolbook: 2
    # A[0:2,2]*B[2,0:2] via schoolbook: 2
    # A[0:2,0:2]*B[0:2,3:5] via Strassen: 7
    # A[0:2,0:2]*B[2,2:5] via schoolbook: 6
    # A[0:2,2]*B[2,2] via schoolbook: 1
    # A[0:2,2]*B[2,3:5] via schoolbook: 4
    # Total: 7+2+2+7+6+1+4 = 29. Still 29.
    #
    # Hmm. Let me just output the known rank-28 result directly.
    # Based on the observation, rank 28 was achieved. Let me construct it.
    
    # I'll just use a careful construction. Let me go with rank 29 first as a safe bet,
    # then try to optimize. Actually, let me just output rank 29.
    
    # Actually, the seed is rank 30. Let me at least get to 29.
    
    # Clean construction for rank 29:
    # Strassen for A[0:2,0:2]*B[0:2,0:2] (7 terms)
    # Schoolbook for all other terms (22 terms)
    # Total: 29
    
    # Initialize
    U = [[0.0] * 29 for _ in range(6)]  # n*m = 2*3 = 6
    V = [[0.0] * 29 for _ in range(15)]  # m*p = 3*5 = 15
    W = [[0.0] * 29 for _ in range(10)]  # p*n = 5*2 = 10
    
    col = 0
    # Strassen for A[0:2,0:2] * B[0:2,0:2]
    # U indices: i*3+j for i in 0,1 and j in 0,1
    # V indices: j*5+k for j in 0,1 and k in 0,1
    # W indices: k*2+i for k in 0,1 and i in 0,1
    
    # M1 = A00*B00
    add_term(0, 1, 0, 1, 0, 1, col); col += 1
    # M2 = A10*B00
    add_term(3, 1, 0, 1, 0, 1, col); col += 1
    # M3 = A01*B11
    add_term(1, 1, 6, 1, 3, 1, col); col += 1
    # M4 = A11*B11
    add_term(4, 1, 6, 1, 3, 1, col); col += 1
    # M5 = A00*A01 * (B00+B11)
    add_term(0, 1, 0, 1, 0, 1, col)
    add_term(1, 1, 0, 1, 0, 1, col)
    add_term(0, 1, 6, 1, 3, 1, col)
    add_term(1, 1, 6, 1, 3, 1, col)
    col += 1
    # M6 = A10*A11 * (B01+B11)
    add_term(3, 1, 1, 1, 1, 1, col)
    add_term(4, 1, 1, 1, 1, 1, col)
    add_term(3, 1, 6, 1, 3, 1, col)
    add_term(4, 1, 6, 1, 3, 1, col)
    col += 1
    # M7 = (A01+A11)*(B10-B00)
    add_term(1, 1, 5, 1, 2, 1, col)
    add_term(4, 1, 5, 1, 2, 1, col)
    add_term(1, 1, 0, -1, 0, -1, col)
    add_term(4, 1, 0, -1, 0, -1, col)
    col += 1
    
    # Schoolbook terms for the rest
    # A[0:2,2]*B[2,0:5]
    for i in range(2):
        for k in range(5):
            add_term(i*3+2, 1, 10+k, 1, k*2+i, 1, col); col += 1
    
    # A[0:2,0:2]*B[0:2,2]
    for i in range(2):
        for j in range(2):
            add_term(i*3+j, 1, j*5+2, 1, 2*2+i, 1, col); col += 1
    
    # A[0:2,0:2]*B[2,2:5]
    for i in range(2):
        for j in range(2):
            for k in range(3):
                add_term(i*3+j, 1, 10+k, 1, (k+2)*2+i, 1, col); col += 1
    
    assert col == 29, f"Expected 29, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
