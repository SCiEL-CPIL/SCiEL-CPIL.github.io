DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen parts for A[0:2, 0:2] * B[0:2, 0:6]
    # A indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # B blocks:
    # Block 0: Cols 0,1. B indices: 0(0,0), 1(0,1), 6(1,0), 7(1,1)
    # Block 1: Cols 2,3. B indices: 2(0,2), 3(0,3), 8(1,2), 9(1,3)
    # Block 2: Cols 4,5. B indices: 4(0,4), 5(0,5), 10(1,4), 11(1,5)
    # C blocks:
    # Block 0: Cols 0,1. C indices: 0(0,0), 1(0,1), 3(1,0), 4(1,1)
    # Block 1: Cols 2,3. C indices: 6(2,0), 7(2,1), 9(3,0), 10(3,1)
    # Block 2: Cols 4,5. C indices: 12(4,0), 13(4,1), 15(5,0), 16(5,1)
    
    a00, a01, a10, a11 = 0, 1, 3, 4
    
    for block in range(3):
        # B indices for this block
        b00 = block * 2
        b01 = block * 2 + 1
        b10 = 6 + block * 2
        b11 = 6 + block * 2 + 1
        
        # C indices for this block
        c00 = block * 6
        c01 = block * 6 + 1
        c10 = block * 6 + 3
        c11 = block * 6 + 4
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[a00][col] = 1; V[b00][col] = 1; W[c00][col] = 1; col += 1
        U[a11][col] = 1; V[b11][col] = 1; W[c00][col] = 1; col += 1
        # Wait, Strassen M1 is a single product of sums.
        # In tensor decomposition, (a+b)(c+d) = ac + ad + bc + bd.
        # This requires 4 rank-1 terms if we expand.
        # But Strassen uses 7 multiplications of linear combinations.
        # The "multiplication" is the rank-1 term.
        # So M1 is ONE rank-1 term where U has coeffs for a00,a11 and V has coeffs for b00,b11.
        # Yes, U[a00]=1, U[a11]=1, V[b00]=1, V[b11]=1.
        # This is 1 rank.
        
        # Let's restart the Strassen block logic correctly.
        # I overwrote col in the dummy add_term above? No, I didn't use add_term.
        # I need to be careful.
        
        # Let's reset col and use a cleaner structure.
        
    # Reset
    col = 0
    
    for block in range(3):
        b00 = block * 2
        b01 = block * 2 + 1
        b10 = 6 + block * 2
        b11 = 6 + block * 2 + 1
        
        c00 = block * 6
        c01 = block * 6 + 1
        c10 = block * 6 + 3
        c11 = block * 6 + 4
        
        # M1 = (a00 + a11) * (b00 + b11)
        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = 1
        col += 1
        
        # M2 = (a10 + a11) * b00
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] = 1
        col += 1
        
        # M3 = a00 * (b01 - b11)
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] = 1
        col += 1
        
        # M4 = a11 * (b10 - b00)
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] = 1; W[c10][col] = 1
        col += 1
        
        # M5 = (a00 + a01) * b11
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] = -1; W[c01][col] = 1
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01)
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] = 1
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11)
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] = 1; W[c11][col] = -1
        col += 1
        
    # Cross terms A[0:2, 2] * B[2, 0:6]
    # a02 (idx 2), a12 (idx 5)
    # b3k (idx 12+k)
    # c0k (idx 3k), c1k (idx 3k+1)
    
    for k in range(6):
        # a02 * b3k -> c0k
        U[2][col] = 1
        V[12+k][col] = 1
        W[3*k][col] = 1
        col += 1
        
        # a12 * b3k -> c1k
        U[5][col] = 1
        V[12+k][col] = 1
        W[3*k+1][col] = 1
        col += 1

    # Cross terms A[2, 0:2] * B[0:2, 0:6]
    # a20 (idx 6), a21 (idx 7)
    # b0k (idx k), b1k (idx 6+k)
    # c2k (idx 3k+2)
    
    for k in range(6):
        # a20 * b0k -> c2k
        U[6][col] = 1
        V[k][col] = 1
        W[3*k+2][col] = 1
        col += 1
        
        # a21 * b1k -> c2k
        U[7][col] = 1
        V[6+k][col] = 1
        W[3*k+2][col] = 1
        col += 1

    # A[2, 2] * B[2, 0:6]
    # a22 (idx 8)
    # b3k (idx 12+k)
    # c2k (idx 3k+2)
    
    for k in range(6):
        U[8][col] = 1
        V[12+k][col] = 1
        W[3*k+2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
