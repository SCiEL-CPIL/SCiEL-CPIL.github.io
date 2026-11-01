"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2 blocks
    # A indices: u = 2*i + j
    # B indices: v = 14*j + k
    # C indices: w = 2*k + i
    
    # We have 7 blocks of 2x2 multiplication
    # Block q covers k = 2q and k = 2q+1
    for q in range(7):
        k0 = 2 * q
        k1 = 2 * q + 1
        
        # B indices
        v_B00 = 14 * 0 + k0  # b_0, k0
        v_B01 = 14 * 0 + k1  # b_0, k1
        v_B10 = 14 * 1 + k0  # b_1, k0
        v_B11 = 14 * 1 + k1  # b_1, k1
        
        # C indices
        w_C00 = 2 * k0 + 0   # c_0, k0
        w_C01 = 2 * k1 + 0   # c_0, k1
        w_C10 = 2 * k0 + 1   # c_1, k0
        w_C11 = 2 * k1 + 1   # c_1, k1
        
        # A indices
        u_A00 = 2 * 0 + 0    # a_00
        u_A01 = 2 * 0 + 1    # a_01
        u_A10 = 2 * 1 + 0    # a_10
        u_A11 = 2 * 1 + 1    # a_11
        
        col_base = q * 7
        
        # M1 = (A00 + A11)(B00 + B11)
        c = col_base + 0
        U[u_A00][c] += 1
        U[u_A11][c] += 1
        V[v_B00][c] += 1
        V[v_B11][c] += 1
        W[w_C00][c] += 1
        W[w_C11][c] += 1
        
        # M2 = (A10 + A11)B00
        c = col_base + 1
        U[u_A10][c] += 1
        U[u_A11][c] += 1
        V[v_B00][c] += 1
        W[w_C10][c] += 1
        W[w_C11][c] -= 1
        
        # M3 = A00(B01 - B11)
        c = col_base + 2
        U[u_A00][c] += 1
        V[v_B01][c] += 1
        V[v_B11][c] -= 1
        W[w_C01][c] += 1
        W[w_C11][c] += 1
        
        # M4 = A11(B10 - B00)
        c = col_base + 3
        U[u_A11][c] += 1
        V[v_B10][c] += 1
        V[v_B00][c] -= 1
        W[w_C00][c] += 1
        W[w_C10][c] += 1
        
        # M5 = (A00 + A01)B11
        c = col_base + 4
        U[u_A00][c] += 1
        U[u_A01][c] += 1
        V[v_B11][c] += 1
        W[w_C00][c] -= 1
        W[w_C01][c] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        c = col_base + 5
        U[u_A10][c] += 1
        U[u_A00][c] -= 1
        V[v_B00][c] += 1
        V[v_B01][c] += 1
        W[w_C11][c] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        c = col_base + 6
        U[u_A01][c] += 1
        U[u_A11][c] -= 1
        V[v_B10][c] += 1
        V[v_B11][c] += 1
        W[w_C00][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
