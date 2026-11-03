"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    def add_strassen_222(i0, j0_A, i0_B, j0_B, col):
        # A indices
        u_x00 = i0 * m + j0_A
        u_x01 = i0 * m + j0_A + 1
        u_x10 = (i0 + 1) * m + j0_A
        u_x11 = (i0 + 1) * m + j0_A + 1
        
        # B indices
        v_y00 = i0_B * p + j0_B
        v_y01 = i0_B * p + j0_B + 1
        v_y10 = (i0_B + 1) * p + j0_B
        v_y11 = (i0_B + 1) * p + j0_B + 1
        
        # C indices
        w_z00 = j0_B * n + i0
        w_z01 = (j0_B + 1) * n + i0
        w_z10 = j0_B * n + (i0 + 1)
        w_z11 = (j0_B + 1) * n + (i0 + 1)
        
        # M1: (x00 + x11)(y00 + y11)
        U[u_x00][col] += 1
        U[u_x11][col] += 1
        V[v_y00][col] += 1
        V[v_y11][col] += 1
        W[w_z00][col] += 1
        W[w_z11][col] += 1
        col += 1
        
        # M2: (x10 + x11)y00
        U[u_x10][col] += 1
        U[u_x11][col] += 1
        V[v_y00][col] += 1
        W[w_z10][col] += 1
        W[w_z11][col] += -1
        col += 1
        
        # M3: x00(y10 - y11)
        U[u_x00][col] += 1
        V[v_y10][col] += 1
        V[v_y11][col] += -1
        W[w_z01][col] += 1
        W[w_z11][col] += 1
        col += 1
        
        # M4: x11(y01 - y00)
        U[u_x11][col] += 1
        V[v_y01][col] += 1
        V[v_y00][col] += -1
        W[w_z00][col] += 1
        W[w_z10][col] += 1
        col += 1
        
        # M5: (x00 + x01)y11
        U[u_x00][col] += 1
        U[u_x01][col] += 1
        V[v_y11][col] += 1
        W[w_z00][col] += -1
        W[w_z01][col] += 1
        col += 1
        
        # M6: (x10 - x00)(y00 + y01)
        U[u_x10][col] += 1
        U[u_x00][col] += -1
        V[v_y00][col] += 1
        V[v_y01][col] += 1
        W[w_z11][col] += 1
        col += 1
        
        # M7: (x01 - x11)(y10 + y11)
        U[u_x01][col] += 1
        U[u_x11][col] += -1
        V[v_y10][col] += 1
        V[v_y11][col] += 1
        W[w_z00][col] += 1
        col += 1
        
        return col

    def add_schoolbook_221(i0, j0_A, i0_B, j0_B, col):
        # A indices
        u_x00 = i0 * m + j0_A
        u_x01 = i0 * m + j0_A + 1
        u_x10 = (i0 + 1) * m + j0_A
        u_x11 = (i0 + 1) * m + j0_A + 1
        
        # B indices (2x1, so only col j0_B)
        v_y00 = i0_B * p + j0_B
        v_y10 = (i0_B + 1) * p + j0_B
        
        # C indices (2x1, so only col j0_B)
        w_z00 = j0_B * n + i0
        w_z10 = j0_B * n + (i0 + 1)
        
        # T1: x00 y00 -> z00
        U[u_x00][col] += 1
        V[v_y00][col] += 1
        W[w_z00][col] += 1
        col += 1
        
        # T2: x01 y10 -> z00
        U[u_x01][col] += 1
        V[v_y10][col] += 1
        W[w_z00][col] += 1
        col += 1
        
        # T3: x10 y00 -> z10
        U[u_x10][col] += 1
        V[v_y00][col] += 1
        W[w_z10][col] += 1
        col += 1
        
        # T4: x11 y10 -> z10
        U[u_x11][col] += 1
        V[v_y10][col] += 1
        W[w_z10][col] += 1
        col += 1
        
        return col

    def add_223(i0, j0_A, i0_B, j0_B, col):
        # 2x2x3: Split B into 2x2 (cols j0_B, j0_B+1) and 2x1 (col j0_B+2)
        col = add_strassen_222(i0, j0_A, i0_B, j0_B, col)
        col = add_schoolbook_221(i0, j0_A, i0_B, j0_B + 2, col)
        return col

    def add_224(i0, j0_A, i0_B, j0_B, col):
        # 2x2x4: Split B into two 2x2 (cols j0_B..j0_B+1 and j0_B+2..j0_B+3)
        col = add_strassen_222(i0, j0_A, i0_B, j0_B, col)
        col = add_strassen_222(i0, j0_A, i0_B, j0_B + 2, col)
        return col

    def add_227(i0, j0_A, i0_B, j0_B, col):
        # 2x2x7: Split B into 2x3 (cols j0_B..j0_B+2) and 2x4 (cols j0_B+3..j0_B+6)
        col = add_223(i0, j0_A, i0_B, j0_B, col)
        col = add_224(i0, j0_A, i0_B, j0_B + 3, col)
        return col

    # Main decomposition
    # Block 1: A rows 0,1; A cols 0,1. B rows 0,1; B cols 0..6
    col = add_227(0, 0, 0, 0, col)
    
    # Block 2: A rows 0,1; A cols 2,3. B rows 2,3; B cols 0..6
    col = add_227(0, 2, 2, 0, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
