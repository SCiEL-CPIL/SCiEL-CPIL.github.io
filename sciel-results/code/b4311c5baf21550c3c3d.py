"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank) -> (8, 56)
    # V: (m*p, rank) -> (32, 56)
    # W: (p*n, rank) -> (16, 56)
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # M1..M7
    # X indices 0,1,2,3
    # Y indices 0,1,2,3
    # Z indices 0,1,2,3
    
    def add_strassen_block(col, x_map, y_map, z_map):
        # x_map: list of 4 global A indices
        # y_map: list of 4 global B indices
        # z_map: list of 4 global C indices
        
        # M1: (X0+X3)(Y0+Y3)
        U[x_map[0]][col] += 1
        U[x_map[3]][col] += 1
        V[y_map[0]][col] += 1
        V[y_map[3]][col] += 1
        W[z_map[0]][col] += 1 # Z0 += M1
        W[z_map[3]][col] += 1 # Z3 += M1
        
        # M2: (X2+X3)Y0
        U[x_map[2]][col] += 1
        U[x_map[3]][col] += 1
        V[y_map[0]][col] += 1
        W[z_map[2]][col] += 1 # Z2 += M2
        W[z_map[3]][col] -= 1 # Z3 -= M2
        
        # M3: X0(Y1-Y3)
        U[x_map[0]][col] += 1
        V[y_map[1]][col] += 1
        V[y_map[3]][col] -= 1
        W[z_map[1]][col] += 1 # Z1 += M3
        W[z_map[3]][col] += 1 # Z3 += M3
        
        # M4: X3(Y2-Y0)
        U[x_map[3]][col] += 1
        V[y_map[2]][col] += 1
        V[y_map[0]][col] -= 1
        W[z_map[0]][col] += 1 # Z0 += M4
        W[z_map[2]][col] += 1 # Z2 += M4
        
        # M5: (X0+X1)Y3
        U[x_map[0]][col] += 1
        U[x_map[1]][col] += 1
        V[y_map[3]][col] += 1
        W[z_map[0]][col] -= 1 # Z0 -= M5
        W[z_map[1]][col] += 1 # Z1 += M5
        
        # M6: (X2-X0)(Y0+Y1)
        U[x_map[2]][col] += 1
        U[x_map[0]][col] -= 1
        V[y_map[0]][col] += 1
        V[y_map[1]][col] += 1
        W[z_map[3]][col] += 1 # Z3 += M6
        
        # M7: (X1-X3)(Y2+Y3)
        U[x_map[1]][col] += 1
        U[x_map[3]][col] -= 1
        V[y_map[2]][col] += 1
        V[y_map[3]][col] += 1
        W[z_map[0]][col] += 1 # Z0 += M7

    col = 0
    
    # Block 1: A1 BL1 (Y1 part)
    # A1: 0,1,4,5
    # BL11: 0,1,8,9
    # Z: 0,1,2,3
    add_strassen_block(col, [0,1,4,5], [0,1,8,9], [0,1,2,3])
    col += 7
    
    # Block 2: A1 BL1 (Y2 part)
    # A1: 0,1,4,5
    # BL12: 2,3,10,11
    # Z: 4,5,6,7
    add_strassen_block(col, [0,1,4,5], [2,3,10,11], [4,5,6,7])
    col += 7
    
    # Block 3: A2 BL2 (Y1 part)
    # A2: 2,3,6,7
    # BL21: 16,17,24,25
    # Z: 0,1,2,3
    add_strassen_block(col, [2,3,6,7], [16,17,24,25], [0,1,2,3])
    col += 7
    
    # Block 4: A2 BL2 (Y2 part)
    # A2: 2,3,6,7
    # BL22: 18,19,26,27
    # Z: 4,5,6,7
    add_strassen_block(col, [2,3,6,7], [18,19,26,27], [4,5,6,7])
    col += 7
    
    # Block 5: A1 BR1 (Y1 part)
    # A1: 0,1,4,5
    # BR11: 4,5,12,13
    # Z: 8,9,10,11
    add_strassen_block(col, [0,1,4,5], [4,5,12,13], [8,9,10,11])
    col += 7
    
    # Block 6: A1 BR1 (Y2 part)
    # A1: 0,1,4,5
    # BR12: 6,7,14,15
    # Z: 12,13,14,15
    add_strassen_block(col, [0,1,4,5], [6,7,14,15], [12,13,14,15])
    col += 7
    
    # Block 7: A2 BR2 (Y1 part)
    # A2: 2,3,6,7
    # BR21: 20,21,28,29
    # Z: 8,9,10,11
    add_strassen_block(col, [2,3,6,7], [20,21,28,29], [8,9,10,11])
    col += 7
    
    # Block 8: A2 BR2 (Y2 part)
    # A2: 2,3,6,7
    # BR22: 22,23,30,31
    # Z: 12,13,14,15
    add_strassen_block(col, [2,3,6,7], [22,23,30,31], [12,13,14,15])
    col += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
