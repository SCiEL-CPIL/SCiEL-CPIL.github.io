import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M1..M7
    # M1: (x00+x11)(y00+y11) -> z00+1, z11+1
    # M2: (x10+x11)y00 -> z10+1, z11-1
    # M3: x00(y01-y11) -> z01+1, z11+1
    # M4: x11(y10-y00) -> z00+1, z10+1
    # M5: (x00+x01)y11 -> z00-1, z01+1
    # M6: (x10-x00)(y00+y01) -> z11+1
    # M7: (x01-x11)(y10+y11) -> z00+1, z11+1
    
    # Define the 7 terms for Strassen
    # Each term is a list of (coeff, x_idx, y_idx) for U/V and (coeff, z_idx) for W
    # x_idx, y_idx are local 0,1,2,3
    # z_idx is local 0,1,2,3
    
    terms = [
        # M1
        [(1, 0, 0), (1, 3, 0), (1, 0, 3), (1, 3, 3)], # (x00+y11) * (y00+y11)
        # This expands to x00y00 + x00y11 + x11y00 + x11y11
        # But Strassen combines them. We need to represent the linear combinations in U and V.
        # U vector for M1: [1, 0, 0, 1] (coeffs for x00, x01, x10, x11)
        # V vector for M1: [1, 0, 0, 1]
        # W contributions: z00 += 1, z11 += 1
    ]
    
    # It is easier to define the U, V vectors directly for each of the 7 products.
    
    def add_strassen_instance(base_col, x_map, y_map, z_map):
        # x_map: dict local (0,1,2,3) -> global U index
        # y_map: dict local (0,1,2,3) -> global V index
        # z_map: dict local (0,1,2,3) -> global W index
        
        # M1
        c = base_col + 0
        # U: x00+x11
        U[x_map[0]][c] += 1
        U[x_map[3]][c] += 1
        # V: y00+y11
        V[y_map[0]][c] += 1
        V[y_map[3]][c] += 1
        # W: z00+1, z11+1
        W[z_map[0]][c] += 1
        W[z_map[3]][c] += 1
        
        # M2
        c = base_col + 1
        # U: x10+x11
        U[x_map[2]][c] += 1
        U[x_map[3]][c] += 1
        # V: y00
        V[y_map[0]][c] += 1
        # W: z10+1, z11-1
        W[z_map[2]][c] += 1
        W[z_map[3]][c] -= 1
        
        # M3
        c = base_col + 2
        # U: x00
        U[x_map[0]][c] += 1
        # V: y01-y11
        V[y_map[1]][c] += 1
        V[y_map[3]][c] -= 1
        # W: z01+1, z11+1
        W[z_map[1]][c] += 1
        W[z_map[3]][c] += 1
        
        # M4
        c = base_col + 3
        # U: x11
        U[x_map[3]][c] += 1
        # V: y10-y00
        V[y_map[2]][c] += 1
        V[y_map[0]][c] -= 1
        # W: z00+1, z10+1
        W[z_map[0]][c] += 1
        W[z_map[2]][c] += 1
        
        # M5
        c = base_col + 4
        # U: x00+x01
        U[x_map[0]][c] += 1
        U[x_map[1]][c] += 1
        # V: y11
        V[y_map[3]][c] += 1
        # W: z00-1, z01+1
        W[z_map[0]][c] -= 1
        W[z_map[1]][c] += 1
        
        # M6
        c = base_col + 5
        # U: x10-x00
        U[x_map[2]][c] += 1
        U[x_map[0]][c] -= 1
        # V: y00+y01
        V[y_map[0]][c] += 1
        V[y_map[1]][c] += 1
        # W: z11+1
        W[z_map[3]][c] += 1
        
        # M7
        c = base_col + 6
        # U: x01-x11
        U[x_map[1]][c] += 1
        U[x_map[3]][c] -= 1
        # V: y10+y11
        V[y_map[2]][c] += 1
        V[y_map[3]][c] += 1
        # W: z00+1, z11+1
        W[z_map[0]][c] += 1
        W[z_map[3]][c] += 1

    # Instance 1: A1, B_top_left, C1
    x_map_1 = {0: 0, 1: 1, 2: 4, 3: 5}
    y_map_1 = {0: 0, 1: 1, 2: 4, 3: 5}
    z_map_1 = {0: 0, 1: 2, 2: 1, 3: 3} # z00->W0, z01->W2, z10->W1, z11->W3
    
    # Instance 2: A1, B_top_right, C2
    x_map_2 = x_map_1
    y_map_2 = {0: 2, 1: 3, 2: 6, 3: 7}
    z_map_2 = {0: 4, 1: 6, 2: 5, 3: 7} # z00->W4, z01->W6, z10->W5, z11->W7
    
    # Instance 3: A2, B_bottom_left, C1
    x_map_3 = {0: 2, 1: 3, 2: 6, 3: 7}
    y_map_3 = {0: 8, 1: 9, 2: 12, 3: 13}
    z_map_3 = z_map_1
    
    # Instance 4: A2, B_bottom_right, C2
    x_map_4 = x_map_3
    y_map_4 = {0: 10, 1: 11, 2: 14, 3: 15}
    z_map_4 = z_map_2

    add_strassen_instance(0, x_map_1, y_map_1, z_map_1)
    add_strassen_instance(7, x_map_2, y_map_2, z_map_2)
    add_strassen_instance(14, x_map_3, y_map_3, z_map_3)
    add_strassen_instance(21, x_map_4, y_map_4, z_map_4)

    return {"U": U, "V": V, "W": W}
