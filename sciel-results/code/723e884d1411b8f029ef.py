"""Matrix multiplication 4x5x5 with Strassen-based decomposition."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = 50  # 49 (Strassen 4x4x4) + 1 (outer product)
    
    U = [[0] * rank for _ in range(n * m)]  # 20 x 50
    V = [[0] * rank for _ in range(m * p)]  # 25 x 50
    W = [[0] * rank for _ in range(p * n)]  # 20 x 50
    
    col = 0
    
    # 2x2 Strassen pattern: 7 terms
    # Each term: (u_dict, v_dict, w_dict) where keys are (row, col) in 0..1
    strassen_2x2 = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # 4x4 Strassen block-level: 7 intermediate products
    # Each: (u_block_dict, v_block_dict, w_block_dict)
    # Keys are (block_row, block_col) in 0..1
    strassen_4x4_blocks = [
        # S1 = (A11+A22)*(B11+B22), contributes +1 to C11, +1 to C22
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # S2 = (A21+A22)*B11, contributes +1 to C21
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # S3 = A11*(B21-B22), contributes +1 to C12, +1 to C22
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # S4 = A22*(B12-B11), contributes +1 to C11
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1}),
        # S5 = (A11+A12)*B22, contributes -1 to C11, +1 to C12
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # S6 = (A21-A11)*(B11+B12), contributes +1 to C22
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # S7 = (A12-A22)*(B21+B22), contributes +1 to C11
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    # For each of the 7 block-level terms, apply 2x2 Strassen
    for u_block, v_block, w_block in strassen_4x4_blocks:
        for u_sub, v_sub, w_sub in strassen_2x2:
            # Combine: u_total[(br*2+sr, bc*2+sc)] = u_block[(br,bc)] * u_sub[(sr,sc)]
            u_total = {}
            for (br, bc), bc_coeff in u_block.items():
                for (sr, sc), sc_coeff in u_sub.items():
                    key = (br*2+sr, bc*2+sc)
                    u_total[key] = u_total.get(key, 0) + bc_coeff * sc_coeff
            
            v_total = {}
            for (br, bc), bc_coeff in v_block.items():
                for (sr, sc), sc_coeff in v_sub.items():
                    key = (br*2+sr, bc*2+sc)
                    v_total[key] = v_total.get(key, 0) + bc_coeff * sc_coeff
            
            w_total = {}
            for (br, bc), bc_coeff in w_block.items():
                for (sr, sc), sc_coeff in w_sub.items():
                    key = (br*2+sr, bc*2+sc)
                    w_total[key] = w_total.get(key, 0) + bc_coeff * sc_coeff
            
            # Add this rank-1 term
            for (i, j), coeff in u_total.items():
                U[i * m + j][col] += coeff
            
            for (j, k), coeff in v_total.items():
                V[j * p + k][col] += coeff
            
            for (k, i), coeff in w_total.items():
                W[k * n + i][col] += coeff
            
            col += 1
    
    # Add outer product term: a*b where a=A[:,4], b=B[4,:]
    for i in range(n):
        U[i * m + 4][col] = 1
    for j in range(p):
        V[4 * p + j][col] = 1
    for j in range(p):
        for i in range(n):
            W[j * n + i][col] = 1
    col += 1
    
    # Convert to int where possible
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
