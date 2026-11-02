"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using 3 Strassen blocks of rank 7
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a, b], [c, d]], B = [[x, y], [z, w]]
    # M1 = (a+b)(z+w)
    # M2 = (c+d)w
    # M3 = a(x-z)
    # M4 = d(y-w)
    # M5 = (a+b)y
    # M6 = (c-d)(x+z)
    # M7 = (b-d)(z+w)
    
    # U parts (coefficients for A)
    # M1: a=1, b=1, c=0, d=0
    # M2: a=0, b=0, c=1, d=1
    # M3: a=1, b=0, c=0, d=0
    # M4: a=0, b=0, c=0, d=1
    # M5: a=1, b=1, c=0, d=0
    # M6: a=0, b=0, c=1, d=-1
    # M7: a=0, b=1, c=0, d=-1
    
    U_parts = [
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    
    # V parts (coefficients for B)
    # M1: z=1, w=1, x=0, y=0
    # M2: w=1
    # M3: x=1, z=-1
    # M4: y=1, w=-1
    # M5: y=1
    # M6: x=1, z=1
    # M7: z=1, w=1
    
    # B indices in V for a block:
    # x is B[0, k0], y is B[0, k1]
    # z is B[1, k0], w is B[1, k1]
    
    V_parts = [
        [0, 0, 1, 1], # x, y, z, w
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, 1, 0, -1],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1]
    ]
    
    # W parts (coefficients for C)
    # C11 = M5 + M7 - M4 + M2
    # C12 = M1 + M4
    # C21 = M3 + M5
    # C22 = M1 - M3 + M6 + M4 ? (Assuming standard Strassen)
    
    # Let's define W contributions for each M to each C entry
    # M1: C12=1, C22=1
    # M2: C11=1
    # M3: C21=1, C22=-1
    # M4: C11=-1, C12=1, C22=1
    # M5: C11=1, C21=1
    # M6: C22=1
    # M7: C11=1
    
    W_parts = [
        [0, 1, 0, 1], # M1: C11=0, C12=1, C21=0, C22=1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [-1, 1, 0, 1],# M4
        [1, 0, 1, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    for b in range(3):
        col_offset = 2 * b
        for t in range(7):
            col_idx = 7 * b + t
            
            # Fill U
            for u_row in range(4):
                U[u_row][col_idx] = U_parts[t][u_row]
                
            # Fill V
            # x -> V[0][col_offset], y -> V[0][col_offset+1]
            # z -> V[1][col_offset], w -> V[1][col_offset+1]
            # V row index = j * p + k
            # x is j=0, k=col_offset
            # y is j=0, k=col_offset+1
            # z is j=1, k=col_offset
            # w is j=1, k=col_offset+1
            
            V[0 * 6 + col_offset][col_idx] = V_parts[t][0]
            V[0 * 6 + col_offset + 1][col_idx] = V_parts[t][1]
            V[1 * 6 + col_offset][col_idx] = V_parts[t][2]
            V[1 * 6 + col_offset + 1][col_idx] = V_parts[t][3]
            
            # Fill W
            # C11 -> W[col_offset * 2 + 0]
            # C12 -> W[(col_offset+1) * 2 + 0]
            # C21 -> W[col_offset * 2 + 1]
            # C22 -> W[(col_offset+1) * 2 + 1]
            
            W[col_offset * 2 + 0][col_idx] = W_parts[t][0]
            W[(col_offset + 1) * 2 + 0][col_idx] = W_parts[t][1]
            W[col_offset * 2 + 1][col_idx] = W_parts[t][2]
            W[(col_offset + 1) * 2 + 1][col_idx] = W_parts[t][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
