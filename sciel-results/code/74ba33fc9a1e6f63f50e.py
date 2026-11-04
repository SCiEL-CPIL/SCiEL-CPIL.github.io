"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # We split the multiplication into 4 blocks of 2x2x2.
    # A is 2x4, split into A1 (2x2) and A2 (2x2) horizontally.
    # B is 4x4, split into B11, B12, B21, B22 (each 2x2).
    # C is 2x4, split into C1 (2x2) and C2 (2x2) horizontally.
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    
    # We use Strassen for each 2x2x2 block.
    # Strassen rank 7 for 2x2x2.
    # Total rank 4 * 7 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c+d)(f-g)
    # M7 = (a-c)(e+g)
    # C11 = M1 - M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6
    
    # We define the linear forms for U, V, W for one Strassen block.
    # U indices: a, b, c, d (0, 1, 2, 3 in local 2x2)
    # V indices: e, f, g, h (0, 1, 2, 3 in local 2x2)
    # W indices: C11, C12, C21, C22 (0, 1, 2, 3 in local 2x2)
    
    def add_strassen_block(u_offset, v_offset, w_offset, base_col):
        # u_offset: start index in U for this block's A part
        # v_offset: start index in V for this block's B part
        # w_offset: start index in W for this block's C part
        # base_col: start column in U, V, W
        
        # M1: (a+d)(e+h)
        U[u_offset+0][base_col+0] += 1
        U[u_offset+2][base_col+0] += 1
        V[v_offset+0][base_col+0] += 1
        V[v_offset+2][base_col+0] += 1
        W[w_offset+0][base_col+0] += 1
        W[w_offset+3][base_col+0] += 1
        
        # M2: (c+d)e
        U[u_offset+2][base_col+1] += 1
        U[u_offset+3][base_col+1] += 1
        V[v_offset+0][base_col+1] += 1
        W[w_offset+2][base_col+1] += 1
        
        # M3: a(g-h)
        U[u_offset+0][base_col+2] += 1
        V[v_offset+2][base_col+2] += 1
        V[v_offset+3][base_col+2] -= 1
        W[w_offset+1][base_col+2] += 1
        W[w_offset+3][base_col+2] -= 1
        
        # M4: d(f-e)
        U[u_offset+3][base_col+3] += 1
        V[v_offset+1][base_col+3] += 1
        V[v_offset+0][base_col+3] -= 1
        W[w_offset+0][base_col+3] -= 1
        W[w_offset+2][base_col+3] += 1
        
        # M5: (a+b)h
        U[u_offset+0][base_col+4] += 1
        U[u_offset+1][base_col+4] += 1
        V[v_offset+3][base_col+4] += 1
        W[w_offset+1][base_col+4] += 1
        
        # M6: (c+d)(f-g)
        U[u_offset+2][base_col+5] += 1
        U[u_offset+3][base_col+5] += 1
        V[v_offset+1][base_col+5] += 1
        V[v_offset+2][base_col+5] -= 1
        W[w_offset+3][base_col+5] += 1
        
        # M7: (a-c)(e+g)
        U[u_offset+0][base_col+6] += 1
        U[u_offset+2][base_col+6] -= 1
        V[v_offset+0][base_col+6] += 1
        V[v_offset+2][base_col+6] += 1
        W[w_offset+0][base_col+6] += 1
        W[w_offset+2][base_col+6] += 1

    # Block 1: A1 * B11 -> C1_part1
    # A1 is A[0:2, 0:2] -> U indices 0,1,2,3
    # B11 is B[0:2, 0:2] -> V indices 0,1,2,3
    # C1 is C[0:2, 0:2] -> W indices 0,1,2,3
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: A2 * B21 -> C1_part2
    # A2 is A[0:2, 2:4] -> U indices 4,5,6,7
    # B21 is B[2:4, 0:2] -> V indices 8,9,10,11
    # C1 is C[0:2, 0:2] -> W indices 0,1,2,3
    add_strassen_block(4, 8, 0, 7)
    
    # Block 3: A1 * B12 -> C2_part1
    # A1 is A[0:2, 0:2] -> U indices 0,1,2,3
    # B12 is B[0:2, 2:4] -> V indices 4,5,6,7
    # C2 is C[0:2, 2:4] -> W indices 4,5,6,7
    add_strassen_block(0, 4, 4, 14)
    
    # Block 4: A2 * B22 -> C2_part2
    # A2 is A[0:2, 2:4] -> U indices 4,5,6,7
    # B22 is B[2:4, 2:4] -> V indices 12,13,14,15
    # C2 is C[0:2, 2:4] -> W indices 4,5,6,7
    add_strassen_block(4, 12, 4, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
