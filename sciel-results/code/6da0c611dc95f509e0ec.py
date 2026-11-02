"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 93
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_2x2_strassen(col, a_i, b_i, c_i, d_i, x_i, y_i, u_i, v_i, 
                         targets, 
                         sa=1, sb=1, sc=1, sd=1, sx=1, sy=1, su=1, sv=1):
        nonlocal U, V, W
        
        # S1: (a+d)(x+u)
        U[a_i][col] += sa
        U[d_i][col] += sd
        V[x_i][col] += sx
        V[u_i][col] += su
        for k_off, i_off, s in targets:
             if s != 0:
                 W[(0 + k_off)*n + (0 + i_off)][col] += s
                 W[(1 + k_off)*n + (1 + i_off)][col] += s
        col += 1
        
        # S2: (c+d)x
        U[c_i][col] += sc
        U[d_i][col] += sd
        V[x_i][col] += sx
        for k_off, i_off, s in targets:
             if s != 0:
                 W[(1 + k_off)*n + (0 + i_off)][col] += s
                 W[(1 + k_off)*n + (1 + i_off)][col] -= s
        col += 1
        
        # S3: a(y-v)
        U[a_i][col] += sa
        V[y_i][col] += sy
        V[v_i][col] -= sv
        for k_off, i_off, s in targets:
             if s != 0:
                 W[(0 + k_off)*n + (1 + i_off)][col] += s
                 W[(1 + k_off)*n + (1 + i_off)][col] += s
        col += 1
        
        # S4: d(u-x)
        U[d_i][col] += sd
        V[u_i][col] += su
        V[x_i][col] -= sx
        for k_off, i_off, s in targets:
             if s != 0:
                 W[(0 + k_off)*n + (0 + i_off)][col] += s
                 W[(1 + k_off)*n + (0 + i_off)][col] += s
        col += 1
        
        # S5: (a+b)v
        U[a_i][col] += sa
        U[b_i][col] += sb
        V[v_i][col] += sv
        for k_off, i_off, s in targets:
             if s != 0:
                 W[(0 + k_off)*n + (0 + i_off)][col] -= s
                 W[(0 + k_off)*n + (1 + i_off)][col] += s
        col += 1
        
        # S6: (c-a)(x+y)
        U[c_i][col] += sc
        U[a_i][col] -= sa
        V[x_i][col] += sx
        V[y_i][col] += sy
        for k_off, i_off, s in targets:
             if s != 0:
                 W[(1 + k_off)*n + (1 + i_off)][col] += s
        col += 1
        
        # S7: (b-d)(u+v)
        U[b_i][col] += sb
        U[d_i][col] -= sd
        V[u_i][col] += su
        V[v_i][col] += sv
        for k_off, i_off, s in targets:
             if s != 0:
                 W[(0 + k_off)*n + (0 + i_off)][col] += s
        col += 1
        
        return col

    # Part 1: 4x4 x 4x4
    # A blocks:
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    # B blocks (first 4 cols):
    B11 = [0, 1, 7, 8]
    B12 = [2, 3, 9, 10]
    B21 = [14, 15, 21, 22]
    B22 = [16, 17, 23, 24]
    
    # M1 = (A11+A22)(B11+B22)
    col = add_2x2_strassen(col, *A11, *A22, *B11, *B22, [(0,0,1), (2,2,1)])
    
    # M2 = (A21+A22)B11
    col = add_2x2_strassen(col, *A21, *A22, *B11, [0,0,0,0], [(2,0,1), (2,2,-1)], sx=1, sy=0, su=1, sv=0)
    # Note: B11 is x=0,y=1,u=7,v=8. My function expects x,y,u,v.
    # I passed *B11 which is 0,1,7,8.
    # The dummy [0,0,0,0] was for the second B block in the previous call signature.
    # My function signature is add_2x2_strassen(col, a_i, b_i, c_i, d_i, x_i, y_i, u_i, v_i, targets, ...)
    # It takes ONE A block and ONE B block.
    # So for M2, A is A21+A22, B is B11.
    # I need to pass the indices for A21+A22 and B11.
    # But my function takes 4 A indices and 4 B indices.
    # For A21+A22, the indices are A21[0], A21[1], A21[2], A21[3] and A22[0]...
    # Wait, the function assumes a single 2x2 block structure (a,b,c,d).
    # If I want to compute (A21+A22)B11, I cannot just pass A21 and A22 separately.
    # I need to pass the SUM.
    # But the function adds to U and V.
    # If I call it with A21 indices and then A22 indices, it will add them.
    # So I can call it twice? No, that would double the rank.
    # I need to pass the combined indices.
    # U[8] += 1, U[9] += 1, U[12] += 1, U[13] += 1 (for A21)
    # U[10] += 1, U[11] += 1, U[14] += 1, U[15] += 1 (for A22)
    # This is not a single 2x2 block.
    
    # I must modify the approach.
    # The Strassen algorithm for 4x4 works on 2x2 blocks.
    # M1 = (A11+A22)(B11+B22).
    # This is a product of two 2x2 matrices.
    # The function add_2x2_strassen expects one 2x2 A and one 2x2 B.
    # For M1, A is A11+A22. This is not a contiguous 2x2 block in U.
    # So I cannot use the function directly with *A11, *A22.
    
    # I need to create a wrapper that handles the sums.
    
    return {"U": U, "V": "W": W}
# EVOLVE-BLOCK-END
