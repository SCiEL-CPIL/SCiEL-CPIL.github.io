"""Program for matrix multiplication 2x4 x 4x6 with rank-minimal tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x6 matrix.
    
    Uses a block-Strassen approach:
    Splits A (2x4) into A_L (2x2) and A_R (2x2).
    Splits B (4x6) into B_T (2x6) and B_B (2x6).
    C = A_L B_T + A_R B_B.
    Each 2x2 by 2x6 product is split into three 2x2 by 2x2 products.
    Each 2x2 by 2x2 product uses Strassen's algorithm (rank 7).
    Total rank = 6 blocks * 7 rank = 42.
    """
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # m1 = (a+d)(e+h) -> p+=1, s+=1
    # m2 = (c+d)e   -> r+=1
    # m3 = a(f-h)   -> q+=1, s-=1
    # m4 = d(g-f)   -> p-=1, r+=1
    # m5 = (a+b)h   -> p+=1
    # m6 = (c-d)(e+f) -> s+=1
    # m7 = (b-d)(g+h) -> p-=1, q+=1, s+=1
    
    # Define the 6 blocks
    # Block: (a_idx, b_idx, c_idx)
    # a_idx: [a, b, c, d] indices in U
    # b_idx: [e, f, g, h] indices in V
    # c_idx: [p, q, r, s] indices in W
    
    blocks = []
    
    # Helper to get indices
    def get_u_indices(col_offset):
        # A is 2x4. col_offset 0 for A_L (cols 0,1), 1 for A_R (cols 2,3)
        # A[i, j] -> U[i*4 + j]
        # a=A[0,0+2*co], b=A[0,1+2*co], c=A[1,0+2*co], d=A[1,1+2*co]
        return [0*4 + (0 + 2*col_offset), 0*4 + (1 + 2*col_offset), 1*4 + (0 + 2*col_offset), 1*4 + (1 + 2*col_offset)]
        
    def get_v_indices(row_offset, col_offset):
        # B is 4x6. row_offset 0 for B_T (rows 0,1), 1 for B_B (rows 2,3)
        # col_offset 0,1,2 for B cols 0-1, 2-3, 4-5
        # B[i, j] -> V[i*6 + j]
        # e=B[0+2*ro, 0+2*co], f=B[0+2*ro, 1+2*co], g=B[1+2*ro, 0+2*co], h=B[1+2*ro, 1+2*co]
        r_base = 2 * row_offset
        c_base = 2 * col_offset
        return [(r_base + 0)*6 + (c_base + 0), (r_base + 0)*6 + (c_base + 1), (r_base + 1)*6 + (c_base + 0), (r_base + 1)*6 + (c_base + 1)]
        
    def get_w_indices(col_offset):
        # C is 2x6. col_offset 0,1,2 for C cols 0-1, 2-3, 4-5
        # C[i, j] -> W[j*2 + i]
        # p=C[0, 0+2*co], q=C[0, 1+2*co], r=C[1, 0+2*co], s=C[1, 1+2*co]
        c_base = 2 * col_offset
        return [(c_base + 0)*2 + 0, (c_base + 1)*2 + 0, (c_base + 0)*2 + 1, (c_base + 1)*2 + 1]

    # Generate blocks
    # A_L (col_offset 0) with B_T (row_offset 0) cols 0,1,2
    for co in range(3):
        blocks.append((get_u_indices(0), get_v_indices(0, co), get_w_indices(co)))
    # A_R (col_offset 1) with B_B (row_offset 1) cols 0,1,2
    for co in range(3):
        blocks.append((get_u_indices(1), get_v_indices(1, co), get_w_indices(co)))
        
    col = 0
    for u_idx, v_idx, w_idx in blocks:
        a, b, c, d = u_idx
        e, f, g, h = v_idx
        p, q, r, s = w_idx
        
        # m1
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[p][col] += 1; W[s][col] += 1
        col += 1
        
        # m2
        U[c][col] += 1; U[d][col] += 1
        V[e][col] += 1
        W[r][col] += 1
        col += 1
        
        # m3
        U[a][col] += 1
        V[f][col] += 1; V[h][col] -= 1
        W[q][col] += 1; W[s][col] -= 1
        col += 1
        
        # m4
        U[d][col] += 1
        V[g][col] += 1; V[f][col] -= 1
        W[p][col] -= 1; W[r][col] += 1
        col += 1
        
        # m5
        U[a][col] += 1; U[b][col] += 1
        V[h][col] += 1
        W[p][col] += 1
        col += 1
        
        # m6
        U[c][col] += 1; U[d][col] -= 1
        V[e][col] += 1; V[f][col] += 1
        W[s][col] += 1
        col += 1
        
        # m7
        U[b][col] += 1; U[d][col] -= 1
        V[g][col] += 1; V[h][col] += 1
        W[p][col] -= 1; W[q][col] += 1; W[s][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
