"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def add_strassen(a_block, b_block, c_block, offset):
        # a_block: list of 4 indices (i*m+j) for A
        # b_block: list of 4 indices (j*p+k) for B
        # c_block: list of 4 indices (k*n+i) for C
        # offset: rank offset
        # Strassen formulas:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A11 + A22)B21
        # M3 = A11(B12 - B22)
        # M4 = A22(B11 + B22)
        # M5 = (A12 - A22)(B11 + B21)
        # M6 = (A11 - A12)(B21 + B22)
        # M7 = (A21 - A11)(B11 + B12)
        
        # Indices in local 2x2 block:
        # A: 00, 01, 10, 11
        # B: 00, 01, 10, 11
        # C: 00, 01, 10, 11
        
        a00, a01, a10, a11 = a_block
        b00, b01, b10, b11 = b_block
        c00, c01, c10, c11 = c_block
        
        # M1
        U[a00][offset] += 1
        U[a11][offset] += 1
        V[b00][offset] += 1
        V[b11][offset] += 1
        W[c00][offset] += 1
        W[c01][offset] -= 1
        W[c10][offset] -= 1
        W[c11][offset] += 1
        
        # M2
        U[a00][offset+1] += 1
        U[a11][offset+1] += 1
        V[b10][offset+1] += 1
        W[c10][offset+1] += 1
        W[c11][offset+1] -= 1
        
        # M3
        U[a00][offset+2] += 1
        V[b01][offset+2] += 1
        V[b11][offset+2] -= 1
        W[c01][offset+2] += 1
        W[c11][offset+2] += 1
        
        # M4
        U[a11][offset+3] += 1
        V[b00][offset+3] += 1
        V[b11][offset+3] += 1
        W[c00][offset+3] += 1
        W[c10][offset+3] += 1
        
        # M5
        U[a01][offset+4] += 1
        U[a11][offset+4] -= 1
        V[b00][offset+4] += 1
        V[b10][offset+4] += 1
        W[c00][offset+4] -= 1
        W[c01][offset+4] += 1
        
        # M6
        U[a00][offset+5] += 1
        U[a01][offset+5] -= 1
        V[b10][offset+5] += 1
        V[b11][offset+5] += 1
        W[c10][offset+5] += 1
        W[c11][offset+5] += 1
        
        # M7
        U[a10][offset+6] += 1
        U[a00][offset+6] -= 1
        V[b00][offset+6] += 1
        V[b01][offset+6] += 1
        W[c00][offset+6] += 1
        W[c10][offset+6] += 1

    # Define index helpers
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i

    # Block A11: rows 0,1; cols 0,1
    a11 = [u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1)]
    # Block A12: rows 0,1; cols 2,3
    a12 = [u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3)]
    
    # Block B11: rows 0,1; cols 0,1
    b11 = [v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1)]
    # Block B12: rows 0,1; cols 2,3
    b12 = [v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3)]
    # Block B21: rows 2,3; cols 0,1
    b21 = [v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1)]
    # Block B22: rows 2,3; cols 2,3
    b22 = [v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3)]
    
    # Block C00: rows 0,1; cols 0,1
    c00 = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]
    # Block C01: rows 0,1; cols 2,3
    c01 = [w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)]
    
    # Top-left 2x2 of C: A11*B11 + A12*B21
    # Wait, C is 3x4.
    # C[0,0] = A[0,0]B[0,0] + ...
    # C[0,1] = A[0,0]B[0,1] + ...
    # C[1,0] = A[1,0]B[0,0] + ...
    # C[1,1] = A[1,0]B[0,1] + ...
    
    # My c_block indices for Strassen output need to map to C indices.
    # For C00 (top-left 2x2 of C):
    c_top_left = [w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)]
    # For C01 (top-right 2x2 of C):
    c_top_right = [w_idx(2,0), w_idx(2,1), w_idx(3,0), w_idx(3,1)]
    
    # 1. A11 * B11 -> C_top_left
    add_strassen(a11, b11, c_top_left, 0)
    
    # 2. A12 * B21 -> C_top_left
    add_strassen(a12, b21, c_top_left, 7)
    
    # 3. A11 * B12 -> C_top_right
    add_strassen(a11, b12, c_top_right, 14)
    
    # 4. A12 * B22 -> C_top_right
    add_strassen(a12, b22, c_top_right, 21)
    
    # 5-8. Bottom row of C: A2 * B
    # A2 is row 2 of A: indices u_idx(2,0), u_idx(2,1), u_idx(2,2), u_idx(2,3)
    # B is 4x4.
    # C[2,0] = sum_j A[2,j] B[j,0]
    # C[2,1] = sum_j A[2,j] B[j,1]
    # C[2,2] = sum_j A[2,j] B[j,2]
    # C[2,3] = sum_j A[2,j] B[j,3]
    
    # Rank 28-31: C[2,0]
    for j in range(4):
        U[u_idx(2, j)][28+j] = 1
        V[v_idx(j, 0)][28+j] = 1
        W[w_idx(0, 2)][28+j] = 1
        
    # Rank 32-35: C[2,1]
    for j in range(4):
        U[u_idx(2, j)][32+j] = 1
        V[v_idx(j, 1)][32+j] = 1
        W[w_idx(1, 2)][32+j] = 1
        
    # Rank 36-39: C[2,2]
    for j in range(4):
        U[u_idx(2, j)][36+j] = 1
        V[v_idx(j, 2)][36+j] = 1
        W[w_idx(2, 2)][36+j] = 1
        
    # Rank 40-43: C[2,3]
    for j in range(4):
        U[u_idx(2, j)][40+j] = 1
        V[v_idx(j, 3)][40+j] = 1
        W[w_idx(3, 2)][40+j] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
