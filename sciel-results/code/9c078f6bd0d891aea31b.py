import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        # Local indices i, j, k in 0..1
        # Global i = i_offset + i
        # Global j = j_offset + j
        # Global k = k_offset + k
        
        # M1
        U[i_offset*3 + (j_offset+0)][col_start+0] += 1
        U[i_offset*3 + (j_offset+1)][col_start+0] += 1
        V[(j_offset+0)*5 + (k_offset+0)][col_start+0] += 1
        V[(j_offset+1)*5 + (k_offset+1)][col_start+0] += 1
        W[(k_offset+0)*2 + (i_offset+0)][col_start+0] += 1
        W[(k_offset+1)*2 + (i_offset+0)][col_start+0] += 1
        W[(k_offset+0)*2 + (i_offset+1)][col_start+0] += 1
        W[(k_offset+1)*2 + (i_offset+1)][col_start+0] += 1
        
        # M2
        U[i_offset*3 + (j_offset+0)][col_start+1] += 1
        U[i_offset*3 + (j_offset+1)][col_start+1] += 1
        V[(j_offset+0)*5 + (k_offset+0)][col_start+1] += 1
        W[(k_offset+0)*2 + (i_offset+1)][col_start+1] += 1
        W[(k_offset+1)*2 + (i_offset+1)][col_start+1] -= 1
        
        # M3
        U[i_offset*3 + (j_offset+0)][col_start+2] += 1
        V[(j_offset+1)*5 + (k_offset+0)][col_start+2] += 1
        V[(j_offset+1)*5 + (k_offset+1)][col_start+2] -= 1
        W[(k_offset+1)*2 + (i_offset+0)][col_start+2] += 1
        W[(k_offset+1)*2 + (i_offset+1)][col_start+2] += 1
        
        # M4
        U[i_offset*3 + (j_offset+1)][col_start+3] += 1
        V[(j_offset+0)*5 + (k_offset+1)][col_start+3] += 1
        V[(j_offset+0)*5 + (k_offset+0)][col_start+3] -= 1
        W[(k_offset+0)*2 + (i_offset+0)][col_start+3] += 1
        W[(k_offset+0)*2 + (i_offset+1)][col_start+3] += 1
        
        # M5
        U[i_offset*3 + (j_offset+0)][col_start+4] += 1
        U[i_offset*3 + (j_offset+1)][col_start+4] += 1
        V[(j_offset+1)*5 + (k_offset+1)][col_start+4] += 1
        W[(k_offset+0)*2 + (i_offset+0)][col_start+4] -= 1
        W[(k_offset+1)*2 + (i_offset+0)][col_start+4] += 1
        
        # M6
        U[i_offset*3 + (j_offset+1)][col_start+5] += 1
        U[i_offset*3 + (j_offset+0)][col_start+5] -= 1
        V[(j_offset+0)*5 + (k_offset+0)][col_start+5] += 1
        V[(j_offset+1)*5 + (k_offset+0)][col_start+5] += 1
        W[(k_offset+1)*2 + (i_offset+1)][col_start+5] += 1
        
        # M7
        U[i_offset*3 + (j_offset+1)][col_start+6] += 1
        U[i_offset*3 + (j_offset+1)][col_start+6] -= 1 # Wait, A01 - A11
        # Correction M7: U: {(0,1):1, (1,1):-1}
        # My indices: i=0, j=1 is (0,1). i=1, j=1 is (1,1).
        # So U[i_offset+0, j_offset+1] += 1
        # U[i_offset+1, j_offset+1] -= 1
        # I wrote U[i_offset*3 + (j_offset+1)] for both. That's wrong.
        
        # Let's rewrite add_strassen_block more carefully.
        
    # Re-implementing add_strassen_block correctly
    def add_strassen(i_off, j_off, k_off, c):
        # M1
        U[i_off*3 + j_off][c] += 1
        U[i_off*3 + j_off + 1][c] += 1
        V[(j_off)*5 + k_off][c] += 1
        V[(j_off+1)*5 + k_off + 1][c] += 1
        W[k_off*2 + i_off][c] += 1
        W[(k_off+1)*2 + i_off][c] += 1
        W[k_off*2 + i_off + 1][c] += 1
        W[(k_off+1)*2 + i_off + 1][c] += 1
        
        # M2
        U[i_off*3 + j_off][c+1] += 1
        U[i_off*3 + j_off + 1][c+1] += 1
        V[j_off*5 + k_off][c+1] += 1
        W[k_off*2 + i_off + 1][c+1] += 1
        W[(k_off+1)*2 + i_off + 1][c+1] -= 1
        
        # M3
        U[i_off*3 + j_off][c+2] += 1
        V[(j_off+1)*5 + k_off][c+2] += 1
        V[(j_off+1)*5 + k_off + 1][c+2] -= 1
        W[(k_off+1)*2 + i_off][c+2] += 1
        W[(k_off+1)*2 + i_off + 1][c+2] += 1
        
        # M4
        U[i_off*3 + j_off + 1][c+3] += 1
        V[j_off*5 + k_off + 1][c+3] += 1
        V[j_off*5 + k_off][c+3] -= 1
        W[k_off*2 + i_off][c+3] += 1
        W[k_off*2 + i_off + 1][c+3] += 1
        
        # M5
        U[i_off*3 + j_off][c+4] += 1
        U[i_off*3 + j_off + 1][c+4] += 1
        V[(j_off+1)*5 + k_off + 1][c+4] += 1
        W[k_off*2 + i_off][c+4] -= 1
        W[(k_off+1)*2 + i_off][c+4] += 1
        
        # M6
        U[i_off*3 + j_off + 1][c+5] += 1
        U[i_off*3 + j_off][c+5] -= 1
        V[j_off*5 + k_off][c+5] += 1
        V[(j_off+1)*5 + k_off][c+5] += 1
        W[(k_off+1)*2 + i_off + 1][c+5] += 1
        
        # M7
        U[i_off*3 + j_off + 1][c+6] += 1
        U[i_off*3 + j_off + 1][c+6] -= 1 # ERROR: i_off+0 vs i_off+1
        # M7: A01 - A11.
        # A01 is i=0, j=1. A11 is i=1, j=1.
        # So U[i_off*3 + (j_off+1)] += 1
        # U[(i_off+1)*3 + (j_off+1)] -= 1
        
        # Fix M7
        U[i_off*3 + j_off + 1][c+6] += 1
        U[(i_off+1)*3 + j_off + 1][c+6] -= 1
        V[(j_off+1)*5 + k_off][c+6] += 1
        V[(j_off+1)*5 + k_off + 1][c+6] += 1
        W[k_off*2 + i_off][c+6] += 1

    # Block 1: i_off=0, j_off=0, k_off=0
    add_strassen(0, 0, 0, 0)
    col = 7
    
    # Block 2: i_off=0, j_off=0, k_off=2
    add_strassen(0, 0, 2, 7)
    col = 14
    
    # Block 3: Schoolbook 2x2x1. i_off=0, j_off=0, k_off=4
    # C_i,4 = A_i,0 B_0,4 + A_i,1 B_1,4
    # Terms:
    # 1. i=0, j=0, k=4
    U[0*3+0][14] = 1
    V[0*5+4][14] = 1
    W[4*2+0][14] = 1
    # 2. i=0, j=1, k=4
    U[0*3+1][15] = 1
    V[1*5+4][15] = 1
    W[4*2+0][15] = 1
    # 3. i=1, j=0, k=4
    U[1*3+0][16] = 1
    V[0*5+4][16] = 1
    W[4*2+1][16] = 1
    # 4. i=1, j=1, k=4
    U[1*3+1][17] = 1
    V[1*5+4][17] = 1
    W[4*2+1][17] = 1
    col = 18
    
    # Block 4: Schoolbook 2x1x5. i_off=0, j_off=2, k_off=0
    # C_i,k = A_i,2 B_2,k
    # 10 terms
    for i in range(2):
        for k in range(5):
            U[i*3+2][col] = 1
            V[2*5+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
