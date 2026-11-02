import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the U, V, W matrices by concatenating the blocks.
    # U is (25, r), V is (25, r), W is (25, r).
    
    # Block 1: 4x4 Strassen
    # Strassen for 2x2 has rank 7.
    # For 4x4, we apply Strassen to 2x2 blocks, so rank is 7*4 = 28? 
    # Wait, Strassen for 4x4 using 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    # Each Mi is a product of 2x2 matrices.
    # If we use schoolbook for 2x2, each Mi has rank 4.
    # Total rank for 4x4 is 7 * 4 = 28.
    
    # Let's implement the 4x4 Strassen decomposition.
    # We need to map the 4x4 indices to the 5x5 indices.
    # The 4x4 block corresponds to i in 0..3, j in 0..3, k in 0..3.
    
    # We will create a helper function to add a term to U, V, W.
    U = [[0] * 100 for _ in range(25)] # Pre-allocate with some slack
    V = [[0] * 100 for _ in range(25)]
    W = [[0] * 100 for _ in range(25)]
    
    col = 0
    
    # Helper to add a product term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Strassen for 4x4
    # A11: i in 0,1; j in 0,1
    # A12: i in 0,1; j in 2,3
    # A21: i in 2,3; j in 0,1
    # A22: i in 2,3; j in 2,3
    
    # M1 = (A11+A22)(B11+B22)
    # This requires 4 terms (schoolbook for 2x2)
    for i in range(2):
        for j in range(2):
            # A11 part
            add_term(i*5+j, j*5+(i+0), (i+0)*5+i, 1, 1, 1) # Wait, indices are tricky.
            # Let's use a more systematic approach.
            
    # Actually, let's just use the known rank 49 decomposition for 4x4.
    # Or simpler: use schoolbook for 4x4 (rank 64) and see if we can improve.
    # Rank 64 + 4 + 16 + 4 + 16 + 4 + 4 + 1 = 113.
    # 113 is better than 125.
    # Let's try rank 113 first. It's easier to implement.
    
    col = 0
    # 4x4 schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*5+j][col] = 1
                V[j*5+k][col] = 1
                W[k*5+i][col] = 1
                col += 1
                
    # A12 B21: i in 0..3, j in 4, k in 0..3
    # A12 is 4x1, B21 is 1x4.
    # C_ik = A_i4 B_4k
    for i in range(4):
        for k in range(4):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*5+i][col] = 1
            col += 1
            
    # A11 B12: i in 0..3, j in 0..3, k in 4
    # A11 is 4x4, B12 is 4x1.
    # C_i4 = sum_j A_ij B_j4
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*5+i][col] = 1
            col += 1
            
    # A12 B22: i in 0..3, j in 4, k in 4
    # C_i4 = A_i4 B_44
    for i in range(4):
        U[i*5+4][col] = 1
        V[4*5+4][col] = 1
        W[4*5+i][col] = 1
        col += 1
        
    # A21 B11: i in 4, j in 0..3, k in 0..3
    # C_4k = sum_j A_4j B_jk
    for j in range(4):
        for k in range(4):
            U[4*5+j][col] = 1
            V[j*5+k][col] = 1
            W[k*5+4][col] = 1
            col += 1
            
    # A22 B21: i in 4, j in 4, k in 0..3
    # C_4k = A_44 B_4k
    for k in range(4):
        U[4*5+4][col] = 1
        V[4*5+k][col] = 1
        W[k*5+4][col] = 1
        col += 1
        
    # A21 B12: i in 4, j in 0..3, k in 4
    # C_44 = sum_j A_4j B_j4
    for j in range(4):
        U[4*5+j][col] = 1
        V[j*5+4][col] = 1
        W[4*5+4][col] = 1
        col += 1
        
    # A22 B22: i in 4, j in 4, k in 4
    U[4*5+4][col] = 1
    V[4*5+4][col] = 1
    W[4*5+4][col] = 1
    col += 1
    
    # Trim the matrices to the actual rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
