import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}

    # Improved rank 43
    # Block 1: Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: Strassen for i in {0,1}, j in {0,1}, k in {2,3}
    # Schoolbook for the rest
    
    rank = 43
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add Strassen block
    # k_offset: 0 or 2
    def add_strassen_block(k_offset, col_start):
        # A indices: 0, 1, 3, 4 (i,j in 0,1)
        # B indices: j*p + k
        # C indices: k*n + i
        
        # M1
        U[0][col_start+0] = 1; U[4][col_start+0] = 1
        V[0][col_start+0] = 1; V[6][col_start+0] = 1 # B[0,0], B[1,1] -> 0, 6
        W[k_offset*3+0][col_start+0] = 1 # C[0,0]
        W[k_offset*3+1][col_start+0] = 0 # C[0,1]
        W[(k_offset+1)*3+0][col_start+0] = 0 # C[1,0]
        W[(k_offset+1)*3+1][col_start+0] = 1 # C[1,1]
        
        # M2
        U[3][col_start+1] = 1; U[4][col_start+1] = 1
        V[0][col_start+1] = 1
        W[k_offset*3+0][col_start+1] = 0
        W[k_offset*3+1][col_start+1] = 0
        W[(k_offset+1)*3+0][col_start+1] = 1
        W[(k_offset+1)*3+1][col_start+1] = -1
        
        # M3
        U[0][col_start+2] = 1
        V[1][col_start+2] = 1; V[6][col_start+2] = -1 # B[0,1], B[1,1]
        W[k_offset*3+0][col_start+2] = 0
        W[k_offset*3+1][col_start+2] = 1
        W[(k_offset+1)*3+0][col_start+2] = 0
        W[(k_offset+1)*3+1][col_start+2] = 1
        
        # M4
        U[4][col_start+3] = 1
        V[5][col_start+3] = 1; V[0][col_start+3] = -1 # B[1,0], B[0,0]
        W[k_offset*3+0][col_start+3] = 1
        W[k_offset*3+1][col_start+3] = 0
        W[(k_offset+1)*3+0][col_start+3] = 1
        W[(k_offset+1)*3+1][col_start+3] = 0
        
        # M5
        U[0][col_start+4] = 1; U[1][col_start+4] = 1
        V[6][col_start+4] = 1
        W[k_offset*3+0][col_start+4] = -1
        W[k_offset*3+1][col_start+4] = 1
        W[(k_offset+1)*3+0][col_start+4] = 0
        W[(k_offset+1)*3+1][col_start+4] = 0
        
        # M6
        U[3][col_start+5] = 1; U[0][col_start+5] = -1
        V[0][col_start+5] = 1; V[1][col_start+5] = 1
        W[k_offset*3+0][col_start+5] = 0
        W[k_offset*3+1][col_start+5] = 0
        W[(k_offset+1)*3+0][col_start+5] = 0
        W[(k_offset+1)*3+1][col_start+5] = 1
        
        # M7
        U[1][col_start+6] = 1; U[4][col_start+6] = -1
        V[5][col_start+6] = 1; V[6][col_start+6] = 1
        W[k_offset*3+0][col_start+6] = 1
        W[k_offset*3+1][col_start+6] = 0
        W[(k_offset+1)*3+0][col_start+6] = 0
        W[(k_offset+1)*3+1][col_start+6] = 0

    # Add Block 1 (k=0,1)
    # B indices for k=0,1:
    # B[0,0]=0, B[0,1]=1
    # B[1,0]=5, B[1,1]=6
    # My helper assumes B[0,0]=0, B[0,1]=1, B[1,0]=5, B[1,1]=6.
    # This matches k_offset=0.
    add_strassen_block(0, 0)
    
    # Add Block 2 (k=2,3)
    # B indices for k=2,3:
    # B[0,2]=2, B[0,3]=3
    # B[1,2]=7, B[1,3]=8
    # The helper uses hardcoded indices 0,1,5,6 for B.
    # I need to modify the helper or pass indices.
    
    # Let's rewrite the helper to take B indices explicitly or adjust the logic.
    # Actually, I can just manually fill the second block or generalize.
    
    # For Block 2, k_offset=2.
    # B[0,0] in local coords -> B[0,2] global = 2
    # B[0,1] in local coords -> B[0,3] global = 3
    # B[1,0] in local coords -> B[1,2] global = 7
    # B[1,1] in local coords -> B[1,3] global = 8
    
    # C indices:
    # C[0,0] local -> C[0,2] global = 2*3+0 = 6? No, W index is k*n+i.
    # k=2, i=0 -> 6.
    # k=2, i=1 -> 7.
    # k=3, i=0 -> 9.
    # k=3, i=1 -> 10.
    
    # My helper used:
    # W[k_offset*3+0] -> C[0,0] of block.
    # If k_offset=2, W[6] is C[2,0] (k=2, i=0). Correct.
    
    # So I just need to fix the V indices in the helper for the second block.
    # I will write a specific block for the second one to be safe.
    
    # Block 2: cols 7-13
    # M1
    U[0][7] = 1; U[4][7] = 1
    V[2][7] = 1; V[8][7] = 1
    W[6][7] = 1; W[7][7] = 0; W[9][7] = 0; W[10][7] = 1
    
    # M2
    U[3][8] = 1; U[4][8] = 1
    V[2][8] = 1
    W[6][8] = 0; W[7][8] = 0; W[9][8] = 1; W[10][8] = -1
    
    # M3
    U[0][9] = 1
    V[3][9] = 1; V[8][9] = -1
    W[6][9] = 0; W[7][9] = 1; W[9][9] = 0; W[10][9] = 1
    
    # M4
    U[4][10] = 1
    V[7][10] = 1; V[2][10] = -1
    W[6][10] = 1; W[7][10] = 0; W[9][10] = 1; W[10][10] = 0
    
    # M5
    U[0][11] = 1; U[1][11] = 1
    V[8][11] = 1
    W[6][11] = -1; W[7][11] = 1; W[9][11] = 0; W[10][11] = 0
    
    # M6
    U[3][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[6][12] = 0; W[7][12] = 0; W[9][12] = 0; W[10][12] = 1
    
    # M7
    U[1][13] = 1; U[4][13] = -1
    V[7][13] = 1; V[8][13] = 1
    W[6][13] = 1; W[7][13] = 0; W[9][13] = 0; W[10][13] = 0

    # Schoolbook for remaining
    col = 14
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if covered by Strassen blocks
                if i < 2 and j < 2 and k < 4:
                    continue
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # col should be 43
    assert col == 43, f"Expected 43, got {col}"
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
