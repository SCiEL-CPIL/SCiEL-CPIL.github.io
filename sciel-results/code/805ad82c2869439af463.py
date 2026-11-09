"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition that exploits the Strassen algorithm for the 2x2x2 sub-block.
    # Partition:
    # n: 2 + 1
    # m: 2 + 3
    # p: 2 + 4
    
    # Total rank:
    # 1 block 2x2x2 -> rank 7 (Strassen)
    # 1 block 2x2x4 -> rank 16 (Schoolbook)
    # 1 block 2x3x2 -> rank 12 (Schoolbook)
    # 1 block 2x3x4 -> rank 24 (Schoolbook)
    # 1 block 1x2x2 -> rank 4 (Schoolbook)
    # 1 block 1x2x4 -> rank 8 (Schoolbook)
    # 1 block 1x3x2 -> rank 6 (Schoolbook)
    # 1 block 1x3x4 -> rank 12 (Schoolbook)
    # Total = 7 + 16 + 12 + 24 + 4 + 8 + 6 + 12 = 89
    
    rank = 89
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add schoolbook terms for a block
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Helper to add Strassen terms for a 2x2x2 block
    # i_start, j_start, k_start are the global starting indices
    def add_strassen(i_start, j_start, k_start):
        nonlocal col
        # Local indices 0,1 map to global i_start, i_start+1 etc.
        # U row index: (i_start + di) * m + (j_start + dj)
        # V row index: (j_start + dj) * p + (k_start + dk)
        # W row index: (k_start + dk) * n + (i_start + di)
        
        # M1 = (A11 + A22)(B11 + B22)
        # A11: i=i_start, j=j_start
        # A22: i=i_start+1, j=j_start+1
        # B11: j=j_start, k=k_start
        # B22: j=j_start+1, k=k_start+1
        
        U[i_start * m + j_start][col] = 1
        U[(i_start + 1) * m + (j_start + 1)][col] = 1
        V[j_start * p + k_start][col] = 1
        V[(j_start + 1) * p + (k_start + 1)][col] = 1
        W[k_start * n + i_start][col] = 1
        W[(k_start + 1) * n + (i_start + 1)][col] = 1
        col += 1
        
        # M2 = (A21 + A22)B11
        # A21: i=i_start+1, j=j_start
        # A22: i=i_start+1, j=j_start+1
        # B11: j=j_start, k=k_start
        
        U[(i_start + 1) * m + j_start][col] = 1
        U[(i_start + 1) * m + (j_start + 1)][col] = 1
        V[j_start * p + k_start][col] = 1
        W[k_start * n + (i_start + 1)][col] = 1
        col += 1
        
        # M3 = A11(B21 - B22)
        # A11: i=i_start, j=j_start
        # B21: j=j_start+1, k=k_start
        # B22: j=j_start+1, k=k_start+1
        
        U[i_start * m + j_start][col] = 1
        V[(j_start + 1) * p + k_start][col] = 1
        V[(j_start + 1) * p + (k_start + 1)][col] = -1
        W[k_start * n + i_start][col] = 1
        W[(k_start + 1) * n + i_start][col] = -1
        col += 1
        
        # M4 = A22(B12 - B11)
        # A22: i=i_start+1, j=j_start+1
        # B12: j=j_start, k=k_start+1
        # B11: j=j_start, k=k_start
        
        U[(i_start + 1) * m + (j_start + 1)][col] = 1
        V[j_start * p + (k_start + 1)][col] = 1
        V[j_start * p + k_start][col] = -1
        W[(k_start + 1) * n + (i_start + 1)][col] = 1
        W[k_start * n + (i_start + 1)][col] = -1
        col += 1
        
        # M5 = (A11 + A12)B22
        # A11: i=i_start, j=j_start
        # A12: i=i_start, j=j_start+1
        # B22: j=j_start+1, k=k_start+1
        
        U[i_start * m + j_start][col] = 1
        U[i_start * m + (j_start + 1)][col] = 1
        V[(j_start + 1) * p + (k_start + 1)][col] = 1
        W[(k_start + 1) * n + i_start][col] = 1
        col += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        # A21: i=i_start+1, j=j_start
        # A11: i=i_start, j=j_start
        # B11: j=j_start, k=k_start
        # B12: j=j_start, k=k_start+1
        
        U[(i_start + 1) * m + j_start][col] = 1
        U[i_start * m + j_start][col] = -1
        V[j_start * p + k_start][col] = 1
        V[j_start * p + (k_start + 1)][col] = 1
        W[k_start * n + (i_start + 1)][col] = 1
        W[(k_start + 1) * n + (i_start + 1)][col] = 1
        col += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        # A12: i=i_start, j=j_start+1
        # A22: i=i_start+1, j=j_start+1
        # B21: j=j_start+1, k=k_start
        # B22: j=j_start+1, k=k_start+1
        
        U[i_start * m + (j_start + 1)][col] = 1
        U[(i_start + 1) * m + (j_start + 1)][col] = -1
        V[(j_start + 1) * p + k_start][col] = 1
        V[(j_start + 1) * p + (k_start + 1)][col] = 1
        W[k_start * n + i_start][col] = 1
        W[(k_start + 1) * n + i_start][col] = 1
        col += 1

    # Block 1: 2x2x2 at (0,0,0) -> Strassen
    add_strassen(0, 0, 0)
    
    # Block 2: 2x2x4 at (0,0,2) -> Schoolbook
    # i in [0,2), j in [0,2), k in [2,6)
    add_schoolbook(0, 2, 0, 2, 2, 6)
    
    # Block 3: 2x3x2 at (0,2,0) -> Schoolbook
    # i in [0,2), j in [2,5), k in [0,2)
    add_schoolbook(0, 2, 2, 5, 0, 2)
    
    # Block 4: 2x3x4 at (0,2,2) -> Schoolbook
    # i in [0,2), j in [2,5), k in [2,6)
    add_schoolbook(0, 2, 2, 5, 2, 6)
    
    # Block 5: 1x2x2 at (2,0,0) -> Schoolbook
    # i in [2,3), j in [0,2), k in [0,2)
    add_schoolbook(2, 3, 0, 2, 0, 2)
    
    # Block 6: 1x2x4 at (2,0,2) -> Schoolbook
    # i in [2,3), j in [0,2), k in [2,6)
    add_schoolbook(2, 3, 0, 2, 2, 6)
    
    # Block 7: 1x3x2 at (2,2,0) -> Schoolbook
    # i in [2,3), j in [2,5), k in [0,2)
    add_schoolbook(2, 3, 2, 5, 0, 2)
    
    # Block 8: 1x3x4 at (2,2,2) -> Schoolbook
    # i in [2,3), j in [2,5), k in [2,6)
    add_schoolbook(2, 3, 2, 5, 2, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
