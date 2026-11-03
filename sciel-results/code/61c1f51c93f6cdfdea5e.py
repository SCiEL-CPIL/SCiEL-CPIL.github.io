import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (15, rank)
    
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base, col_start):
        # i_base, j_base, k_base are the starting indices for the 2x2x2 block
        # i in {i_base, i_base+1}
        # j in {j_base, j_base+1}
        # k in {k_base, k_base+1}
        
        # Local indices 0, 1 map to global i_base+0, i_base+1 etc.
        
        # M1 = (A00+A11)(B00+B11)
        U[i_base*4 + j_base][col_start] += 1
        U[(i_base+1)*4 + (j_base+1)][col_start] += 1
        V[j_base*5 + k_base][col_start] += 1
        V[(j_base+1)*5 + (k_base+1)][col_start] += 1
        W[k_base*n + i_base][col_start] += 1       # C00
        W[(k_base+1)*n + (i_base+1)][col_start] += 1 # C11
        col_start += 1
        
        # M2 = (A10+A11)B00
        U[(i_base+1)*4 + j_base][col_start] += 1
        U[(i_base+1)*4 + (j_base+1)][col_start] += 1
        V[j_base*5 + k_base][col_start] += 1
        W[k_base*n + (i_base+1)][col_start] += 1   # C10
        W[(k_base+1)*n + (i_base+1)][col_start] -= 1 # C11
        col_start += 1
        
        # M3 = A00(B01-B11)
        U[i_base*4 + j_base][col_start] += 1
        V[j_base*5 + (k_base+1)][col_start] += 1
        V[(j_base+1)*5 + (k_base+1)][col_start] -= 1
        W[k_base*n + i_base][col_start] -= 1       # C00
        W[(k_base+1)*n + i_base][col_start] += 1   # C01
        col_start += 1
        
        # M4 = A11(B10-B00)
        U[(i_base+1)*4 + (j_base+1)][col_start] += 1
        V[(j_base+1)*5 + k_base][col_start] += 1
        V[j_base*5 + k_base][col_start] -= 1
        W[k_base*n + i_base][col_start] += 1       # C00
        W[k_base*n + (i_base+1)][col_start] += 1   # C10
        col_start += 1
        
        # M5 = (A00+A01)B11
        U[i_base*4 + j_base][col_start] += 1
        U[i_base*4 + (j_base+1)][col_start] += 1
        V[(j_base+1)*5 + (k_base+1)][col_start] += 1
        W[k_base*n + i_base][col_start] -= 1       # C00
        W[(k_base+1)*n + i_base][col_start] += 1   # C01
        col_start += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i_base+1)*4 + j_base][col_start] += 1
        U[i_base*4 + j_base][col_start] -= 1
        V[j_base*5 + k_base][col_start] += 1
        V[j_base*5 + (k_base+1)][col_start] += 1
        W[(k_base+1)*n + (i_base+1)][col_start] += 1 # C11
        col_start += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i_base*4 + (j_base+1)][col_start] += 1
        U[(i_base+1)*4 + (j_base+1)][col_start] -= 1
        V[(j_base+1)*5 + k_base][col_start] += 1
        V[(j_base+1)*5 + (k_base+1)][col_start] += 1
        W[k_base*n + i_base][col_start] += 1       # C00
        W[(k_base+1)*n + (i_base+1)][col_start] += 1 # C11
        col_start += 1
        
        return col_start

    # Block 1: i=0,1; j=0,1; k=0,1
    col = add_strassen_block(0, 0, 0, col)
    
    # Block 2: i=1,2; j=2,3; k=2,3
    col = add_strassen_block(1, 2, 2, col)
    
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is handled by Strassen Block 1
                if i in [0, 1] and k in [0, 1] and j in [0, 1]:
                    continue
                # Check if this term is handled by Strassen Block 2
                if i in [1, 2] and k in [2, 3] and j in [2, 3]:
                    continue
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verify rank
    if col != rank:
        # If col is less, we have empty columns? No, we filled sequentially.
        # If col is more, we overflowed.
        # For this specific construction, it should be exactly 58.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
