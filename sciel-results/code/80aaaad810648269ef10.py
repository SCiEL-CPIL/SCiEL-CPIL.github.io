import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 31
    rank = 31
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    def add_strassen(i_offset, j_offset, k_offset):
        # i_offset: row offset in A and C
        # j_offset: col offset in A, row offset in B
        # k_offset: col offset in B and C
        # Local A is 2x2, B is 2x2, C is 2x2
        # A indices: (i_offset+i, j_offset+j)
        # B indices: (j_offset+j, k_offset+k)
        # C indices: (k_offset+k, i_offset+i)
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11, A22
        # V: B11, B22
        # W: C11, C22
        add_term(i_offset*m + j_offset, (j_offset)*p + k_offset, k_offset*n + i_offset, 1, 1, 1)
        add_term((i_offset+1)*m + (j_offset+1), (j_offset+1)*p + (k_offset+1), (k_offset+1)*n + (i_offset+1), 1, 1, 1)
        # This is wrong. add_term adds to a single column.
        # Strassen M1 requires two non-zero entries in U, two in V, two in W.
        # But add_term creates a new column.
        # We need to combine them into one column.
        
        # Let's rewrite add_term to accept lists or just handle it manually.
        pass

    # It is easier to manually construct the columns for each block.
    
    # Helper to add a Strassen block
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # M1
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        W[(k_off)*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = -1
        col += 1
        
        # M3 = A11(B12-B22)
        U[i_off*m + j_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[(k_off+1)*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M4 = A22(B21-B11)
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[j_off*p + k_off][col] = -1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M5 = (A11+A12)B22
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = -1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[(i_off+1)*m + j_off][col] = 1
        U[i_off*m + j_off][col] = -1
        V[j_off*p + k_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[i_off*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1

    # Helper to add schoolbook 2x2x2x1 (A 2x2, B 2x1)
    def add_2x2_2x1(i_off, j_off, k_off):
        nonlocal col
        # C1 = A11 B1 + A12 B2
        # C2 = A21 B1 + A22 B2
        # Term 1: A11 B1 -> C1
        U[i_off*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + i_off][col] = 1
        col += 1
        
        # Term 2: A12 B2 -> C1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[k_off*n + i_off][col] = 1
        col += 1
        
        # Term 3: A21 B1 -> C2
        U[(i_off+1)*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 4: A22 B2 -> C2
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1

    # Helper to add 2x1x1x2 (A 2x1, B 1x2)
    def add_2x1_1x2(i_off, j_off, k_off):
        nonlocal col
        # C1 = A1 B1 + A1 B2? No.
        # A is 2x1 (col j_off). B is 1x2 (row j_off, cols k_off, k_off+1).
        # C is 2x2.
        # C11 = A1 B1
        # C12 = A1 B2
        # C21 = A2 B1
        # C22 = A2 B2
        # Term 1: A1 B1 -> C11
        U[i_off*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + i_off][col] = 1
        col += 1
        
        # Term 2: A1 B2 -> C12
        U[i_off*m + j_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # Term 3: A2 B1 -> C21
        U[(i_off+1)*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1
        
        # Term 4: A2 B2 -> C22
        U[(i_off+1)*m + j_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # Wait, rank of 2x1x1x2 is 2, not 4.
        # u v^T = u1 e1 v^T + u2 e2 v^T.
        # Term 1: u1 (A1) * v^T (B1, B2).
        # U: A1=1. V: B1=1, B2=1. W: C11=1, C12=1.
        # Term 2: u2 (A2) * v^T (B1, B2).
        # U: A2=1. V: B1=1, B2=1. W: C21=1, C22=1.
        
        # Let's rewrite add_2x1_1x2
        pass

    # Correct add_2x1_1x2
    def add_2x1_1x2_correct(i_off, j_off, k_off):
        nonlocal col
        # Term 1: A1 * (B1, B2)
        U[i_off*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # Term 2: A2 * (B1, B2)
        U[(i_off+1)*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1

    # Helper to add 2x1x1x3 (A 2x1, B 1x3)
    def add_2x1_1x3(i_off, j_off, k_off):
        nonlocal col
        # Term 1: A1 * (B1, B2, B3)
        U[i_off*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[j_off*p + (k_off+2)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        W[(k_off+2)*n + i_off][col] = 1
        col += 1
        
        # Term 2: A2 * (B1, B2, B3)
        U[(i_off+1)*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[j_off*p + (k_off+2)][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        W[(k_off+2)*n + (i_off+1)][col] = 1
        col += 1

    # Block 1: k in {0, 1}
    add_strassen_block(0, 0, 0) # 7
    add_2x1_1x2_correct(0, 2, 0) # 2
    
    # Block 2: k in {2, 3}
    add_strassen_block(0, 0, 2) # 7
    add_2x1_1x2_correct(0, 2, 2) # 2
    
    # Block 3: k in {4, 5, 6}
    add_strassen_block(0, 0, 4) # 7 (for k 4,5)
    add_2x2_2x1(0, 0, 6) # 4 (for k 6)
    add_2x1_1x3(0, 2, 4) # 2 (for k 4,5,6)
    
    return {"U": U, "V": V, "W": W}
