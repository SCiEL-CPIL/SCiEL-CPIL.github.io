import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 42
    
    # Initialize factor matrices
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)
    # W has shape (p*n, rank) = (12, 42)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Helper to apply a specific Strassen term
    # term_id: 0..6 corresponding to M1..M7
    # col: current column index
    # A_idx: list of 4 indices [a, b, c, d] in U
    # B_idx: list of 4 indices [e, f, g, h] in V
    # C_idx: list of 4 indices [c11, c12, c21, c22] in W
    
    def apply_strassen_term(term_id, col, A_idx, B_idx, C_idx):
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx

        # Strassen's 7 multiplications for 2x2 matrices
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Results:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        if term_id == 0: # M1
            U[a][col] += 1
            U[d][col] += 1
            V[e][col] += 1
            V[h][col] += 1
            W[c11][col] += 1
            W[c22][col] += 1
        elif term_id == 1: # M2
            U[c][col] += 1
            U[d][col] += 1
            V[e][col] += 1
            W[c21][col] += 1
            W[c22][col] -= 1
        elif term_id == 2: # M3
            U[a][col] += 1
            V[f][col] += 1
            V[h][col] -= 1
            W[c12][col] += 1
            W[c22][col] += 1
        elif term_id == 3: # M4
            U[d][col] += 1
            V[g][col] += 1
            V[e][col] -= 1
            W[c11][col] += 1
            W[c21][col] += 1
        elif term_id == 4: # M5
            U[a][col] += 1
            U[b][col] += 1
            V[h][col] += 1
            W[c11][col] -= 1
            W[c12][col] += 1
        elif term_id == 5: # M6
            U[c][col] += 1
            U[a][col] -= 1
            V[e][col] += 1
            V[f][col] += 1
            W[c22][col] += 1
        elif term_id == 6: # M7
            U[b][col] += 1
            U[d][col] -= 1
            V[g][col] += 1
            V[h][col] += 1
            W[c11][col] += 1

    # Indices for A left block (cols 0,1)
    # A is 2x4. Rows 0,1.
    # a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=4, d=A[1,1]=5
    A_L = [0, 1, 4, 5]
    
    # Indices for A right block (cols 2,3)
    # a=A[0,2]=2, b=A[0,3]=3, c=A[1,2]=6, d=A[1,3]=7
    A_R = [2, 3, 6, 7]

    col = 0
    for k in range(3):
        # Column offset in B (4x6) and C (2x6)
        col_offset = 2 * k

        # Indices for B top block (rows 0,1)
        # B is 4x6.
        # e=B[0, col_offset]=col_offset
        # f=B[0, col_offset+1]=col_offset+1
        # g=B[1, col_offset]=6+col_offset
        # h=B[1, col_offset+1]=6+col_offset+1
        B_T = [col_offset, col_offset + 1, 6 + col_offset, 6 + col_offset + 1]
        
        # Indices for B bottom block (rows 2,3)
        # e=B[2, col_offset]=12+col_offset
        # f=B[2, col_offset+1]=12+col_offset+1
        # g=B[3, col_offset]=18+col_offset
        # h=B[3, col_offset+1]=18+col_offset+1
        B_B = [12 + col_offset, 12 + col_offset + 1, 18 + col_offset, 18 + col_offset + 1]

        # Indices for C block
        # C is 2x6. W[k*n+i, r] => W[row*n + col_idx, rank_idx]? 
        # Task says: W[k*n+i, r] where k is row of C? No, standard convention for tensor T[i,j,k] is often flattened.
        # Let's check the seed program logic.
        # Seed: C11 = C[0, 2k] -> 2k*2+0. 
        # My seed interpretation: W index = col_index_in_C * n + row_index_in_C? 
        # Seed code: `c11 = C[0, 2k] -> 2k*2+0`. C[0, 2k] is row 0, col 2k.
        # Formula used: `2k*2 + 0`. This is `col * n + row`.
        # Let's verify with c12 = C[0, 2k+1] -> (2k+1)*2+0. Yes, `col * n + row`.
        # c21 = C[1, 2k] -> 2k*2+1. Yes.
        # c22 = C[1, 2k+1] -> (2k+1)*2+1. Yes.
        
        C_idx = [
            (2 * k) * 2 + 0,         # c11: row 0, col 2k
            (2 * k + 1) * 2 + 0,     # c12: row 0, col 2k+1
            (2 * k) * 2 + 1,         # c21: row 1, col 2k
            (2 * k + 1) * 2 + 1      # c22: row 1, col 2k+1
        ]

        # Apply 7 Strassen terms for A_L @ B_T
        for t in range(7):
            apply_strassen_term(t, col, A_L, B_T, C_idx)
            col += 1

        # Apply 7 Strassen terms for A_R @ B_B
        for t in range(7):
            apply_strassen_term(t, col, A_R, B_B, C_idx)
            col += 1

    return {"U": U, "V": V, "W": W}
