"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a term
    def set_term(col, u_idx, u_val, v_idx, v_val, w_idx, w_val):
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        
    # Strassen terms for 2x2x2 subblock (i in 0,1; j in 0,1; k in 0,1)
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    
    # Term 1: (A00+A11)(B00+B11)
    # U: 0, 4 ; V: 0, 4 ; W: 0 (C00), 3 (C11)
    set_term(0, 0, 1, 0, 1, 0, 1)
    U[4][0] = 1
    V[4][0] = 1
    W[3][0] = 1
    
    # Term 2: (A10+A11)B00
    # U: 3, 4 ; V: 0 ; W: 1 (C10)
    set_term(1, 3, 1, 0, 1, 1, 1)
    U[4][1] = 1
    
    # Term 3: A00(B10-B11)
    # U: 0 ; V: 3 (1), 4 (-1) ; W: 2 (C01), 3 (C11)
    set_term(2, 0, 1, 3, 1, 2, 1)
    V[4][2] = -1
    W[3][2] = 1
    
    # Term 4: A11(B01-B00)
    # U: 4 ; V: 1 (1), 0 (-1) ; W: 0 (-1, C00), 1 (1, C10)
    set_term(3, 4, 1, 1, 1, 0, -1)
    V[0][3] = -1
    W[1][3] = 1
    
    # Term 5: (A00+A10)B11
    # U: 0, 3 ; V: 4 ; W: 0 (C00), 2 (C01)
    set_term(4, 0, 1, 4, 1, 0, 1)
    U[3][4] = 1
    W[2][4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    # U: 3 (1), 0 (-1) ; V: 0, 1 ; W: 3 (C11)
    set_term(5, 3, 1, 0, 1, 3, 1)
    U[0][5] = -1
    V[1][5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    # U: 1 (1), 4 (-1) ; V: 3, 4 ; W: 0 (C00)
    set_term(6, 1, 1, 3, 1, 0, 1)
    U[4][6] = -1
    V[4][6] = 1
    
    # Schoolbook terms for j=2 (6 terms)
    # (0,2,0), (0,2,1), (0,2,2), (1,2,0), (1,2,1), (1,2,2)
    # U idx: 2 (0*3+2), 5 (1*3+2)
    # V idx: 6 (2*3+0), 7 (2*3+1), 8 (2*3+2)
    # W idx: 0, 2, 4 for i=0; 1, 3, 5 for i=1
    
    set_term(7, 2, 1, 6, 1, 0, 1)
    set_term(8, 2, 1, 7, 1, 2, 1)
    set_term(9, 2, 1, 8, 1, 4, 1)
    set_term(10, 5, 1, 6, 1, 1, 1)
    set_term(11, 5, 1, 7, 1, 3, 1)
    set_term(12, 5, 1, 8, 1, 5, 1)
    
    # Schoolbook terms for k=2, j in 0,1 (4 terms)
    # (0,0,2), (0,1,2), (1,0,2), (1,1,2)
    # U idx: 0, 1, 3, 4
    # V idx: 2 (0*3+2), 5 (1*3+2)
    # W idx: 4 (2*2+0), 5 (2*2+1)
    
    set_term(13, 0, 1, 2, 1, 4, 1)
    set_term(14, 1, 1, 5, 1, 4, 1)
    set_term(15, 3, 1, 2, 1, 5, 1)
    set_term(16, 4, 1, 5, 1, 5, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
