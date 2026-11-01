"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        U[u_idx][w_idx] += coeff
        V[v_idx][w_idx] += coeff
        W[w_idx][w_idx] += coeff

    # Strassen for top-left 2x2 block (A[0:2, 0:2] * B[0:2, 0:2])
    # A indices: 0,1,3,4. B indices: 0,1,3,4.
    # C indices: 0,1,3,4. W indices: 3*k+i.
    # C[0,0] -> W[0], C[0,1] -> W[3], C[1,0] -> W[1], C[1,1] -> W[4]
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1); add_term(0, 4, 0, 1); add_term(3, 0, 0, 1); add_term(3, 4, 0, 1)
    # M2 = (A10+A11)B00
    add_term(3, 0, 1, 1); add_term(4, 0, 1, 1)
    # M3 = A00(B01-B11)
    add_term(0, 1, 2, 1); add_term(0, 4, 2, -1)
    # M4 = A11(B10-B00)
    add_term(4, 3, 3, 1); add_term(4, 0, 3, -1)
    # M5 = (A00+A01)B11
    add_term(0, 4, 4, 1); add_term(1, 4, 4, 1)
    # M6 = (A10-A00)(B00+B01)
    add_term(3, 0, 5, 1); add_term(3, 1, 5, 1); add_term(0, 0, 5, -1); add_term(0, 1, 5, -1)
    # M7 = (A01-A11)(B10+B11)
    add_term(1, 3, 6, 1); add_term(1, 4, 6, 1); add_term(4, 3, 6, -1); add_term(4, 4, 6, -1)
    
    # Combine for C[0,0] (W[0])
    W[0][0] = 1; W[0][3] = -1; W[0][4] = -1; W[0][6] = 1
    # Combine for C[0,1] (W[3])
    W[3][2] = 1; W[3][4] = 1
    # Combine for C[1,0] (W[1])
    W[1][1] = 1; W[1][3] = 1
    # Combine for C[1,1] (W[4])
    W[4][0] = 1; W[4][1] = -1; W[4][2] = -1; W[4][5] = 1
    
    # A2B3 (2x1 * 1x2) -> C[0:2, 0:2]
    # A[0,2]=2, A[1,2]=5. B[2,0]=6, B[2,1]=7.
    # C[0,0] += A[0,2]B[2,0] -> W[0]
    add_term(2, 6, 7, 1)
    # C[0,1] += A[0,2]B[2,1] -> W[3]
    add_term(2, 7, 8, 1)
    # C[1,0] += A[1,2]B[2,0] -> W[1]
    add_term(5, 6, 9, 1)
    # C[1,1] += A[1,2]B[2,1] -> W[4]
    add_term(5, 7, 10, 1)
    
    # A1B2 (2x2 * 2x1) -> C[0:2, 2]
    # A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4. B[0,2]=2, B[1,2]=5.
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] -> W[6] (3*2+0)
    add_term(0, 2, 11, 1)
    add_term(1, 5, 11, 1)
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] -> W[7] (3*2+1)
    add_term(3, 2, 12, 1)
    add_term(4, 5, 12, 1)
    
    # A2B4 (2x1 * 1x1) -> C[0:2, 2]
    # A[0,2]=2, A[1,2]=5. B[2,2]=8.
    # C[0,2] += A[0,2]B[2,2] -> W[6]
    add_term(2, 8, 13, 1)
    # C[1,2] += A[1,2]B[2,2] -> W[7]
    add_term(5, 8, 14, 1)
    
    # A3B1 (1x2 * 2x2) -> C[2, 0:2]
    # A[2,0]=6, A[2,1]=7. B[0,0]=0, B[0,1]=1, B[1,0]=3, B[1,1]=4.
    # C[2,0] = A[2,0]B[0,0] + A[2,1]B[1,0] -> W[2] (3*0+2)
    add_term(6, 0, 15, 1)
    add_term(7, 3, 15, 1)
    # C[2,1] = A[2,0]B[0,1] + A[2,1]B[1,1] -> W[5] (3*1+2)
    add_term(6, 1, 16, 1)
    add_term(7, 4, 16, 1)
    
    # A4B3 (1x1 * 1x2) -> C[2, 0:2]
    # A[2,2]=8. B[2,0]=6, B[2,1]=7.
    # C[2,0] += A[2,2]B[2,0] -> W[2]
    add_term(8, 6, 17, 1)
    # C[2,1] += A[2,2]B[2,1] -> W[5]
    add_term(8, 7, 18, 1)
    
    # A3B2 (1x2 * 2x1) -> C[2, 2]
    # A[2,0]=6, A[2,1]=7. B[0,2]=2, B[1,2]=5.
    # C[2,2] = A[2,0]B[0,2] + A[2,1]B[1,2] -> W[8] (3*2+2)
    add_term(6, 2, 19, 1)
    add_term(7, 5, 19, 1)
    
    # A4B4 (1x1 * 1x1) -> C[2, 2]
    # A[2,2]=8. B[2,2]=8.
    # C[2,2] += A[2,2]B[2,2] -> W[8]
    add_term(8, 8, 20, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
