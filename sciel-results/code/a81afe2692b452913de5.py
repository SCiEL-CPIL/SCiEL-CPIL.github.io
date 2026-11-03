"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # Part 1: C1 = A1 B (1x6)
    # A1 is row 0 of A.
    # B is 4x6.
    for j in range(4):
        for k in range(6):
            # U index: A[0, j] -> 0*4 + j = j
            U[j][c] = 1
            # V index: B[j, k] -> j*6 + k
            V[j * 6 + k][c] = 1
            # W index: C[0, k] -> k*3 + 0
            W[k * 3][c] = 1
            c += 1
            
    # Part 2: C2 = A2 B (2x6)
    # A2 is rows 1,2 of A.
    # Split A2 into A21 (cols 0,1) and A22 (cols 2,3).
    # Split B into B1 (rows 0,1) and B2 (rows 2,3).
    # C2 = A21 B1 + A22 B2.
    
    # Helper to add Strassen terms for 2x2 multiplication
    # A_idx: [a00, a01, a10, a11]
    # B_idx: [b00, b01, b10, b11]
    # W_idx: [w00, w01, w10, w11]
    def add_strassen(A_idx, B_idx, W_idx):
        nonlocal c
        # P1 = (a00+a11)(b00+b11)
        U[A_idx[0]][c] += 1
        U[A_idx[3]][c] += 1
        V[B_idx[0]][c] += 1
        V[B_idx[3]][c] += 1
        W[W_idx[0]][c] += 1 # c00
        W[W_idx[3]][c] += 1 # c11
        c += 1
        
        # P2 = (a10+a11)b00
        U[A_idx[2]][c] += 1
        U[A_idx[3]][c] += 1
        V[B_idx[0]][c] += 1
        W[W_idx[2]][c] += 1 # c10
        W[W_idx[3]][c] -= 1 # c11
        c += 1
        
        # P3 = a00(b01-b11)
        U[A_idx[0]][c] += 1
        V[B_idx[1]][c] += 1
        V[B_idx[3]][c] -= 1
        W[W_idx[1]][c] += 1 # c01
        W[W_idx[3]][c] += 1 # c11
        c += 1
        
        # P4 = a11(b10-b00)
        U[A_idx[3]][c] += 1
        V[B_idx[2]][c] += 1
        V[B_idx[0]][c] -= 1
        W[W_idx[0]][c] += 1 # c00
        W[W_idx[2]][c] += 1 # c10
        c += 1
        
        # P5 = (a00+a01)b11
        U[A_idx[0]][c] += 1
        U[A_idx[1]][c] += 1
        V[B_idx[3]][c] += 1
        W[W_idx[0]][c] -= 1 # c00
        W[W_idx[1]][c] += 1 # c01
        c += 1
        
        # P6 = (a10-a00)(b00+b01)
        U[A_idx[2]][c] += 1
        U[A_idx[0]][c] -= 1
        V[B_idx[0]][c] += 1
        V[B_idx[1]][c] += 1
        W[W_idx[3]][c] += 1 # c11
        c += 1
        
        # P7 = (a01-a11)(b10+b11)
        U[A_idx[1]][c] += 1
        U[A_idx[3]][c] -= 1
        V[B_idx[2]][c] += 1
        V[B_idx[3]][c] += 1
        W[W_idx[0]][c] += 1 # c00
        c += 1

    # A21 indices in U:
    A21_idx = [4, 5, 8, 9]
    
    # A22 indices in U:
    A22_idx = [6, 7, 10, 11]
    
    for b in range(3):
        # B1 block b
        B1_idx = [2*b, 2*b+1, 6+2*b, 7+2*b]
        W_idx = [6*b+1, 6*b+4, 6*b+2, 6*b+5]
        add_strassen(A21_idx, B1_idx, W_idx)
        
        # B2 block b
        B2_idx = [12+2*b, 13+2*b, 18+2*b, 19+2*b]
        add_strassen(A22_idx, B2_idx, W_idx)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
