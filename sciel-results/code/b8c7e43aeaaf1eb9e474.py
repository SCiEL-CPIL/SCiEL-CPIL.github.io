"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=4
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (12, rank)
    # W shape: (p*n, rank) = (8, rank)
    
    # Rank 22 decomposition
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen_block(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w01, w10, w11]
        
        # P1 = (A00+A11)(B00+B11)
        U[u_idx[0]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # P2 = (A10+A11)B00
        U[u_idx[2]][col] += 1
        U[u_idx[3]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[2]][col] += 1
        W[w_idx[3]][col] -= 1
        col += 1
        
        # P3 = A00(B01-B11)
        U[u_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        V[v_idx[3]][col] -= 1
        W[w_idx[1]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # P4 = A11(B10-B00)
        U[u_idx[3]][col] += 1
        V[v_idx[2]][col] += 1
        V[v_idx[0]][col] -= 1
        W[w_idx[0]][col] += 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # P5 = (A00+A01)B11
        U[u_idx[0]][col] += 1
        U[u_idx[1]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] -= 1
        col += 1
        
        # P6 = (A10-A00)(B00+B01)
        U[u_idx[2]][col] += 1
        U[u_idx[0]][col] -= 1
        V[v_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # P7 = (A01-A11)(B10+B11)
        U[u_idx[1]][col] += 1
        U[u_idx[3]][col] -= 1
        V[v_idx[2]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1

    # Helper to add a Schoolbook 2x1x4 block
    def add_schoolbook_block(u_idx, v_idx, w_idx):
        nonlocal col
        # u_idx: [u0, u1]
        # v_idx: [v0, v1, v2, v3]
        # w_idx: [w00, w01, w02, w03, w10, w11, w12, w13]
        
        # Q00 = A0 B0
        U[u_idx[0]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[0]][col] += 1
        col += 1
        
        # Q01 = A0 B1
        U[u_idx[0]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[1]][col] += 1
        col += 1
        
        # Q02 = A0 B2
        U[u_idx[0]][col] += 1
        V[v_idx[2]][col] += 1
        W[w_idx[2]][col] += 1
        col += 1
        
        # Q03 = A0 B3
        U[u_idx[0]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[3]][col] += 1
        col += 1
        
        # Q10 = A1 B0
        U[u_idx[1]][col] += 1
        V[v_idx[0]][col] += 1
        W[w_idx[4]][col] += 1
        col += 1
        
        # Q11 = A1 B1
        U[u_idx[1]][col] += 1
        V[v_idx[1]][col] += 1
        W[w_idx[5]][col] += 1
        col += 1
        
        # Q12 = A1 B2
        U[u_idx[1]][col] += 1
        V[v_idx[2]][col] += 1
        W[w_idx[6]][col] += 1
        col += 1
        
        # Q13 = A1 B3
        U[u_idx[1]][col] += 1
        V[v_idx[3]][col] += 1
        W[w_idx[7]][col] += 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # U indices for A[0:2, 0:2]
    u1 = [0*3+0, 0*3+1, 1*3+0, 1*3+1] # [0, 1, 3, 4]
    # V indices for B[0:2, 0:2]
    v1 = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # [0, 1, 4, 5]
    # W indices for C[0:2, 0:2]
    w1 = [0*2+0, 0*2+1, 1*2+0, 1*2+1] # [0, 1, 2, 3]
    
    add_strassen_block(u1, v1, w1)

    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # U indices same as Block 1
    u2 = u1
    # V indices for B[0:2, 2:4]
    v2 = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # [2, 3, 6, 7]
    # W indices for C[0:2, 2:4]
    w2 = [2*2+0, 2*2+1, 3*2+0, 3*2+1] # [4, 5, 6, 7]
    
    add_strassen_block(u2, v2, w2)

    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # U indices for A[0:2, 2:3]
    u3 = [0*3+2, 1*3+2] # [2, 5]
    # V indices for B[2:3, 0:4]
    v3 = [2*4+0, 2*4+1, 2*4+2, 2*4+3] # [8, 9, 10, 11]
    # W indices for C[0:2, 0:4]
    w3 = [0*2+0, 0*2+1, 0*2+2, 0*2+3, 1*2+0, 1*2+1, 1*2+2, 1*2+3] # [0, 1, 2, 3, 4, 5, 6, 7]
    
    add_schoolbook_block(u3, v3, w3)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
