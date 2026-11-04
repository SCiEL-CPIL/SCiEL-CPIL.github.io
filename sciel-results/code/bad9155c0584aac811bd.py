"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 36
    rank = 36
    
    # U shape: (n*m, rank) = (8, 36)
    # V shape: (m*p, rank) = (20, 36)
    # W shape: (p*n, rank) = (10, 36)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(u_idx, v_idx, w_idx, offset):
        """
        u_idx: [u00, u01, u10, u11]
        v_idx: [v00, v01, v10, v11]
        w_idx: [w00, w01, w10, w11]
        """
        # M1 = (A00 + A11)(B00 + B11)
        U[u_idx[0]][offset] += 1
        U[u_idx[3]][offset] += 1
        V[v_idx[0]][offset] += 1
        V[v_idx[3]][offset] += 1
        W[w_idx[0]][offset] += 1
        W[w_idx[3]][offset] += 1
        W[w_idx[1]][offset] -= 1
        W[w_idx[2]][offset] -= 1
        
        # M2 = (A10 + A11)B00
        U[u_idx[2]][offset+1] += 1
        U[u_idx[3]][offset+1] += 1
        V[v_idx[0]][offset+1] += 1
        W[w_idx[0]][offset+1] -= 1
        W[w_idx[2]][offset+1] += 1
        
        # M3 = A00 B10
        U[u_idx[0]][offset+2] += 1
        V[v_idx[2]][offset+2] += 1
        W[w_idx[1]][offset+2] += 1
        W[w_idx[2]][offset+2] += 1
        
        # M4 = A11 B01
        U[u_idx[3]][offset+3] += 1
        V[v_idx[1]][offset+3] += 1
        W[w_idx[0]][offset+3] += 1
        W[w_idx[2]][offset+3] += 1
        
        # M5 = (A00 + A01) B11
        U[u_idx[0]][offset+4] += 1
        U[u_idx[1]][offset+4] += 1
        V[v_idx[3]][offset+4] += 1
        W[w_idx[0]][offset+4] -= 1
        W[w_idx[1]][offset+4] += 1
        
        # M6 = (A10 - A00)(B01 + B11)
        U[u_idx[2]][offset+5] += 1
        U[u_idx[0]][offset+5] -= 1
        V[v_idx[1]][offset+5] += 1
        V[v_idx[3]][offset+5] += 1
        W[w_idx[3]][offset+5] += 1
        
        # M7 = (A01 - A11)(B00 + B10)
        U[u_idx[1]][offset+6] += 1
        U[u_idx[3]][offset+6] -= 1
        V[v_idx[0]][offset+6] += 1
        V[v_idx[2]][offset+6] += 1
        W[w_idx[0]][offset+6] += 1
        W[w_idx[1]][offset+6] += 1

    def add_schoolbook_2x2x1_block(u_idx, v_idx, w_idx, offset):
        """
        u_idx: [u00, u01, u10, u11]
        v_idx: [v0, v1]  # Corresponds to B[0,0], B[1,0]
        w_idx: [w0, w1]  # Corresponds to C[0,0], C[1,0]
        """
        # A00 B00 -> C00
        U[u_idx[0]][offset] += 1
        V[v_idx[0]][offset] += 1
        W[w_idx[0]][offset] += 1
        
        # A01 B10 -> C00
        U[u_idx[1]][offset+1] += 1
        V[v_idx[1]][offset+1] += 1
        W[w_idx[0]][offset+1] += 1
        
        # A10 B00 -> C10
        U[u_idx[2]][offset+2] += 1
        V[v_idx[0]][offset+2] += 1
        W[w_idx[1]][offset+2] += 1
        
        # A11 B10 -> C10
        U[u_idx[3]][offset+3] += 1
        V[v_idx[1]][offset+3] += 1
        W[w_idx[1]][offset+3] += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    u1 = [0, 1, 4, 5]
    v1 = [0, 1, 5, 6]
    w1 = [0, 1, 2, 3]
    add_strassen_block(u1, v1, w1, 0)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    u2 = [0, 1, 4, 5]
    v2 = [2, 3, 7, 8]
    w2 = [4, 5, 6, 7]
    add_strassen_block(u2, v2, w2, 7)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    u3 = [0, 1, 4, 5]
    v3 = [4, 9]
    w3 = [8, 9]
    add_schoolbook_2x2x1_block(u3, v3, w3, 14)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    u4 = [2, 3, 6, 7]
    v4 = [10, 11, 15, 16]
    w4 = [0, 1, 2, 3]
    add_strassen_block(u4, v4, w4, 18)
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    u5 = [2, 3, 6, 7]
    v5 = [12, 13, 17, 18]
    w5 = [4, 5, 6, 7]
    add_strassen_block(u5, v5, w5, 25)
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    u6 = [2, 3, 6, 7]
    v6 = [14, 19]
    w6 = [8, 9]
    add_schoolbook_2x2x1_block(u6, v6, w6, 32)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
