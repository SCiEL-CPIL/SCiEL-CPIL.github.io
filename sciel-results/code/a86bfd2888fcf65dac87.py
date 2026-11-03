"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=7
    # U shape (n*m, R) = (8, R)
    # V shape (m*p, R) = (28, R)
    # W shape (p*n, R) = (14, R)
    
    # Rank 50
    R = 50
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    def add_strassen_block(u_rows, v_rows, w_rows, start_col):
        # u_rows: x00, x01, x10, x11
        # v_rows: y00, y01, y10, y11
        # w_rows: z00, z01, z10, z11
        
        # m1 = (x00+x11)(y00+y11)
        U[u_rows[0]][start_col + 0] += 1
        U[u_rows[3]][start_col + 0] += 1
        V[v_rows[0]][start_col + 0] += 1
        V[v_rows[3]][start_col + 0] += 1
        W[w_rows[0]][start_col + 0] += 1
        W[w_rows[3]][start_col + 0] += 1
        
        # m2 = (x10+x11)y00
        U[u_rows[1]][start_col + 1] += 1
        U[u_rows[3]][start_col + 1] += 1
        V[v_rows[0]][start_col + 1] += 1
        W[w_rows[2]][start_col + 1] += 1
        
        # m3 = x00(y10-y11)
        U[u_rows[0]][start_col + 2] += 1
        V[v_rows[2]][start_col + 2] += 1
        V[v_rows[3]][start_col + 2] -= 1
        W[w_rows[1]][start_col + 2] += 1
        
        # m4 = x11(y01-y00)
        U[u_rows[3]][start_col + 3] += 1
        V[v_rows[1]][start_col + 3] += 1
        V[v_rows[0]][start_col + 3] -= 1
        W[w_rows[0]][start_col + 3] += 1
        W[w_rows[2]][start_col + 3] += 1
        
        # m5 = (x00+x01)y11
        U[u_rows[0]][start_col + 4] += 1
        U[u_rows[1]][start_col + 4] += 1
        V[v_rows[3]][start_col + 4] += 1
        W[w_rows[0]][start_col + 4] -= 1
        W[w_rows[1]][start_col + 4] += 1
        
        # m6 = (x10-x00)(y00+y01)
        U[u_rows[1]][start_col + 5] += 1
        U[u_rows[0]][start_col + 5] -= 1
        V[v_rows[0]][start_col + 5] += 1
        V[v_rows[1]][start_col + 5] += 1
        W[w_rows[3]][start_col + 5] += 1
        
        # m7 = (x01-x11)(y10+y11)
        U[u_rows[1]][start_col + 6] += 1
        U[u_rows[3]][start_col + 6] -= 1
        V[v_rows[2]][start_col + 6] += 1
        V[v_rows[3]][start_col + 6] += 1
        W[w_rows[0]][start_col + 6] += 1
        W[w_rows[3]][start_col + 6] += 1
        
        return start_col + 7

    def add_schoolbook_block_2x2x1(u_rows, v_rows, w_rows, start_col):
        # u_rows: x00, x01, x10, x11
        # v_rows: y0, y1
        # w_rows: z0, z1
        
        U[u_rows[0]][start_col + 0] += 1
        V[v_rows[0]][start_col + 0] += 1
        W[w_rows[0]][start_col + 0] += 1
        
        U[u_rows[1]][start_col + 1] += 1
        V[v_rows[1]][start_col + 1] += 1
        W[w_rows[0]][start_col + 1] += 1
        
        U[u_rows[2]][start_col + 2] += 1
        V[v_rows[0]][start_col + 2] += 1
        W[w_rows[1]][start_col + 2] += 1
        
        U[u_rows[3]][start_col + 3] += 1
        V[v_rows[1]][start_col + 3] += 1
        W[w_rows[1]][start_col + 3] += 1
        
        return start_col + 4

    for b in range(2):
        u_rows = [2*b, 2*b+1, 4+2*b, 4+2*b+1]
        
        for k in range(3):
            v_rows = [2*b*7 + 2*k, 2*b*7 + 2*k+1, (2*b+1)*7 + 2*k, (2*b+1)*7 + 2*k+1]
            w_rows = [4*k, 4*k+2, 4*k+1, 4*k+3]
            col = add_strassen_block(u_rows, v_rows, w_rows, col)
            
        v_rows = [2*b*7 + 6, (2*b+1)*7 + 6]
        w_rows = [12, 13]
        col = add_schoolbook_block_2x2x1(u_rows, v_rows, w_rows, col)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
