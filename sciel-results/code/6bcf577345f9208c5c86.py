"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 46 decomposition:
    # 1. Schoolbook for first 4 columns of B (2x4x4): Rank 32
    # 2. Strassen for last 2 columns of B (2x4x2 split into two 2x2x2): Rank 14
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Schoolbook for k in 0..3
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col is now 32
    
    # Strassen coefficients for 2x2x2
    # U_s, V_s, W_s are 2x2x7 or similar
    # i, j, k in 0..1
    # r in 0..6
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We need U[i,j,r], V[j,k,r], W[k,i,r]
    # A = [[a, b], [c, d]] -> indices (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B = [[e, f], [g, h]] -> indices (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    
    # r=0: M1
    # (a+d)(e+h) = ae + ah + de + dh
    # U: a(1,0,0), d(1,1,1) -> U[0,0]=1, U[1,1]=1
    # V: e(0,0,0), h(1,1,1) -> V[0,0]=1, V[1,1]=1
    # W: C11 += M1, C12 += 0, C21 += 0, C22 += M1
    # W[0,0]=1, W[1,1]=1
    
    # r=1: M2
    # (c+d)e = ce + de
    # U: c(1,0), d(1,1) -> U[1,0]=1, U[1,1]=1
    # V: e(0,0) -> V[0,0]=1
    # W: C21 += M2
    # W[0,1]=1 (k=0, i=1)
    
    # r=2: M3
    # a(f-h) = af - ah
    # U: a(0,0) -> U[0,0]=1
    # V: f(0,1)=1, h(1,1)=-1 -> V[0,1]=1, V[1,1]=-1
    # W: C12 += M3
    # W[1,0]=1 (k=1, i=0)
    
    # r=3: M4
    # d(g-c) = dg - dc
    # U: d(1,1) -> U[1,1]=1
    # V: g(1,0)=1, c(0,0)=-1 -> V[1,0]=1, V[0,0]=-1
    # W: C11 += M4, C21 += M4
    # W[0,0]=1, W[0,1]=1
    
    # r=4: M5
    # (a+b)h = ah + bh
    # U: a(0,0)=1, b(0,1)=1 -> U[0,0]=1, U[0,1]=1
    # V: h(1,1)=1 -> V[1,1]=1
    # W: C11 -= M5, C12 += M5
    # W[0,0]=-1, W[1,0]=1
    
    # r=5: M6
    # (c-a)(e+f) = ce + cf - ae - af
    # U: c(1,0)=1, a(0,0)=-1 -> U[1,0]=1, U[0,0]=-1
    # V: e(0,0)=1, f(0,1)=1 -> V[0,0]=1, V[0,1]=1
    # W: C22 += M6
    # W[1,1]=1
    
    # r=6: M7
    # (b-d)(g+h) = bg + bh - dg - dh
    # U: b(0,1)=1, d(1,1)=-1 -> U[0,1]=1, U[1,1]=-1
    # V: g(1,0)=1, h(1,1)=1 -> V[1,0]=1, V[1,1]=1
    # W: C11 += M7, C22 -= M7
    # W[0,0]=1, W[1,1]=-1

    # Helper to add Strassen block
    def add_strassen(offset_u, offset_v, offset_w, base_col):
        # offset_u: index offset in U for i*m+j
        # offset_v: index offset in V for j*p+k
        # offset_w: index offset in W for k*n+i
        # base_col: starting column in U, V, W
        
        # r=0
        U[0 + offset_u][base_col] += 1
        U[1 + offset_u][base_col] += 1
        V[0 + offset_v][base_col] += 1
        V[1 + offset_v + 1] += 1 # Wait, V index is j*p+k.
        # V[0,0] is j=0,k=0. V[1,1] is j=1,k=1.
        # In local 2x2, j=0,1 and k=0,1.
        # V local index: j*2+k.
        # V global index: (j_local + j_offset)*p + (k_local + k_offset).
        
        # Let's re-implement carefully with local to global mapping.
        
    # Local Strassen Matrices
    # U_local[2][2][7], V_local[2][2][7], W_local[2][2][7]
    U_local = [[[0]*7 for _ in range(2)] for _ in range(2)]
    V_local = [[[0]*7 for _ in range(2)] for _ in range(2)]
    W_local = [[[0]*7 for _ in range(2)] for _ in range(2)]
    
    # r=0
    U_local[0][0][0] = 1; U_local[1][1][0] = 1
    V_local[0][0][0] = 1; V_local[1][1][0] = 1
    W_local[0][0][0] = 1; W_local[1][1][0] = 1
    
    # r=1
    U_local[1][0][1] = 1; U_local[1][1][1] = 1
    V_local[0][0][1] = 1
    W_local[0][1][1] = 1
    
    # r=2
    U_local[0][0][2] = 1
    V_local[0][1][2] = 1; V_local[1][1][2] = -1
    W_local[1][0][2] = 1
    
    # r=3
    U_local[1][1][3] = 1
    V_local[1][0][3] = 1; V_local[0][0][3] = -1
    W_local[0][0][3] = 1; W_local[0][1][3] = 1
    
    # r=4
    U_local[0][0][4] = 1; U_local[0][1][4] = 1
    V_local[1][1][4] = 1
    W_local[0][0][4] = -1; W_local[1][0][4] = 1
    
    # r=5
    U_local[1][0][5] = 1; U_local[0][0][5] = -1
    V_local[0][0][5] = 1; V_local[0][1][5] = 1
    W_local[1][1][5] = 1
    
    # r=6
    U_local[0][1][6] = 1; U_local[1][1][6] = -1
    V_local[1][0][6] = 1; V_local[1][1][6] = 1
    W_local[0][0][6] = 1; W_local[1][1][6] = -1

    # Add Strassen for A_L (j=0,1) and B_2T (k=4,5)
    # j_global = j_local
    # k_global = k_local + 4
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    u_val = U_local[i][j][r]
                    v_val = V_local[j][k][r]
                    w_val = W_local[k][i][r]
                    
                    u_idx = i * m + j
                    v_idx = j * p + (k + 4)
                    w_idx = (k + 4) * n + i
                    col_idx = 32 + r
                    
                    U[u_idx][col_idx] += u_val
                    V[v_idx][col_idx] += v_val
                    W[w_idx][col_idx] += w_val

    # Add Strassen for A_R (j=2,3) and B_2B (k=4,5)
    # j_global = j_local + 2
    # k_global = k_local + 4
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for r in range(7):
                    u_val = U_local[i][j][r]
                    v_val = V_local[j][k][r]
                    w_val = W_local[k][i][r]
                    
                    u_idx = i * m + (j + 2)
                    v_idx = (j + 2) * p + (k + 4)
                    w_idx = (k + 4) * n + i
                    col_idx = 39 + r
                    
                    U[u_idx][col_idx] += u_val
                    V[v_idx][col_idx] += v_val
                    W[w_idx][col_idx] += w_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
