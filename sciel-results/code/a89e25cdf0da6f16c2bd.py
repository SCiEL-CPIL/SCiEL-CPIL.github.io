"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We will build lists of columns for U, V, W
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_222(i_start, j_start, k_start):
        # Local indices for X (A), Y (B), Z (C)
        # X: 2x2, Y: 2x2, Z: 2x2
        # Global indices
        # A: i in {i_start, i_start+1}, j in {j_start, j_start+1}
        # B: j in {j_start, j_start+1}, k in {k_start, k_start+1}
        # C: i in {i_start, i_start+1}, k in {k_start, k_start+1}
        
        # Helper to get global index
        def idx_A(i, j): return i * m + j
        def idx_B(j, k): return j * p + k
        def idx_C(k, i): return k * n + i
        
        # Terms
        # M1 = (X00+X11)(Y00+Y11)
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[idx_A(i_start, j_start)] += 1
        U1[idx_A(i_start+1, j_start+1)] += 1
        V1[idx_B(j_start, k_start)] += 1
        V1[idx_B(j_start+1, k_start+1)] += 1
        W1[idx_C(k_start, i_start)] += 1
        W1[idx_C(k_start+1, i_start+1)] += 1
        U_cols.append(U1); V_cols.append(V1); W_cols.append(W1)
        
        # M2 = (X10+X11)Y00
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[idx_A(i_start+1, j_start)] += 1
        U2[idx_A(i_start+1, j_start+1)] += 1
        V2[idx_B(j_start, k_start)] += 1
        W2[idx_C(k_start, i_start+1)] += 1
        W2[idx_C(k_start+1, i_start+1)] -= 1
        U_cols.append(U2); V_cols.append(V2); W_cols.append(W2)
        
        # M3 = X00(Y01-Y11)
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[idx_A(i_start, j_start)] += 1
        V3[idx_B(j_start, k_start+1)] += 1
        V3[idx_B(j_start+1, k_start+1)] -= 1
        W3[idx_C(k_start+1, i_start)] += 1
        W3[idx_C(k_start+1, i_start+1)] += 1
        U_cols.append(U3); V_cols.append(V3); W_cols.append(W3)
        
        # M4 = X11(Y10-Y00)
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[idx_A(i_start+1, j_start+1)] += 1
        V4[idx_B(j_start+1, k_start)] += 1
        V4[idx_B(j_start, k_start)] -= 1
        W4[idx_C(k_start, i_start)] += 1
        W4[idx_C(k_start, i_start+1)] += 1
        U_cols.append(U4); V_cols.append(V4); W_cols.append(W4)
        
        # M5 = (X00+X01)Y11
        U5 = [0]*(n*m); V5 = [0]*(m*p); W5 = [0]*(p*n)
        U5[idx_A(i_start, j_start)] += 1
        U5[idx_A(i_start, j_start+1)] += 1
        V5[idx_B(j_start+1, k_start+1)] += 1
        W5[idx_C(k_start, i_start)] -= 1
        W5[idx_C(k_start+1, i_start)] += 1
        U_cols.append(U5); V_cols.append(V5); W_cols.append(W5)
        
        # M6 = (X10-X00)(Y00+Y01)
        U6 = [0]*(n*m); V6 = [0]*(m*p); W6 = [0]*(p*n)
        U6[idx_A(i_start+1, j_start)] += 1
        U6[idx_A(i_start, j_start)] -= 1
        V6[idx_B(j_start, k_start)] += 1
        V6[idx_B(j_start, k_start+1)] += 1
        W6[idx_C(k_start+1, i_start+1)] += 1
        U_cols.append(U6); V_cols.append(V6); W_cols.append(W6)
        
        # M7 = (X01-X11)(Y10+Y11)
        U7 = [0]*(n*m); V7 = [0]*(m*p); W7 = [0]*(p*n)
        U7[idx_A(i_start, j_start+1)] += 1
        U7[idx_A(i_start+1, j_start+1)] -= 1
        V7[idx_B(j_start+1, k_start)] += 1
        V7[idx_B(j_start+1, k_start+1)] += 1
        W7[idx_C(k_start, i_start)] += 1
        U_cols.append(U7); V_cols.append(V7); W_cols.append(W7)

    def add_block_221(i_start, j_start, k_start):
        # X: 2x2, Y: 2x1, Z: 2x1
        def idx_A(i, j): return i * m + j
        def idx_B(j, k): return j * p + k
        def idx_C(k, i): return k * n + i
        
        # Z0 = X00*Y0 + X01*Y1
        # Z1 = X10*Y0 + X11*Y1
        
        # T1 = X00*Y0 -> Z0
        U1 = [0]*(n*m); V1 = [0]*(m*p); W1 = [0]*(p*n)
        U1[idx_A(i_start, j_start)] += 1
        V1[idx_B(j_start, k_start)] += 1
        W1[idx_C(k_start, i_start)] += 1
        U_cols.append(U1); V_cols.append(V1); W_cols.append(W1)
        
        # T2 = X01*Y1 -> Z0
        U2 = [0]*(n*m); V2 = [0]*(m*p); W2 = [0]*(p*n)
        U2[idx_A(i_start, j_start+1)] += 1
        V2[idx_B(j_start+1, k_start)] += 1
        W2[idx_C(k_start, i_start)] += 1
        U_cols.append(U2); V_cols.append(V2); W_cols.append(W2)
        
        # T3 = X10*Y0 -> Z1
        U3 = [0]*(n*m); V3 = [0]*(m*p); W3 = [0]*(p*n)
        U3[idx_A(i_start+1, j_start)] += 1
        V3[idx_B(j_start, k_start)] += 1
        W3[idx_C(k_start, i_start+1)] += 1
        U_cols.append(U3); V_cols.append(V3); W_cols.append(W3)
        
        # T4 = X11*Y1 -> Z1
        U4 = [0]*(n*m); V4 = [0]*(m*p); W4 = [0]*(p*n)
        U4[idx_A(i_start+1, j_start+1)] += 1
        V4[idx_B(j_start+1, k_start)] += 1
        W4[idx_C(k_start, i_start+1)] += 1
        U_cols.append(U4); V_cols.append(V4); W_cols.append(W4)

    # Add blocks
    # k=0,1 (j=0,1 and j=2,3)
    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 2, 0)
    # k=2,3 (j=0,1 and j=2,3)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 2, 2)
    # k=4,5 (j=0,1 and j=2,3)
    add_strassen_222(0, 0, 4)
    add_strassen_222(0, 2, 4)
    # k=6 (j=0,1 and j=2,3)
    add_block_221(0, 0, 6)
    add_block_221(0, 2, 6)
    
    rank = len(U_cols)
    U = [list(col) for col in zip(*U_cols)]
    V = [list(col) for col in zip(*V_cols)]
    W = [list(col) for col in zip(*W_cols)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
