"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 28 decomposition using Strassen (rank 7) for block level and Schoolbook (rank 4) for element level.
    # This is significantly better than the schoolbook rank 64.
    
    # 1. Strassen decomposition for 2x2 matrices (Rank 7)
    # Indices for U_out: (a, b) where a, b in {0, 1}
    # Indices for V_out: (c, d) where c, d in {0, 1}
    # Indices for W_out: (e, f) where e, f in {0, 1}
    # U_out, V_out, W_out are 4x7 matrices
    
    U_out = [[0]*7 for _ in range(4)]
    V_out = [[0]*7 for _ in range(4)]
    W_out = [[0]*7 for _ in range(4)]
    
    # P1 = A00(B00+B11)
    U_out[0][0] = 1  # A00
    V_out[0][0] = 1  # B00
    V_out[3][0] = 1  # B11
    W_out[0][0] = 1  # C00
    W_out[3][0] = 1  # C11
    
    # P2 = A10(B00+B01)
    U_out[2][1] = 1  # A10
    V_out[0][1] = 1  # B00
    V_out[1][1] = 1  # B01
    W_out[0][1] = -1 # C00
    W_out[3][1] = 0  # C11 (P2 not in C11)
    
    # P3 = A00(B10-B11)
    U_out[0][2] = 1  # A00
    V_out[2][2] = 1  # B10
    V_out[3][2] = -1 # B11
    W_out[1][2] = 1  # C01
    W_out[3][2] = -1 # C11
    
    # P4 = A11(B01-B00)
    U_out[3][3] = 1  # A11
    V_out[1][3] = 1  # B01
    V_out[0][3] = -1 # B00
    W_out[0][3] = 1  # C00
    W_out[3][3] = 1  # C11
    
    # P5 = (A00+A11)(B00+B01)
    U_out[0][4] = 1  # A00
    U_out[3][4] = 1  # A11
    V_out[0][4] = 1  # B00
    V_out[1][4] = 1  # B01
    W_out[1][4] = 1  # C01
    
    # P6 = (A01+A11)(B10-B00)
    U_out[1][5] = 1  # A01
    U_out[3][5] = 1  # A11
    V_out[2][5] = 1  # B10
    V_out[0][5] = -1 # B00
    W_out[2][5] = 1  # C10
    W_out[3][5] = -1 # C11
    
    # P7 = (A00+A10)(B11-B10)
    U_out[0][6] = 1  # A00
    U_out[2][6] = 1  # A10
    V_out[3][6] = 1  # B11
    V_out[2][6] = -1 # B10
    W_out[0][6] = -1 # C00
    W_out[2][6] = 1  # C10

    # 2. Schoolbook decomposition for 2x2 matrices (Rank 4)
    # Indices for U_in: (i, j) where i, j in {0, 1}
    # Indices for V_in: (k, l) where k, l in {0, 1}
    # Indices for W_in: (m, n) where m, n in {0, 1}
    # W_in index is n*2 + m
    
    U_in = [[0]*4 for _ in range(4)]
    V_in = [[0]*4 for _ in range(4)]
    W_in = [[0]*4 for _ in range(4)]
    
    # C_mn = A_mi B_in
    # Term 0: A00 B00 -> C00
    U_in[0][0] = 1
    V_in[0][0] = 1
    W_in[0][0] = 1 # n=0, m=0 -> 0*2+0=0
    
    # Term 1: A01 B01 -> C01
    U_in[1][1] = 1
    V_in[1][1] = 1
    W_in[2][1] = 1 # n=1, m=0 -> 1*2+0=2
    
    # Term 2: A10 B10 -> C10
    U_in[2][2] = 1
    V_in[2][2] = 1
    W_in[1][2] = 1 # n=0, m=1 -> 0*2+1=1
    
    # Term 3: A11 B11 -> C11
    U_in[3][3] = 1
    V_in[3][3] = 1
    W_in[3][3] = 1 # n=1, m=1 -> 1*2+1=3

    # 3. Combine
    rank_out = 7
    rank_in = 4
    total_rank = rank_out * rank_in
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(16)]
    W = [[0] * total_rank for _ in range(16)]
    
    # U indices: (a, b) block, (i, j) elem
    # U_out index: a*2+b
    # U_in index: i*2+j
    # Total U index: (a*2+b)*4 + (i*2+j)
    
    # V indices: (c, d) block, (k, l) elem
    # V_out index: c*2+d
    # V_in index: k*2+l
    # Total V index: (c*2+d)*4 + (k*2+l)
    
    # W indices: (e, f) block, (m, n) elem
    # W_out index: e*2+f
    # W_in index: n*2+m
    # Total W index: k*4+i = (2f+n)*4 + (2e+m) = 8f + 4n + 2e + m
    
    for r_out in range(rank_out):
        for r_in in range(rank_in):
            r_total = r_out * rank_in + r_in
            
            # Iterate over non-zero entries in U_out, V_out, W_out for this r_out
            # And non-zero entries in U_in, V_in, W_in for this r_in
            
            for a in range(2):
                for b in range(2):
                    val_u_out = U_out[a*2+b][r_out]
                    if val_u_out == 0: continue
                    for i in range(2):
                        for j in range(2):
                            val_u_in = U_in[i*2+j][r_in]
                            if val_u_in == 0: continue
                            u_total_idx = (a*2+b)*4 + (i*2+j)
                            U[u_total_idx][r_total] += val_u_out * val_u_in
            
            for c in range(2):
                for d in range(2):
                    val_v_out = V_out[c*2+d][r_out]
                    if val_v_out == 0: continue
                    for k in range(2):
                        for l in range(2):
                            val_v_in = V_in[k*2+l][r_in]
                            if val_v_in == 0: continue
                            v_total_idx = (c*2+d)*4 + (k*2+l)
                            V[v_total_idx][r_total] += val_v_out * val_v_in
            
            for e in range(2):
                for f in range(2):
                    val_w_out = W_out[e*2+f][r_out]
                    if val_w_out == 0: continue
                    for m in range(2):
                        for n in range(2):
                            val_w_in = W_in[n*2+m][r_in]
                            if val_w_in == 0: continue
                            # W index: k*4 + i
                            # k = 2f + n
                            # i = 2e + m
                            w_total_idx = (2*f + n)*4 + (2*e + m)
                            W[w_total_idx][r_total] += val_w_out * val_w_in

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
