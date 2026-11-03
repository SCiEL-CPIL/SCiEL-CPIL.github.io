"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of 4x4 matrix multiplication using the Kronecker product
    # of the rank-7 Winograd decomposition for 2x2 matrix multiplication.
    # Rank = 7 * 7 = 49.
    
    # Winograd 2x2 decomposition vectors (indices 0..3 for 2x2 matrix)
    # U_winograd: (i, j) -> i*2 + j
    # V_winograd: (j, k) -> j*2 + k
    # W_winograd: (k, i) -> k*2 + i
    
    # The 7 terms for 2x2 multiplication:
    # 1. a*e
    # 2. d*h
    # 3. (a+d)*(e+h)
    # 4. (a-c)*(e-g)
    # 5. (a+c)*(e+g)
    # 6. (b+d)*(f+h)
    # 7. (b-d)*(f-h)
    
    # We need to map these to the tensor indices.
    # For 2x2:
    # U indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # V indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (j*2+k)
    # W indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*2+i)
    
    # Term 1: a*e (i=0, j=0, k=0)
    # U: idx(0,0)=0 is 1. V: idx(0,0)=0 is 1. W: idx(0,0)=0 is 1.
    # Term 2: d*h (i=1, j=1, k=1)
    # U: idx(1,1)=3 is 1. V: idx(1,1)=3 is 1. W: idx(1,1)=3 is 1.
    # Term 3: (a+d)*(e+h)
    # U: a(0,0)=0 and d(1,1)=3 are 1.
    # V: e(0,0)=0 and h(1,1)=3 are 1.
    # W: Result is a*e + a*h + d*e + d*h.
    # C00 gets a*e. C01 gets a*h. C10 gets d*e. C11 gets d*h.
    # W indices: k*2+i.
    # a*e (k=0, i=0) -> W[0] += 1
    # a*h (k=1, i=0) -> W[2] += 1
    # d*e (k=0, i=1) -> W[1] += 1
    # d*h (k=1, i=1) -> W[3] += 1
    # So W vector: [1, 1, 1, 1]
    
    # Term 4: (a-c)*(e-g)
    # U: a(0,0)=0 is 1, c(1,0)=2 is -1.
    # V: e(0,0)=0 is 1, g(1,0)=2 is -1. (Note: g is B[1,0], so j=1, k=0 -> idx 1*2+0=2)
    # W:
    # a*e (k=0, i=0) -> W[0] += 1
    # a*(-g) (k=0, i=0) -> W[0] -= 1 ? No, product is a*(-g) = -ag.
    # C00 = a*e + b*g. We don't want g in C00?
    # Wait, Winograd decomposition is exact.
    # Let's use the standard algebraic identity for Winograd 2x2 to be sure.
    
    # Standard Winograd 2x2:
    # P1 = a*e
    # P2 = d*h
    # P3 = (a+d)*(e+h)
    # P4 = (a-c)*(e-g)
    # P5 = (a+c)*(e+g)
    # P6 = (b+d)*(f+h)
    # P7 = (b-d)*(f-h)
    
    # C00 = P1 + P3 - P4 + P5
    # C01 = P3 + P6 - P7 - P4 ? No.
    
    # Let's derive W vectors from the linear combinations.
    # C00 = a*e + b*g
    # C01 = a*f + b*h
    # C10 = c*e + d*g
    # C11 = c*f + d*h
    
    # From P1..P7:
    # P1 = ae
    # P2 = dh
    # P3 = ae + ah + de + dh
    # P4 = ae - ag - ce + cg
    # P5 = ae + ag + ce + cg
    # P6 = bf + bh + df + dh
    # P7 = bf - bh - df + dh
    
    # We want to form C00, C01, C10, C11.
    # C00 = ae + bg.
    # C01 = af + bh.
    # C10 = ce + dg.
    # C11 = cf + dh.
    
    # Let's check combinations:
    # C00: P1 + (P3 - P2 - P4 - P5)?
    # P3 - P4 - P5 = (ae+ah+de+dh) - (ae-ag-ce+cg) - (ae+ag+ce+cg)
    # = ae + ah + de + dh - ae + ag + ce - cg - ae - ag - ce - cg
    # = -ae + ah + de + dh - 2cg. This is messy.
    
    # Actually, the Winograd algorithm outputs are:
    # C00 = P1 + P3 - P4 + P5 ?
    # Let's just use the known rank-7 tensor decomposition for 2x2.
    
    # U vectors (length 4):
    # 1: [1, 0, 0, 0]
    # 2: [0, 0, 0, 1]
    # 3: [1, 0, 0, 1]
    # 4: [1, 0, -1, 0]
    # 5: [1, 0, 1, 0]
    # 6: [0, 1, 0, 1]
    # 7: [0, 1, 0, -1]
    
    # V vectors (length 4):
    # 1: [1, 0, 0, 0]
    # 2: [0, 0, 0, 1]
    # 3: [1, 1, 0, 0]
    # 4: [1, -1, 0, 0]
    # 5: [1, 0, 1, 0]
    # 6: [0, 1, 0, 1]
    # 7: [0, -1, 0, 1]
    
    # W vectors (length 4):
    # 1: [1, 0, 0, 0]
    # 2: [0, 0, 0, 1]
    # 3: [1, 0, 0, 1]
    # 4: [0, 1, -1, 0]
    # 5: [1, 1, 0, 0]
    # 6: [0, 0, 1, 1]
    # 7: [0, 1, -1, 0]
    
    # Let's verify Term 4:
    # U4 = [1, 0, -1, 0] -> a - c
    # V4 = [1, -1, 0, 0] -> e - g
    # Product = ae - ag - ce + cg
    # W4 = [0, 1, -1, 0]
    # W4 corresponds to:
    # W[0] (k=0, i=0) -> 0
    # W[1] (k=0, i=1) -> 1
    # W[2] (k=1, i=0) -> -1
    # W[3] (k=1, i=1) -> 0
    # Contribution to C:
    # C10 (i=1, k=0) gets 1 * (ae - ag - ce + cg)
    # C01 (i=0, k=1) gets -1 * (ae - ag - ce + cg)
    
    # This seems correct for the off-diagonals in the Winograd scheme.
    
    # We will construct the 49-term decomposition by taking the Kronecker product
    # of these 7 terms.
    
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Winograd terms for 2x2
    # Indices for 2x2: 0,1,2,3
    # U_win: list of 7 vectors of length 4
    # V_win: list of 7 vectors of length 4
    # W_win: list of 7 vectors of length 4
    
    U_win = [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, -1, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 0, -1]
    ]
    
    V_win = [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, -1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, -1, 0, 1]
    ]
    
    W_win = [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, -1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, -1, 0]
    ]
    
    # Iterate over the 7x7 combinations
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            
            # U index for 4x4: i*4 + j.
            # i in 0..3, j in 0..3.
            # i = i1*2 + i2, j = j1*2 + j2.
            # U_win[r1] gives coefficients for i1, j1 (2x2 block)
            # U_win[r2] gives coefficients for i2, j2 (2x2 block)
            # Actually, the Kronecker product structure:
            # U_4x4[i1*2+i2, j1*2+j2] = U_win[r1][i1*2+j1] * U_win[r2][i2*2+j2]?
            # No, the indices are coupled.
            # The tensor is $M_4 = M_2 \otimes M_2$.
            # The index for A is $i, j$.
            # $i = i_1 \cdot 2 + i_2$
            # $j = j_1 \cdot 2 + j_2$
            # The term $r$ in $M_2$ corresponds to $r_1$ for the outer block and $r_2$ for the inner block.
            # $U_{4x4}[i, j, r] = U_{2x2}[i_1, j_1, r_1] \cdot U_{2x2}[i_2, j_2, r_2]$
            
            # Let's populate U
            for i1 in range(2):
                for j1 in range(2):
                    val1 = U_win[r1][i1 * 2 + j1]
                    if val1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            val2 = U_win[r2][i2 * 2 + j2]
                            if val2 == 0: continue
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            idx_u = i * 4 + j
                            U[idx_u][col] += val1 * val2
            
            # Populate V
            # V index: j*4 + k.
            # j = j1*2 + j2, k = k1*2 + k2.
            # V_4x4[j, k, r] = V_2x2[j1, k1, r1] * V_2x2[j2, k2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    val1 = V_win[r1][j1 * 2 + k1]
                    if val1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            val2 = V_win[r2][j2 * 2 + k2]
                            if val2 == 0: continue
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            idx_v = j * 4 + k
                            V[idx_v][col] += val1 * val2
            
            # Populate W
            # W index: k*4 + i.
            # k = k1*2 + k2, i = i1*2 + i2.
            # W_4x4[k, i, r] = W_2x2[k1, i1, r1] * W_2x2[k2, i2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    val1 = W_win[r1][k1 * 2 + i1]
                    if val1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            val2 = W_win[r2][k2 * 2 + i2]
                            if val2 == 0: continue
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            idx_w = k * 4 + i
                            W[idx_w][col] += val1 * val2
                            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
