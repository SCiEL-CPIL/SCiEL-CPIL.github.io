import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()

    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # We decompose the multiplication C = A * B
    # A is n x m (2 x 4)
    # B is m x p (4 x 8)
    # C is n x p (2 x 8)
    
    # Split A into two 2x2 blocks: A0 (cols 0,1) and A1 (cols 2,3)
    # Split B into two 2x8 blocks: B0 (rows 0,1) and B1 (rows 2,3)
    # C = A0 * B0 + A1 * B1
    
    # Each term is a (2x2) * (2x8) multiplication.
    # We split the (2x8) part into four (2x2) blocks along the columns.
    # So each (2x2) * (2x8) is sum of four (2x2) * (2x2) multiplications.
    # Rank of (2x2) * (2x2) is 7 (Strassen).
    # Rank of (2x2) * (2x8) is 4 * 7 = 28.
    # Total rank = 28 + 28 = 56.

    # Strassen 2x2 local factorization (rank 7).
    # Local A indices: a00=0, a01=1, a10=2, a11=3
    # Local B indices: b00=0, b01=1, b10=2, b11=3
    # Local C indices: c00=0, c01=1, c10=2, c11=3
    # U_loc[local_a_idx][r], V_loc[local_b_idx][r], W_loc[local_c_idx][r]
    
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]

    # r=0: M1=(a00+a11)(b00+b11) -> c00+1, c11+1
    U_loc[0][0]=1; U_loc[3][0]=1
    V_loc[0][0]=1; V_loc[3][0]=1
    W_loc[0][0]=1; W_loc[3][0]=1
    
    # r=1: M2=(a10+a11)b00 -> c10+1, c11-1
    U_loc[2][1]=1; U_loc[3][1]=1
    V_loc[0][1]=1
    W_loc[2][1]=1; W_loc[3][1]=-1
    
    # r=2: M3=a00(b01-b11) -> c01+1, c11+1
    U_loc[0][2]=1
    V_loc[1][2]=1; V_loc[3][2]=-1
    W_loc[1][2]=1; W_loc[3][2]=1
    
    # r=3: M4=a11(b10-b00) -> c00+1, c10+1
    U_loc[3][3]=1
    V_loc[2][3]=1; V_loc[0][3]=-1
    W_loc[0][3]=1; W_loc[2][3]=1
    
    # r=4: M5=(a00+a01)b11 -> c00-1, c01+1
    U_loc[0][4]=1; U_loc[1][4]=1
    V_loc[3][4]=1
    W_loc[0][4]=-1; W_loc[1][4]=1
    
    # r=5: M6=(a10-a00)(b00+b01) -> c11+1
    U_loc[2][5]=1; U_loc[0][5]=-1
    V_loc[0][5]=1; V_loc[1][5]=1
    W_loc[3][5]=1
    
    # r=6: M7=(a01-a11)(b10+b11) -> c00+1
    U_loc[1][6]=1; U_loc[3][6]=-1
    V_loc[2][6]=1; V_loc[3][6]=1
    W_loc[0][6]=1

    total_rank = 56
    U = [[0]*total_rank for _ in range(n*m)]   # 8 x 56
    V = [[0]*total_rank for _ in range(m*p)]   # 32 x 56
    W = [[0]*total_rank for _ in range(p*n)]   # 16 x 56

    # Global index helpers.
    # A is n x m. Flat index i*m + j.
    # B is m x p. Flat index j*p + k.
    # W convention: W[k*n + i, r] -> output index = k * n + i.

    # We have 2 "halves" of A/B (split along m)
    # and 4 "blocks" of B (split along p)
    # So 2 * 4 = 8 groups of 2x2x2 Strassen blocks.
    
    group = 0
    for half in range(2):       # 0: A cols 0,1 & B rows 0,1 ; 1: A cols 2,3 & B rows 2,3
        for jj in range(p // 2): # 0..3: B cols 2jj, 2jj+1
            base_r = group * 7
            group += 1

            # A-half 'half': columns 2*half, 2*half+1
            # local a00->(i=0, j=2h), a01->(i=0, j=2h+1), a10->(i=1, j=2h), a11->(i=1, j=2h+1)
            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]
            
            # B-half 'half': rows 2*half, 2*half+1
            # B-block 'jj': cols 2*jj, 2*jj+1
            # local b00->(row=2h, col=2jj), b01->(row=2h, col=2jj+1)
            # local b10->(row=2h+1, col=2jj), b11->(row=2h+1, col=2jj+1)
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]
            
            # C local: c00->(i=0, k=2jj), c01->(i=0, k=2jj+1)
            # c10->(i=1, k=2jj), c11->(i=1, k=2jj+1)
            # W index = k*n + i
            c_map = [
                (2*jj + 0)*n + 0,
                (2*jj + 1)*n + 0,
                (2*jj + 0)*n + 1,
                (2*jj + 1)*n + 1,
            ]

            for r in range(7):
                rr = base_r + r
                for la in range(4):
                    val = U_loc[la][r]
                    if val:
                        U[a_map[la]][rr] += val
                for lb in range(4):
                    val = V_loc[lb][r]
                    if val:
                        V[b_map[lb]][rr] += val
                for lc in range(4):
                    val = W_loc[lc][r]
                    if val:
                        W[c_map[lc]][rr] += val

    # Safety deadline guard
    if time.time() - start_time > 120:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
