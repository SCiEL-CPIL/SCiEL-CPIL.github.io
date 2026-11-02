import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()

    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Rank of 2x2x2 matrix multiplication is 7 (Strassen).
    # We decompose 2x4 * 4x8 into 8 independent 2x2 * 2x2 multiplications.
    # Structure:
    # A (2x4) split into two 2x2 blocks: A0 (cols 0,1), A1 (cols 2,3)
    # B (4x8) split into four 2x2 blocks for each half:
    #   B0 (rows 0,1) split into B0_0, B0_1, B0_2, B0_3 (cols 0-1, 2-3, 4-5, 6-7)
    #   B1 (rows 2,3) split into B1_0, B1_1, B1_2, B1_3
    # C (2x8) is sum of these products.
    # Since the k-indices (inner dimension) for A0/B0 are 0,1 and for A1/B1 are 2,3,
    # the tensors are disjoint in the B-dimension (and A-dimension), so ranks add.
    # Total rank = 8 * 7 = 56.

    total_rank = 56
    U = [[0]*total_rank for _ in range(n*m)]   # 8 x 56
    V = [[0]*total_rank for _ in range(m*p)]   # 32 x 56
    W = [[0]*total_rank for _ in range(p*n)]   # 16 x 56

    # Strassen decomposition for 2x2 matrices.
    # Local A indices: 0:00, 1:01, 2:10, 3:11
    # Local B indices: 0:00, 1:01, 2:10, 3:11
    # Local C indices: 0:00, 1:01, 2:10, 3:11
    # U_loc[local_a][r], V_loc[local_b][r], W_loc[local_c][r]
    
    U_loc = [[0]*7 for _ in range(4)]
    V_loc = [[0]*7 for _ in range(4)]
    W_loc = [[0]*7 for _ in range(4)]

    # r=0: (a00+a11)(b00+b11) -> c00+1, c11+1
    U_loc[0][0]=1; U_loc[3][0]=1
    V_loc[0][0]=1; V_loc[3][0]=1
    W_loc[0][0]=1; W_loc[3][0]=1
    
    # r=1: (a10+a11)b00 -> c10+1, c11-1
    U_loc[2][1]=1; U_loc[3][1]=1
    V_loc[0][1]=1
    W_loc[2][1]=1; W_loc[3][1]=-1
    
    # r=2: a00(b01-b11) -> c01+1, c11+1
    U_loc[0][2]=1
    V_loc[1][2]=1; V_loc[3][2]=-1
    W_loc[1][2]=1; W_loc[3][2]=1
    
    # r=3: a11(b10-b00) -> c00+1, c10+1
    U_loc[3][3]=1
    V_loc[2][3]=1; V_loc[0][3]=-1
    W_loc[0][3]=1; W_loc[2][3]=1
    
    # r=4: (a00+a01)b11 -> c00-1, c01+1
    U_loc[0][4]=1; U_loc[1][4]=1
    V_loc[3][4]=1
    W_loc[0][4]=-1; W_loc[1][4]=1
    
    # r=5: (a10-a00)(b00+b01) -> c11+1
    U_loc[2][5]=1; U_loc[0][5]=-1
    V_loc[0][5]=1; V_loc[1][5]=1
    W_loc[3][5]=1
    
    # r=6: (a01-a11)(b10+b11) -> c00+1
    U_loc[1][6]=1; U_loc[3][6]=-1
    V_loc[2][6]=1; V_loc[3][6]=1
    W_loc[0][6]=1

    # Global index mapping
    # A flat index: i * m + k  (i in 0..1, k in 0..3)
    # B flat index: k * p + l  (k in 0..3, l in 0..7)
    # W flat index: l * n + i  (l in 0..7, i in 0..1) -- Note: Task says W[k*n+i, r] where k is col index? 
    # Let's re-verify the W convention from the prompt: "W[k*n+i,r]"
    # In matrix multiplication C = AB, C has dimensions n x p.
    # Usually indices are C[i, l].
    # If the vector W is of size p*n, and index is k*n+i, it implies column-major order for C.
    # i is row (0..n-1), k is col (0..p-1).
    # Index = k * n + i. This matches my derivation above.

    group = 0
    for half in range(2):       # 0: A cols 0,1 & B rows 0,1 ; 1: A cols 2,3 & B rows 2,3
        for jj in range(p // 2): # 0..3: B cols 2jj, 2jj+1
            base_r = group * 7
            group += 1

            # A-half 'half': columns 2*half, 2*half+1
            # Local a00 -> global (i=0, k=2*half)
            # Local a01 -> global (i=0, k=2*half+1)
            # Local a10 -> global (i=1, k=2*half)
            # Local a11 -> global (i=1, k=2*half+1)
            a_map = [
                0*m + (2*half + 0),
                0*m + (2*half + 1),
                1*m + (2*half + 0),
                1*m + (2*half + 1),
            ]
            
            # B-half 'half': rows 2*half, 2*half+1
            # B-block 'jj': cols 2*jj, 2*jj+1
            # Local b00 -> global (k=2*half, l=2*jj)
            # Local b01 -> global (k=2*half, l=2*jj+1)
            # Local b10 -> global (k=2*half+1, l=2*jj)
            # Local b11 -> global (k=2*half+1, l=2*jj+1)
            b_map = [
                (2*half + 0)*p + (2*jj + 0),
                (2*half + 0)*p + (2*jj + 1),
                (2*half + 1)*p + (2*jj + 0),
                (2*half + 1)*p + (2*jj + 1),
            ]
            
            # C local:
            # c00 -> global (i=0, l=2*jj)
            # c01 -> global (i=0, l=2*jj+1)
            # c10 -> global (i=1, l=2*jj)
            # c11 -> global (i=1, l=2*jj+1)
            # W index = l * n + i
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
    if time.time() - start_time > 110:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
