import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56

    # We'll build the decomposition using the Kronecker product structure.
    # The matrix multiplication tensor T[i,j,k] = delta[i,k] * (A_{i,j} * B_{j,k})
    # can be decomposed as sum over r of U[i,j,r] * V[j,k,r] * W[k,i,r]
    # (with W indexed as W[k*n + i, r] per the trace convention)

    # Build U, V, W as (n*m, rank), (m*p, rank), (p*n, rank)
    # Using numpy for convenience

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Strassen's 2x2x2 decomposition (rank 7)
    # For multiplying 2x2 matrix A by 2x2 matrix B:
    # C[0,0] = M0 + M3 - M5 - M7  (wait, let me use standard Strassen)
    # 
    # Standard Strassen 2x2x2:
    # M1 = A11*(B11+B22)
    # M2 = (A11+A22)*B22
    # M3 = (A11+A12)*B11
    # M4 = (A21+A22)*(B11+B12)
    # M5 = A11*B11
    # M6 = (A12-A22)*(B21+B22)
    # M7 = (A11-A12)*B11
    #
    # Wait, let me use the standard formulation:
    # M1 = A11 * (B11 + B22)
    # M2 = (A11 + A22) * B22
    # M3 = (A11 + A12) * B11
    # M4 = (A21 + A22) * (B11 + B12)
    # M5 = A11 * B11
    # M6 = (A12 - A22) * (B21 + B22)
    # M7 = (A11 - A12) * B11
    #
    # C11 = M1 + M4 - M2 + M7
    # C12 = M3 + M5
    # C21 = M6 + M2
    # C22 = M1 - M5 + M4 + M6
    #
    # In tensor form, for indices i,j,k in {0,1}:
    # U[i,j,r], V[j,k,r], W[k,i,r] where r = 0..6

    # Let me define the 2x2x2 Strassen factors explicitly.
    # U[i,j] picks out A entries, V[j,k] picks out B entries, W[k,i] gives C entries.

    # M1 = A11 * (B11 + B22): i=0,j=0 -> A[0,0]; j,k: V has B[0,0]+B[1,1]; W[k,i]: C[0,0] gets +1, C[1,1] gets -1
    # Actually W[k*n+i, r] = W[k*2+i, r]
    # For M1: contributes to C[0,0] with +1 and C[1,1] with -1
    # W[0*2+0] = W[0] += 1 (k=0,i=0)
    # W[1*2+1] = W[3] += -1 (k=1,i=1)

    # Let me just define the 7 terms directly:
    # Term r, U indices (i,j), V indices (j,k), W indices (k,i)

    # r=0: M1 = A[0,0] * (B[0,0] + B[1,1])
    # U[0,0] = 1
    # V[0,0] = 1, V[1,1] = 1
    # W[0,0] = 1, W[1,1] = -1  -> W[0*2+0]=1, W[1*2+1]=-1 -> W[0]=1, W[3]=-1

    # r=1: M2 = (A[0,0]+A[1,1]) * B[1,1]
    # U[0,0]=1, U[1,1]=1
    # V[1,1]=1
    # W[1,0] = -1, W[1,1] = 1 -> W[1*2+0]=-1, W[1*2+1]=1 -> W[2]=-1, W[3]=1
    # Wait: C[0,0] = M1 + M4 - M2 + M7, so M2 contributes -1 to C[0,0]
    # C[1,1] = M1 - M5 + M4 + M6, so M2 doesn't appear... 
    # Let me recheck: C[1,1] = M1 - M5 + M4 + M6? No.
    # Standard: C11 = M1 + M4 - M2 + M7, C12 = M3 + M5, C21 = M6 + M2, C22 = M1 - M5 + M4 + M6
    # So M2: C[0,0] += -1, C[1,1] += 0, C[1,0] += 1 (C21), C[0,1] += 0
    # W[k*2+i]: C[0,0] -> k=0,i=0 -> W[0] += -1
    # C[1,0] -> k=0,i=1 -> W[1] += 1

    # This is getting complex. Let me just use the known rank-56 decomposition
    # by taking the Kronecker product of Strassen with identity blocks.

    # For 2x4 times 4x8:
    # Split A into 2 blocks of 2x2: A_L (cols 0-1), A_R (cols 2-3)
    # Split B into 4 blocks of 2x2: B_0 (rows 0-1), B_1 (rows 2-3), B_2 (rows 4-5), B_3 (rows 6-7)
    # Wait, B is 4x8. Split B into 4 blocks of 2x2 along rows: 
    #   B_top = B[0:2, :] (2x8), B_bot = B[2:4, :] (2x8)
    # Each 2x8 can be split into 4 blocks of 2x2.
    # 
    # So we have:
    # C = A_L * B_top + A_R * B_bot
    # = sum over 4 k-blocks: A_L * B_top[:, 2k:2k+2] + A_R * B_bot[:, 2k:2k+2]
    # = 8 terms of 2x2 * 2x2
    # Each term uses rank 7, total rank 56.

    # Let me define Strassen factors for 2x2x2 more carefully.
    # I'll use the formulation from the current program which was validated.

    # From the current program:
    strassen_U = [
        [(0, 1), (3, 1)],   # r=0: i=0,j=0; i=1,j=1
        [(2, 1), (3, 1)],   # r=1: i=1,j=0; i=1,j=1
        [(0, 1)],           # r=2: i=0,j=0
        [(3, 1)],           # r=3: i=1,j=1
        [(0, 1), (1, 1)],   # r=4: i=0,j=0; i=0,j=1
        [(2, 1), (0, -1)],  # r=5: i=1,j=0; i=0,j=0
        [(1, 1), (3, -1)],  # r=6: i=0,j=1; i=1,j=1
    ]
    strassen_V = [
        [(0, 1), (3, 1)],   # r=0
        [(0, 1)],           # r=1
        [(1, 1), (3, -1)],  # r=2
        [(2, 1), (0, -1)],  # r=3
        [(3, 1)],           # r=4
        [(0, 1), (1, 1)],   # r=5
        [(2, 1), (3, 1)],   # r=6
    ]
    strassen_W = [
        [(0, 1), (3, 1)],   # r=0: k=0,i=0; k=1,i=1
        [(1, 1), (3, -1)],  # r=1: k=0,i=1; k=1,i=1
        [(2, 1), (3, 1)],   # r=2: k=1,i=0; k=1,i=1
        [(0, 1), (1, 1)],   # r=3: k=0,i=0; k=0,i=1
        [(0, -1), (2, 1)],  # r=4: k=0,i=0; k=1,i=0
        [(3, 1)],           # r=5: k=1,i=1
        [(0, 1)],           # r=6: k=0,i=0
    ]

    # Now apply 8 blocks:
    # For block b (0..7):
    #   i_offset = 0 (always, since n=2)
    #   j_offset = 2 * (b // 4)  (0 or 2)
    #   k_offset = 2 * (b % 4)   (0, 2, 4, 6)
    #   base_rank = 7 * b

    for b in range(8):
        i_off = 0
        j_off = 2 * (b // 4)
        k_off = 2 * (b % 4)
        base = 7 * b

        for r in range(7):
            cr = base + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                gi = local_i + i_off
                gj = local_j + j_off
                U[gi * 4 + gj, cr] += val

            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                gj = local_j + j_off
                gk = local_k + k_off
                V[gj * 8 + gk, cr] += val

            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                gk = local_k + k_off
                gi = local_i + i_off
                W[gk * 2 + gi, cr] += val

    # Convert to lists for the expected return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
