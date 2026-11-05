import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-56 tensor decomposition for 3x4x5 matrix multiplication.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1.
    Strategy: split A into A1 (rows 0,1) and A2 (row 2); split B's rows into
    two groups of 2; split B's cols into 2+2+1. Use Strassen (rank 7) for the
    2x2x2 blocks and schoolbook for the remaining small blocks.
    """
    n, m, p = DIMENSIONS
    N1, N2, N3 = n * m, m * p, p * n

    U_parts = []
    V_parts = []
    W_parts = []

    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            u = np.zeros(N1)
            v = np.zeros(N2)
            w = np.zeros(N3)
            for i, idx in enumerate(u_idx):
                u[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                v[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                w[idx] = W_blk[i, r]
            U_parts.append(u)
            V_parts.append(v)
            W_parts.append(w)

    def strassen_2x2x2():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # M1=(A00+A11)(B00+B11)
        U[0,0]=1;U[3,0]=1;V[0,0]=1;V[3,0]=1;W[0,0]=1;W[3,0]=1
        # M2=(A10+A11)B00
        U[2,1]=1;U[3,1]=1;V[0,1]=1;W[2,1]=1;W[3,1]=-1
        # M3=A00(B01-B11)
        U[0,2]=1;V[1,2]=1;V[3,2]=-1;W[1,2]=1;W[3,2]=1
        # M4=A11(B10-B00)
        U[3,3]=1;V[2,3]=1;V[0,3]=-1;W[0,3]=1;W[2,3]=1
        # M5=(A00+A01)B11
        U[0,4]=1;U[1,4]=1;V[3,4]=1;W[0,4]=-1;W[1,4]=1
        # M6=(A10-A00)(B00+B01)
        U[2,5]=1;U[0,5]=-1;V[0,5]=1;V[1,5]=1;W[3,5]=1
        # M7=(A01-A11)(B10+B11)
        U[1,6]=1;U[3,6]=-1;V[2,6]=1;V[3,6]=1;W[0,6]=1
        return U, V, W

    def schoolbook_2x2x1():
        U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
        U[0,0]=1;V[0,0]=1;W[0,0]=1
        U[1,1]=1;V[1,1]=1;W[0,1]=1
        U[2,2]=1;V[0,2]=1;W[1,2]=1
        U[3,3]=1;V[1,3]=1;W[1,3]=1
        return U, V, W

    def schoolbook_1x2x2():
        U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
        U[0,0]=1;V[0,0]=1;W[0,0]=1
        U[1,1]=1;V[2,1]=1;W[0,1]=1
        U[0,2]=1;V[1,2]=1;W[1,2]=1
        U[1,3]=1;V[3,3]=1;W[1,3]=1
        return U, V, W

    def schoolbook_1x2x1():
        U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
        U[0,0]=1;V[0,0]=1;W[0,0]=1
        U[1,1]=1;V[1,1]=1;W[0,1]=1
        return U, V, W

    # U indices (i*m+j): row0=0..3, row1=4..7, row2=8..11
    # V indices (j*p+k): row0=0..4, row1=5..9, row2=10..14, row3=15..19
    # W indices (k*n+i): col k, row i
    # C row0 cols0..4 -> W 0,3,6,9,12 ; row1 -> 1,4,7,10,13 ; row2 -> 2,5,8,11,14

    # Part 1: A1 (rows 0,1)
    # A1_top cols0,1 -> U 0,1,4,5 ; A1_bottom cols2,3 -> U 2,3,6,7
    # B_top_left V 0,1,5,6 ; B_top_mid V 2,3,7,8 ; B_top_right V 4,9
    # B_bot_left V 10,11,15,16 ; B_bot_mid V 12,13,17,18 ; B_bot_right V 14,19
    U_top = [0,1,4,5]; U_bot = [2,3,6,7]
    V_tl = [0,1,5,6]; V_tm = [2,3,7,8]; V_tr = [4,9]
    V_bl = [10,11,15,16]; V_bm = [12,13,17,18]; V_br = [14,19]
    W_c01 = [0,3,1,4]; W_c23 = [6,9,7,10]; W_c4 = [12,13]

    for u_idx, v_idx, w_idx, blk in [
        (U_top, V_tl, W_c01, strassen_2x2x2()),
        (U_top, V_tm, W_c23, strassen_2x2x2()),
        (U_top, V_tr, W_c4,  schoolbook_2x2x1()),
        (U_bot, V_bl, W_c01, strassen_2x2x2()),
        (U_bot, V_bm, W_c23, strassen_2x2x2()),
        (U_bot, V_br, W_c4,  schoolbook_2x2x1()),
    ]:
        add_block(u_idx, v_idx, w_idx, *blk)

    # Part 2: A2 (row 2)
    U_at = [8,9]; U_ab = [10,11]
    W_c01_2 = [2,5]; W_c23_2 = [8,11]; W_c4_2 = [14]

    for u_idx, v_idx, w_idx, blk in [
        (U_at, V_tl, W_c01_2, schoolbook_1x2x2()),
        (U_at, V_tm, W_c23_2, schoolbook_1x2x2()),
        (U_at, V_tr, W_c4_2,  schoolbook_1x2x1()),
        (U_ab, V_bl, W_c01_2, schoolbook_1x2x2()),
        (U_ab, V_bm, W_c23_2, schoolbook_1x2x2()),
        (U_ab, V_br, W_c4_2,  schoolbook_1x2x1()),
    ]:
        add_block(u_idx, v_idx, w_idx, *blk)

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T

    # Verification (cheap, 56 terms)
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = 0.0
                for r in range(U.shape[1]):
                    acc += W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r]
                T[i, j, k] = acc
    if not np.allclose(T, np.ones((n, m, p)), atol=1e-9):
        # Fallback to guaranteed schoolbook rank 60
        rank = n*m*p
        Us = np.zeros((n*m, rank)); Vs = np.zeros((m*p, rank)); Ws = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i*m+j, c] = 1; Vs[j*p+k, c] = 1; Ws[k*n+i, c] = 1; c += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
