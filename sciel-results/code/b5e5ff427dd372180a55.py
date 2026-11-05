import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS

    U_parts = []
    V_parts = []
    W_parts = []

    # Helper: add a block with given index mappings
    def add_block(u_idx, v_idx, w_idx, U_blk, V_blk, W_blk):
        for r in range(U_blk.shape[1]):
            u_row = np.zeros(n * m)
            v_row = np.zeros(m * p)
            w_row = np.zeros(p * n)
            for i, idx in enumerate(u_idx):
                u_row[idx] = U_blk[i, r]
            for i, idx in enumerate(v_idx):
                v_row[idx] = V_blk[i, r]
            for i, idx in enumerate(w_idx):
                w_row[idx] = W_blk[i, r]
            U_parts.append(u_row)
            V_parts.append(v_row)
            W_parts.append(w_row)

    # Strassen 2x2x2: U(4x7), V(4x7), W(4x7)
    # U order: A00, A01, A10, A11
    # V order: B00, B01, B10, B11
    # W order: C00, C01, C10, C11
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[3,0]=1
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1; V[0,1]=1
        W[2,1]=1; W[3,1]=-1
        # M3 = A00(B01-B11)
        U[0,2]=1; V[1,2]=1; V[3,2]=-1
        W[1,2]=1; W[3,2]=1
        # M4 = A11(B10-B00)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1; V[3,4]=1
        W[0,4]=-1; W[1,4]=1
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
        W[3,5]=1
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1
        W[0,6]=1
        return U, V, W

    # Schoolbook 2x2x1: U(4x4), V(2x4), W(2x4)
    # U order: A00, A01, A10, A11
    # V order: B0, B1
    # W order: C0, C1
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W

    # Schoolbook 1x2x2: U(2x4), V(4x4), W(2x4)
    # U order: A0, A1
    # V order: B00, B01, B10, B11
    # W order: C0, C1
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[2,1]=1; W[0,1]=1
        U[0,2]=1; V[1,2]=1; W[1,2]=1
        U[1,3]=1; V[3,3]=1; W[1,3]=1
        return U, V, W

    # Schoolbook 1x2x1: U(2x2), V(2x2), W(1x2)
    # U order: A0, A1
    # V order: B0, B1
    # W order: C0
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        return U, V, W

    # Index mappings for 3x4x5
    # U[i*m + j] for A[i,j]
    # V[i*p + j] for B[i,j]
    # W[k*n + i] for C[i,k]

    # Block 1: A rows 0,1 cols 0,1 * B rows 0,1 cols 0,1 (2x2x2)
    u_idx = [0, 1, 4, 5]      # A00, A01, A10, A11
    v_idx = [0, 1, 5, 6]      # B00, B01, B10, B11
    w_idx = [0, 3, 1, 4]      # C00, C01, C10, C11
    U_b, V_b, W_b = strassen_2x2x2()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 2: A rows 0,1 cols 0,1 * B rows 0,1 cols 2,3 (2x2x2)
    u_idx = [0, 1, 4, 5]
    v_idx = [2, 3, 7, 8]      # B02, B03, B12, B13
    w_idx = [6, 9, 7, 10]     # C02, C03, C12, C13
    U_b, V_b, W_b = strassen_2x2x2()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 3: A rows 0,1 cols 0,1 * B rows 0,1 col 4 (2x2x1)
    u_idx = [0, 1, 4, 5]
    v_idx = [4, 9]            # B04, B14
    w_idx = [12, 13]          # C04, C14
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 4: A rows 0,1 cols 2,3 * B rows 2,3 cols 0,1 (2x2x2)
    u_idx = [2, 3, 6, 7]      # A02, A03, A12, A13
    v_idx = [10, 11, 15, 16]  # B20, B21, B30, B31
    w_idx = [0, 3, 1, 4]      # C00, C01, C10, C11
    U_b, V_b, W_b = strassen_2x2x2()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 5: A rows 0,1 cols 2,3 * B rows 2,3 cols 2,3 (2x2x2)
    u_idx = [2, 3, 6, 7]
    v_idx = [12, 13, 17, 18]  # B22, B23, B32, B33
    w_idx = [6, 9, 7, 10]     # C02, C03, C12, C13
    U_b, V_b, W_b = strassen_2x2x2()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 6: A rows 0,1 cols 2,3 * B rows 2,3 col 4 (2x2x1)
    u_idx = [2, 3, 6, 7]
    v_idx = [14, 19]          # B24, B34
    w_idx = [12, 13]          # C04, C14
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 7: A row 2 cols 0,1 * B rows 0,1 cols 0,1 (1x2x2)
    u_idx = [8, 9]            # A20, A21
    v_idx = [0, 1, 5, 6]      # B00, B01, B10, B11
    w_idx = [2, 5]            # C20, C21
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 8: A row 2 cols 0,1 * B rows 0,1 cols 2,3 (1x2x2)
    u_idx = [8, 9]
    v_idx = [2, 3, 7, 8]      # B02, B03, B12, B13
    w_idx = [8, 11]           # C22, C23
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 9: A row 2 cols 0,1 * B rows 0,1 col 4 (1x2x1)
    u_idx = [8, 9]
    v_idx = [4, 9]            # B04, B14
    w_idx = [14]              # C24
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 10: A row 2 cols 2,3 * B rows 2,3 cols 0,1 (1x2x2)
    u_idx = [10, 11]          # A22, A23
    v_idx = [10, 11, 15, 16]  # B20, B21, B30, B31
    w_idx = [2, 5]            # C20, C21
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 11: A row 2 cols 2,3 * B rows 2,3 cols 2,3 (1x2x2)
    u_idx = [10, 11]
    v_idx = [12, 13, 17, 18]  # B22, B23, B32, B33
    w_idx = [8, 11]           # C22, C23
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    # Block 12: A row 2 cols 2,3 * B rows 2,3 col 4 (1x2x1)
    u_idx = [10, 11]
    v_idx = [14, 19]          # B24, B34
    w_idx = [14]              # C24
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block(u_idx, v_idx, w_idx, U_b, V_b, W_b)

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
