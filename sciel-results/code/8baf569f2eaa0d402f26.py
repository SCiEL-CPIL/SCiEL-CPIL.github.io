import time
import numpy as np

N, M, P = 3, 4, 5
SCHOOLBOOK_RANK = N * M * P  # 60

START_TIME = time.time()
DEADLINE = START_TIME + 110

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)

def _schoolbook():
    rank = SCHOOLBOOK_RANK
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _rank11_2x2x3_factors():
    # Not used in the new strategy, kept for reference or fallback if needed
    pass

def _build_block(U, V, W, u_indices, v_indices, w_indices, block_U, block_V, block_W, start_col):
    """
    Adds a block decomposition to the main U, V, W matrices.
    u_indices: list of global U indices corresponding to local U rows 0..len-1
    v_indices: list of global V indices corresponding to local V rows 0..len-1
    w_indices: list of global W indices corresponding to local W rows 0..len-1
    """
    for r in range(block_U.shape[1]):
        for i, u_idx in enumerate(u_indices):
            U[u_idx, start_col + r] += block_U[i, r]
        for j, v_idx in enumerate(v_indices):
            V[v_idx, start_col + r] += block_V[j, r]
        for k, w_idx in enumerate(w_indices):
            W[w_idx, start_col + r] += block_W[k, r]
    return start_col + block_U.shape[1]

def _structured_3x4x5():
    """
    Construct a structured decomposition for 3x4x5 matrix multiplication.
    Rank 56.
    """
    U_str, V_str, W_str = _strassen_2x2_factors()
    
    # Schoolbook 2x2x1
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # Term 1: A00*B0 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A01*B1 -> C0
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # Term 3: A10*B0 -> C1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # Term 4: A11*B1 -> C1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Schoolbook 1x2x2
    def schoolbook_1x2x2():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        # Term 1: A0*B00 -> C0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A1*B10 -> C0
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        # Term 3: A0*B01 -> C1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        # Term 4: A1*B11 -> C1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    # Schoolbook 1x2x1
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        # Term 1: A0*B0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 2: A1*B1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    U = np.zeros((12, 56))
    V = np.zeros((20, 56))
    W = np.zeros((15, 56))
    
    col = 0
    
    # Indices for U (3x4)
    # A_1L (2x2): rows 0,1; cols 0,1 -> 0,1,4,5
    u_1L = [0, 1, 4, 5]
    # A_2L (1x2): row 2; cols 0,1 -> 8,9
    u_2L = [8, 9]
    # A_1R (2x2): rows 0,1; cols 2,3 -> 2,3,6,7
    u_1R = [2, 3, 6, 7]
    # A_2R (1x2): row 2; cols 2,3 -> 10,11
    u_2R = [10, 11]
    
    # Indices for V (4x5)
    # B_TL (2x2): rows 0,1; cols 0,1 -> 0,1,5,6
    v_TL = [0, 1, 5, 6]
    # B_TM (2x2): rows 0,1; cols 2,3 -> 2,3,7,8
    v_TM = [2, 3, 7, 8]
    # B_TR (2x1): rows 0,1; col 4 -> 4,9
    v_TR = [4, 9]
    # B_BL (2x2): rows 2,3; cols 0,1 -> 10,11,15,16
    v_BL = [10, 11, 15, 16]
    # B_BM (2x2): rows 2,3; cols 2,3 -> 12,13,17,18
    v_BM = [12, 13, 17, 18]
    # B_BR (2x1): rows 2,3; col 4 -> 14,19
    v_BR = [14, 19]
    
    # Indices for W (5x3) -> index k*3+i
    # C block (2x2) at rows i0,i1 and cols k0,k1
    # Local W indices: 0:(k0,i0), 1:(k0,i1), 2:(k1,i0), 3:(k1,i1)
    # Global W indices: k*3+i
    
    def get_w_indices(i0, k0):
        # i0 is start row, k0 is start col
        # Local 0: k=k0, i=i0 -> k0*3+i0
        # Local 1: k=k0, i=i0+1 -> k0*3+i0+1
        # Local 2: k=k0+1, i=i0 -> (k0+1)*3+i0
        # Local 3: k=k0+1, i=i0+1 -> (k0+1)*3+i0+1
        return [k0*3+i0, k0*3+i0+1, (k0+1)*3+i0, (k0+1)*3+i0+1]

    def get_w_indices_1x2(i0, k0):
        # 1x2 block: row i0, cols k0, k0+1
        # Local W indices: 0:(k0,i0), 1:(k0+1,i0)
        return [k0*3+i0, (k0+1)*3+i0]

    def get_w_indices_2x1(i0, k0):
        # 2x1 block: rows i0, i0+1, col k0
        # Local W indices: 0:(k0,i0), 1:(k0,i0+1)
        return [k0*3+i0, k0*3+i0+1]

    def get_w_indices_1x1(i0, k0):
        # 1x1 block
        return [k0*3+i0]

    # Block 1: A_1L * B_TL (2x2x2)
    # A_1L rows 0,1. B_TL cols 0,1. C rows 0,1 cols 0,1.
    w_idx = get_w_indices(0, 0)
    col = _build_block(U, V, W, u_1L, v_TL, w_idx, U_str, V_str, W_str, col)
    
    # Block 2: A_1L * B_TM (2x2x2)
    # A_1L rows 0,1. B_TM cols 2,3. C rows 0,1 cols 2,3.
    w_idx = get_w_indices(0, 2)
    col = _build_block(U, V, W, u_1L, v_TM, w_idx, U_str, V_str, W_str, col)
    
    # Block 3: A_1L * B_TR (2x2x1)
    # A_1L rows 0,1. B_TR col 4. C rows 0,1 col 4.
    U_b, V_b, W_b = schoolbook_2x2x1()
    w_idx = get_w_indices_2x1(0, 4)
    col = _build_block(U, V, W, u_1L, v_TR, w_idx, U_b, V_b, W_b, col)
    
    # Block 4: A_2L * B_TL (1x2x2)
    # A_2L row 2. B_TL cols 0,1. C row 2 cols 0,1.
    U_b, V_b, W_b = schoolbook_1x2x2()
    w_idx = get_w_indices_1x2(2, 0)
    col = _build_block(U, V, W, u_2L, v_TL, w_idx, U_b, V_b, W_b, col)
    
    # Block 5: A_2L * B_TM (1x2x2)
    # A_2L row 2. B_TM cols 2,3. C row 2 cols 2,3.
    U_b, V_b, W_b = schoolbook_1x2x2()
    w_idx = get_w_indices_1x2(2, 2)
    col = _build_block(U, V, W, u_2L, v_TM, w_idx, U_b, V_b, W_b, col)
    
    # Block 6: A_2L * B_TR (1x2x1)
    # A_2L row 2. B_TR col 4. C row 2 col 4.
    U_b, V_b, W_b = schoolbook_1x2x1()
    w_idx = get_w_indices_1x1(2, 4)
    col = _build_block(U, V, W, u_2L, v_TR, w_idx, U_b, V_b, W_b, col)
    
    # Block 7: A_1R * B_BL (2x2x2)
    # A_1R rows 0,1. B_BL cols 0,1. C rows 0,1 cols 0,1.
    w_idx = get_w_indices(0, 0)
    col = _build_block(U, V, W, u_1R, v_BL, w_idx, U_str, V_str, W_str, col)
    
    # Block 8: A_1R * B_BM (2x2x2)
    # A_1R rows 0,1. B_BM cols 2,3. C rows 0,1 cols 2,3.
    w_idx = get_w_indices(0, 2)
    col = _build_block(U, V, W, u_1R, v_BM, w_idx, U_str, V_str, W_str, col)
    
    # Block 9: A_1R * B_BR (2x2x1)
    # A_1R rows 0,1. B_BR col 4. C rows 0,1 col 4.
    U_b, V_b, W_b = schoolbook_2x2x1()
    w_idx = get_w_indices_2x1(0, 4)
    col = _build_block(U, V, W, u_1R, v_BR, w_idx, U_b, V_b, W_b, col)
    
    # Block 10: A_2R * B_BL (1x2x2)
    # A_2R row 2. B_BL cols 0,1. C row 2 cols 0,1.
    U_b, V_b, W_b = schoolbook_1x2x2()
    w_idx = get_w_indices_1x2(2, 0)
    col = _build_block(U, V, W, u_2R, v_BL, w_idx, U_b, V_b, W_b, col)
    
    # Block 11: A_2R * B_BM (1x2x2)
    # A_2R row 2. B_BM cols 2,3. C row 2 cols 2,3.
    U_b, V_b, W_b = schoolbook_1x2x2()
    w_idx = get_w_indices_1x2(2, 2)
    col = _build_block(U, V, W, u_2R, v_BM, w_idx, U_b, V_b, W_b, col)
    
    # Block 12: A_2R * B_BR (1x2x1)
    # A_2R row 2. B_BR col 4. C row 2 col 4.
    U_b, V_b, W_b = schoolbook_1x2x1()
    w_idx = get_w_indices_1x1(2, 4)
    col = _build_block(U, V, W, u_2R, v_BR, w_idx, U_b, V_b, W_b, col)
    
    assert col == 56, f"Expected rank 56, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    # Try structured decomposition first
    try:
        U_str, V_str, W_str = _structured_3x4x5()
        if _verify(U_str, V_str, W_str):
            return {"U": U_str, "V": V_str, "W": W_str}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb = _schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
