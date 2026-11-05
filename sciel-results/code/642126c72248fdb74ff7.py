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
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_2x2x1():
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    return U, V, W

def _schoolbook_1x2x2():
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
    U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
    U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
    return U, V, W

def _schoolbook_1x2x1():
    U = np.zeros((2, 2))
    V = np.zeros((2, 2))
    W = np.zeros((1, 2))
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
    return U, V, W

def _structured_3x4x5():
    U_str, V_str, W_str = _strassen_2x2_factors()
    
    U = np.zeros((12, 56))
    V = np.zeros((20, 56))
    W = np.zeros((15, 56))
    
    u_1L = [0, 1, 4, 5]
    u_2L = [8, 9]
    u_1R = [2, 3, 6, 7]
    u_2R = [10, 11]
    
    v_TL = [0, 1, 5, 6]
    v_TM = [2, 3, 7, 8]
    v_TR = [4, 9]
    v_BL = [10, 11, 15, 16]
    v_BM = [12, 13, 17, 18]
    v_BR = [14, 19]
    
    def get_w_indices(i0, k0):
        return [k0*3+i0, k0*3+i0+1, (k0+1)*3+i0, (k0+1)*3+i0+1]

    def get_w_indices_1x2(i0, k0):
        return [k0*3+i0, (k0+1)*3+i0]

    def get_w_indices_2x1(i0, k0):
        return [k0*3+i0, k0*3+i0+1]

    def get_w_indices_1x1(i0, k0):
        return [k0*3+i0]

    def add_block(u_idx, v_idx, w_idx, bU, bV, bW, col):
        for r in range(bU.shape[1]):
            U[u_idx, col+r] = bU[:, r]
            V[v_idx, col+r] = bV[:, r]
            W[w_idx, col+r] = bW[:, r]
        return col + bU.shape[1]

    col = 0
    col = add_block(u_1L, v_TL, get_w_indices(0,0), U_str, V_str, W_str, col)
    col = add_block(u_1L, v_TM, get_w_indices(0,2), U_str, V_str, W_str, col)
    
    U_b, V_b, W_b = _schoolbook_2x2x1()
    col = add_block(u_1L, v_TR, get_w_indices_2x1(0,4), U_b, V_b, W_b, col)
    
    U_b, V_b, W_b = _schoolbook_1x2x2()
    col = add_block(u_2L, v_TL, get_w_indices_1x2(2,0), U_b, V_b, W_b, col)
    col = add_block(u_2L, v_TM, get_w_indices_1x2(2,2), U_b, V_b, W_b, col)
    
    U_b, V_b, W_b = _schoolbook_1x2x1()
    col = add_block(u_2L, v_TR, get_w_indices_1x1(2,4), U_b, V_b, W_b, col)
    
    col = add_block(u_1R, v_BL, get_w_indices(0,0), U_str, V_str, W_str, col)
    col = add_block(u_1R, v_BM, get_w_indices(0,2), U_str, V_str, W_str, col)
    
    U_b, V_b, W_b = _schoolbook_2x2x1()
    col = add_block(u_1R, v_BR, get_w_indices_2x1(0,4), U_b, V_b, W_b, col)
    
    U_b, V_b, W_b = _schoolbook_1x2x2()
    col = add_block(u_2R, v_BL, get_w_indices_1x2(2,0), U_b, V_b, W_b, col)
    col = add_block(u_2R, v_BM, get_w_indices_1x2(2,2), U_b, V_b, W_b, col)
    
    U_b, V_b, W_b = _schoolbook_1x2x1()
    col = add_block(u_2R, v_BR, get_w_indices_1x1(2,4), U_b, V_b, W_b, col)
    
    assert col == 56
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    try:
        U_str, V_str, W_str = _structured_3x4x5()
        if _verify(U_str, V_str, W_str):
            return {"U": U_str, "V": V_str, "W": W_str}
    except Exception:
        pass
    U_fb, V_fb, W_fb = _schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
