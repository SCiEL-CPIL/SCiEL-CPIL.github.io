import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 4, 4, 6
    
    def strassen_2x2():
        return [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
            ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
            ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
            ([1,-1,0,0], [1,0,1,0], [0,0,0,1]),
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
        ]
    
    def build_4x4x4_strassen():
        U, V, W = [], [], []
        st = strassen_2x2()
        def add_bp(a_r, a_c, b_c, c_c):
            a_idx = [a_r*4+a_c, a_r*4+a_c+1, (a_r+1)*4+a_c, (a_r+1)*4+a_c+1]
            b_idx = [a_c*4+b_c, a_c*4+b_c+1, (a_c+1)*4+b_c, (a_c+1)*4+b_c+1]
            c_idx = [a_r*4+b_c, a_r*4+b_c+1, (a_r+1)*4+b_c, (a_r+1)*4+b_c+1]
            for u, v, w in st:
                Un, Vn, Wn = [0]*16, [0]*16, [0]*16
                for i in range(4):
                    Un[a_idx[i]] = u[i]
                    Vn[b_idx[i]] = v[i]
                    Wn[c_idx[i]] = w[i]
                U.append(Un); V.append(Vn); W.append(Wn)
        add_bp(0,0,0,0); add_bp(0,2,2,0)
        add_bp(0,0,2,2); add_bp(0,2,2,2)
        add_bp(2,0,0,0); add_bp(2,2,2,0)
        add_bp(2,0,2,2); add_bp(2,2,2,2)
        return U, V, W
    
    def build_4x4x2_naive():
        U, V, W = [], [], []
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    u = [0]*16; v = [0]*8; w = [0]*8
                    u[i*4+j] = 1
                    v[j*2+k] = 1
                    w[i*2+k] = 1
                    U.append(u); V.append(v); W.append(w)
        return U, V, W
    
    U44, V44, W44 = build_4x4x4_strassen()
    U42, V42, W42 = build_4x4x2_naive()
    
    rank = len(U44) + len(U42)
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    for t in range(len(U44)):
        for i in range(16):
            if U44[t][i]: U[i][t] = U44[t][i]
            if V44[t][i]: V[i][t] = V44[t][i]
            if W44[t][i]: W[i][t] = W44[t][i]
    off = len(U44)
    for t in range(len(U42)):
        for i in range(16):
            if U42[t][i]: U[i][off+t] = U42[t][i]
        for i in range(8):
            if V42[t][i]: V[16+i][off+t] = V42[t][i]
            if W42[t][i]: W[16+i][off+t] = W42[t][i]
    
    np.random.seed(99)
    A_t = np.random.randn(4,4)
    B_t = np.random.randn(4,6)
    C_t = A_t @ B_t
    C_c = np.zeros((4,6))
    for t in range(rank):
        av = sum(U[i][t]*A_t[i//4][i%4] for i in range(16))
        bv = sum(V[i][t]*B_t[i//6][i%6] for i in range(24))
        for i in range(24):
            k, ii = i//4, i%4
            C_c[ii][k] += W[i][t]*av*bv
    assert np.allclose(C_t, C_c, atol=1e-10)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
