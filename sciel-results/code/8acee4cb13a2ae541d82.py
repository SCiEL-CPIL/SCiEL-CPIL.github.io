"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen (rank 7)."""
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # c=0: (A00+A11)*(B00+B11) -> C00(+), C11(+)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # c=1: (A10+A11)*B00 -> C10(+)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # c=2: A00*(B01-B11) -> C01(+), C11(+)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # c=3: A11*(B10-B00) -> C00(+), C10(+)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # c=4: (A00+A01)*B11 -> C00(-), C01(+)
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # c=5: (A10-A00)*(B00+B01) -> C11(+)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # c=6: (A01-A11)*(B10+B11) -> C00(+), C11(+)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _build_4x4x4():
    """Build 4x4x4 Strassen decomposition (rank 49)."""
    n = m = p = 4
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    for c_top in range(7):
        for c_sub in range(7):
            c = c_top * 7 + c_sub
            for bi in range(2):
                for bj in range(2):
                    ca = U2[bi*2 + bj][c_top]
                    if ca == 0:
                        continue
                    for bk in range(2):
                        for bl in range(2):
                            cb = V2[bj*2 + bl][c_top]
                            if cb == 0:
                                continue
                            for ri in range(2):
                                for cj in range(2):
                                    for rl in range(2):
                                        ua = U2[ri*2 + cj][c_sub]
                                        vb = V2[cj*2 + rl][c_sub]
                                        wc = W2[rl*2 + ri][c_sub]
                                        if ua == 0 or vb == 0 or wc == 0:
                                            continue
                                        u_idx = (2*bi + ri) * 4 + (2*bj + cj)
                                        v_idx = (2*bk + cj) * 4 + (2*bl + rl)
                                        w_idx = (2*bk + rl) * 4 + (2*bi + ri)
                                        U[u_idx][c] += ca * ua
                                        V[v_idx][c] += cb * vb
                                        W[w_idx][c] += wc
    return U, V, W

def _build_4x4x2():
    """Build 4x4x2 decomposition (rank 28)."""
    n, m, p = 4, 4, 2
    rank = 28
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # 4 2x2x2 multiplications
    mults = [(0,0,0,0), (0,1,1,0), (1,0,0,0), (1,1,1,0)]
    for mult_idx, (bi, bj, bk, bl) in enumerate(mults):
        for c_sub in range(7):
            c = mult_idx * 7 + c_sub
            for ri in range(2):
                for cj in range(2):
                    for rl in range(2):
                        ua = U2[ri*2 + cj][c_sub]
                        vb = V2[cj*2 + rl][c_sub]
                        wc = W2[rl*2 + ri][c_sub]
                        if ua == 0 or vb == 0 or wc == 0:
                            continue
                        u_idx = (2*bi + ri) * 4 + (2*bj + cj)
                        v_idx = (2*bk + cj) * 2 + bl
                        w_idx = (2*bk + rl) * 4 + (2*bi + ri)
                        U[u_idx][c] += ua
                        V[v_idx][c] += vb
                        W[w_idx][c] += wc
    return U, V, W

def construct():
    start_time = time.time()
    
    U44, V44, W44 = _build_4x4x4()
    U42, V42, W42 = _build_4x4x2()
    
    rank = 77
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(24)]
    
    # 4x4x4 part (first 49 cols)
    for i in range(16):
        for c in range(49):
            U[i][c] = U44[i][c]
            V[i][c] = V44[i][c]
            W[i][c] = W44[i][c]
    
    # 4x4x2 part (next 28 cols)
    for i in range(16):
        for c in range(28):
            U[i][49+c] = U42[i][c]
    
    for i in range(8):
        for c in range(28):
            j = i // 2
            k = i % 2
            full_v = j * 6 + (k + 4)
            V[full_v][49+c] = V42[i][c]
    
    for i in range(8):
        for c in range(28):
            k = i // 4
            i_rem = i % 4
            full_w = (k + 4) * 4 + i_rem
            W[full_w][49+c] = W42[i][c]
    
    # Verify
    if time.time() - start_time < 140:
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        n, m, p = 4, 4, 6
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
            if not ok:
                break
        if not ok:
            # Fallback
            rank = 96
            U = [[0]*rank for _ in range(16)]
            V = [[0]*rank for _ in range(24)]
            W = [[0]*rank for _ in range(24)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
    
    return {"U": U, "V": W if False else W, "W": W}
# EVOLVE-BLOCK-END
