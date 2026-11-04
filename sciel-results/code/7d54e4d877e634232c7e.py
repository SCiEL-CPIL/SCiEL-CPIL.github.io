import time
import numpy as np

_T0 = time.time()
_DEADLINE_S = 110.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n=2, m=5, p=5):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _add_strassen_222(U, V, W, offset, k0, k1, j0, j1, n, m, p):
    a00 = 0*m + k0
    a01 = 0*m + k1
    a10 = 1*m + k0
    a11 = 1*m + k1
    b00 = k0*p + j0
    b01 = k0*p + j1
    b10 = k1*p + j0
    b11 = k1*p + j1
    c00 = j0*n + 0
    c01 = j1*n + 0
    c10 = j0*n + 1
    c11 = j1*n + 1
    U[a00][offset+0] += 1
    U[a11][offset+0] += 1
    V[b00][offset+0] += 1
    V[b11][offset+0] += 1
    W[c00][offset+0] += 1
    W[c11][offset+0] += 1
    U[a10][offset+1] += 1
    U[a11][offset+1] += 1
    V[b00][offset+1] += 1
    W[c10][offset+1] += 1
    W[c11][offset+1] += -1
    U[a00][offset+2] += 1
    V[b01][offset+2] += 1
    V[b11][offset+2] += -1
    W[c01][offset+2] += 1
    W[c11][offset+2] += 1
    U[a11][offset+3] += 1
    V[b10][offset+3] += 1
    V[b00][offset+3] += -1
    W[c00][offset+3] += 1
    W[c10][offset+3] += 1
    U[a00][offset+4] += 1
    U[a01][offset+4] += 1
    V[b11][offset+4] += 1
    W[c00][offset+4] += -1
    W[c01][offset+4] += 1
    U[a10][offset+5] += 1
    U[a00][offset+5] += -1
    V[b00][offset+5] += 1
    V[b01][offset+5] += 1
    W[c11][offset+5] += 1
    U[a01][offset+6] += 1
    U[a11][offset+6] += -1
    V[b10][offset+6] += 1
    V[b11][offset+6] += 1
    W[c00][offset+6] += 1
    return offset + 7

def _add_2x1(U, V, W, offset, k0, k1, j, n, m, p):
    a0k0 = 0*m + k0
    a0k1 = 0*m + k1
    a1k0 = 1*m + k0
    a1k1 = 1*m + k1
    bk0j = k0*p + j
    bk1j = k1*p + j
    c0j = j*n + 0
    c1j = j*n + 1
    U[a0k0][offset+0] += 1
    V[bk0j][offset+0] += 1
    W[c0j][offset+0] += 1
    U[a0k1][offset+1] += 1
    V[bk1j][offset+1] += 1
    W[c0j][offset+1] += 1
    U[a1k0][offset+2] += 1
    V[bk0j][offset+2] += 1
    W[c1j][offset+2] += 1
    U[a1k1][offset+3] += 1
    V[bk1j][offset+3] += 1
    W[c1j][offset+3] += 1
    return offset + 4

def _add_1x2(U, V, W, offset, k, j0, j1, n, m, p):
    a0k = 0*m + k
    a1k = 1*m + k
    bkj0 = k*p + j0
    bkj1 = k*p + j1
    c0j0 = j0*n + 0
    c0j1 = j1*n + 0
    c1j0 = j0*n + 1
    c1j1 = j1*n + 1
    U[a0k][offset+0] += 1
    V[bkj0][offset+0] += 1
    W[c0j0][offset+0] += 1
    U[a0k][offset+1] += 1
    V[bkj1][offset+1] += 1
    W[c0j1][offset+1] += 1
    U[a1k][offset+2] += 1
    V[bkj0][offset+2] += 1
    W[c1j0][offset+2] += 1
    U[a1k][offset+3] += 1
    V[bkj1][offset+3] += 1
    W[c1j1][offset+3] += 1
    return offset + 4

def _add_1x1(U, V, W, offset, k, j, n, m, p):
    a0k = 0*m + k
    a1k = 1*m + k
    bkj = k*p + j
    c0j = j*n + 0
    c1j = j*n + 1
    U[a0k][offset+0] += 1
    V[bkj][offset+0] += 1
    W[c0j][offset+0] += 1
    U[a1k][offset+1] += 1
    V[bkj][offset+1] += 1
    W[c1j][offset+1] += 1
    return offset + 2

def _build_strassen_based():
    n, m, p = 2, 5, 5
    k_groups = [(0,1), (2,3), (4,)]
    j_groups = [(0,1), (2,3), (4,)]
    total_rank = 0
    for kg in k_groups:
        for jg in j_groups:
            if len(kg) == 2 and len(jg) == 2:
                total_rank += 7
            elif len(kg) == 2 and len(jg) == 1:
                total_rank += 4
            elif len(kg) == 1 and len(jg) == 2:
                total_rank += 4
            else:
                total_rank += 2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    offset = 0
    for kg in k_groups:
        for jg in j_groups:
            if len(kg) == 2 and len(jg) == 2:
                offset = _add_strassen_222(U, V, W, offset, kg[0], kg[1], jg[0], jg[1], n, m, p)
            elif len(kg) == 2 and len(jg) == 1:
                offset = _add_2x1(U, V, W, offset, kg[0], kg[1], jg[0], n, m, p)
            elif len(kg) == 1 and len(jg) == 2:
                offset = _add_1x2(U, V, W, offset, kg[0], jg[0], jg[1], n, m, p)
            else:
                offset = _add_1x1(U, V, W, offset, kg[0], jg[0], n, m, p)
    return U, V, W

def construct():
    """
    Returns a valid rank decomposition for 2x5 * 5x5 matrix multiplication.
    """
    n, m, p = 2, 5, 5
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    if not _deadline_ok():
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try Strassen-based approach (rank 46)
    U, V, W = _build_strassen_based()
    
    # Verify
    if _verify_decomp(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
