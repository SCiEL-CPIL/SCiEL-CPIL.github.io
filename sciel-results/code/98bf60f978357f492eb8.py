import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_start_time = time.time()
TIME_LIMIT = 140

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 matrix multiplication.
    U: shape (4, 7)
    V: shape (4, 7)
    W: shape (4, 7), W[k*2+i, r]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0], # A00, A00, A00, A00, A00+A01, A10-A00, A01-A11
        [0, 0, 0, 0, 1, 0, 1],  # A01, A01, A01, A01, A00+A01, A10-A00, A01-A11
        [0, 1, 0, 0, 0, 1, 0],  # A10, A10, A10, A10, A00+A01, A10-A00, A01-A11
        [1, 1, 0, 1, 0, 0, -1]  # A11, A10+A11, A11, A11, A00+A01, A10-A00, A01-A11
    ]
    V = [
        [1, 1, 0, 0, 0, 1, 0],  # B00, B00+B11, B00, B01-B00, B11, B00+B01, B10+B11
        [0, 0, 1, 1, 0, 1, 1],  # B01, B00+B11, B10-B11, B01-B00, B11, B00+B01, B10+B11
        [1, 0, 1, 0, 0, 0, 1],  # B10, B00+B11, B10-B11, B01-B00, B11, B00+B01, B10+B11
        [1, 0, -1, -1, 0, 0, 1] # B11, B00+B11, B10-B11, B01-B00, B11, B00+B01, B10+B11
    ]
    W = [
        [1, 0, 0, -1, -1, 1, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, -1, 0, 0, 0, 1]  # C11
    ]
    return U, V, W

def _schoolbook(m, n, p):
    rank = m * n * p
    U = [[0] * rank for _ in range(m * n)]
    V = [[0] * rank for _ in range(n * p)]
    W = [[0] * rank for _ in range(p * m)]
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j][col] = 1
                V[j * p + k][col] = 1
                W[k * m + i][col] = 1
                col += 1
    return U, V, W

_decompose_cache = {}

def _decompose(m, n, p):
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()
    
    key = (m, n, p)
    if key in _decompose_cache:
        return _decompose_cache[key]
    
    if m == 0 or n == 0 or p == 0:
        return [], [], []
    
    if m == 1 and n == 1 and p == 1:
        res = ([[1]], [[1]], [[1]])
        _decompose_cache[key] = res
        return res
    
    if m == 1:
        res = _schoolbook(1, n, p)
        _decompose_cache[key] = res
        return res
        
    if p == 1:
        res = _schoolbook(m, n, 1)
        _decompose_cache[key] = res
        return res
        
    if n == 1:
        res = _schoolbook(m, 1, p)
        _decompose_cache[key] = res
        return res
        
    if m == 2 and n == 2 and p == 2:
        res = _strassen_2x2x2()
        _decompose_cache[key] = res
        return res
    
    best = None
    best_rank = m * n * p
    
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                
                try:
                    U1, V1, W1 = _decompose(m1, n1, p1)
                    U2, V2, W2 = _decompose(m1, n1, p2)
                    U3, V3, W3 = _decompose(m1, n2, p1)
                    U4, V4, W4 = _decompose(m1, n2, p2)
                    U5, V5, W5 = _decompose(m2, n1, p1)
                    U6, V6, W6 = _decompose(m2, n1, p2)
                    U7, V7, W7 = _decompose(m2, n2, p1)
                    U8, V8, W8 = _decompose(m2, n2, p2)
                    
                    total_rank = (len(U1) + len(U2) + len(U3) + len(U4) +
                                 len(U5) + len(U6) + len(U7) + len(U8))
                    
                    if total_rank < best_rank:
                        best_rank = total_rank
                        U_new = []
                        V_new = []
                        W_new = []
                        
                        def add_decomp(U_s, V_s, W_s, m_i, n_i, p_i, offset_m, offset_n, offset_p):
                            for r in range(len(U_s)):
                                u_row = [0] * (m * n)
                                v_row = [0] * (n * p)
                                w_row = [0] * (p * m)
                                for ii in range(m_i):
                                    for jj in range(n_i):
                                        u_row[(offset_m + ii) * n + (offset_n + jj)] = U_s[r][ii * n_i + jj]
                                for jj in range(n_i):
                                    for kk in range(p_i):
                                        v_row[(offset_n + jj) * p + (offset_p + kk)] = V_s[r][jj * p_i + kk]
                                for kk in range(p_i):
                                    for ii in range(m_i):
                                        w_row[(offset_p + kk) * m + (offset_m + ii)] = W_s[r][kk * m_i + ii]
                                U_new.append(u_row)
                                V_new.append(v_row)
                                W_new.append(w_row)
                        
                        add_decomp(U1, V1, W1, m1, n1, p1, 0, 0, 0)
                        add_decomp(U3, V3, W3, m1, n2, p1, 0, n1, 0)
                        add_decomp(U2, V2, W2, m1, n1, p2, 0, 0, p1)
                        add_decomp(U4, V4, W4, m1, n2, p2, 0, n1, p1)
                        add_decomp(U5, V5, W5, m2, n1, p1, m1, 0, 0)
                        add_decomp(U6, V6, W6, m2, n1, p2, m1, 0, p1)
                        add_decomp(U7, V7, W7, m2, n2, p1, m1, n1, 0)
                        add_decomp(U8, V8, W8, m2, n2, p2, m1, n1, p1)
                        
                        best = (U_new, V_new, W_new)
                except TimeoutError:
                    raise
                except Exception:
                    continue
    
    if best is not None:
        _decompose_cache[key] = best
        return best
    else:
        res = _schoolbook(m, n, p)
        _decompose_cache[key] = res
        return res

def construct():
    global _start_time
    _start_time = time.time()
    n, m, p = DIMENSIONS
    U, V, W = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
