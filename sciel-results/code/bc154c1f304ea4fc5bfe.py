import time
from fractions import Fraction

N, M, P = 2, 4, 5
_T0 = time.time()
_DEADLINE = _T0 + 110.0

def _elapsed():
    return time.time() - _T0

def _add_strassen_block(U, V, W, col_offset, i_offset, j_offset, k_offset, n, m, p):
    """
    Adds a Strassen 2x2x2 decomposition for a sub-block.
    U rows: (i_offset + i) * m + (j_offset + j) for i,j in {0,1}
    V rows: (j_offset + j) * p + (k_offset + k) for j,k in {0,1}
    W rows: (k_offset + k) * n + (i_offset + i) for k,i in {0,1}
    """
    # Strassen coefficients
    # U (A part)
    sU = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    # V (B part)
    sV = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 0, 1],  # B01
        [0, 0, 0, 1, 0, 1, 0],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    # W (C part)
    sW = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                u_idx = (i_offset + i) * m + (j_offset + j)
                U[u_idx][col_offset + r] += sU[i*2+j][r]
        for j in range(2):
            for k in range(2):
                v_idx = (j_offset + j) * p + (k_offset + k)
                V[v_idx][col_offset + r] += sV[j*2+k][r]
        for k in range(2):
            for i in range(2):
                w_idx = (k_offset + k) * n + (i_offset + i)
                W[w_idx][col_offset + r] += sW[k*2+i][r]
    return col_offset + 7

def _add_schoolbook_block(U, V, W, col_offset, i_offset, j_offset, k_offset, n, m, p, num_i, num_j, num_k):
    """
    Adds a schoolbook decomposition for a sub-block.
    """
    for i in range(num_i):
        for j in range(num_j):
            for k in range(num_k):
                u_idx = (i_offset + i) * m + (j_offset + j)
                v_idx = (j_offset + j) * p + (k_offset + k)
                w_idx = (k_offset + k) * n + (i_offset + i)
                U[u_idx][col_offset] = 1
                V[v_idx][col_offset] = 1
                W[w_idx][col_offset] = 1
                col_offset += 1
    return col_offset

def _build_rank36():
    n, m, p = N, M, P
    rank = 36
    U = [[Fraction(0) for _ in range(rank)] for _ in range(n * m)]
    V = [[Fraction(0) for _ in range(rank)] for _ in range(m * p)]
    W = [[Fraction(0) for _ in range(rank)] for _ in range(p * n)]
    
    col = 0
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen (7)
    col = _add_strassen_block(U, V, W, col, 0, 0, 0, n, m, p)
    
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1} -> Strassen (7)
    col = _add_strassen_block(U, V, W, col, 0, 2, 0, n, m, p)
    
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3,4}
    # Split k into {2,3} (Strassen 7) and {4} (Schoolbook 4)
    # 3a: j in {0,1}, k in {2,3}
    col = _add_strassen_block(U, V, W, col, 0, 0, 2, n, m, p)
    # 3b: j in {0,1}, k in {4}
    col = _add_schoolbook_block(U, V, W, col, 0, 0, 4, n, m, p, 2, 2, 1)
    
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3,4}
    # Split k into {2,3} (Strassen 7) and {4} (Schoolbook 4)
    # 4a: j in {2,3}, k in {2,3}
    col = _add_strassen_block(U, V, W, col, 0, 2, 2, n, m, p)
    # 4b: j in {2,3}, k in {4}
    col = _add_schoolbook_block(U, V, W, col, 0, 2, 4, n, m, p, 2, 2, 1)
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                acc = Fraction(0)
                for r in range(rank):
                    acc += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                expected = Fraction(1) if i == k else Fraction(0)
                if acc != expected:
                    return False
    return True

def _to_list(mat):
    out = []
    for row in mat:
        r = []
        for x in row:
            f = x.numerator / x.denominator
            if f == int(f):
                r.append(int(f))
            else:
                r.append(f)
        out.append(r)
    return out

def construct():
    try:
        if _elapsed() > _DEADLINE:
            return _fallback()
            
        U, V, W = _build_rank36()
        n, m, p = N, M, P
        ok = _verify(U, V, W, n, m, p)
        if not ok:
            return _fallback()
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    except Exception:
        return _fallback()

def _fallback():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
