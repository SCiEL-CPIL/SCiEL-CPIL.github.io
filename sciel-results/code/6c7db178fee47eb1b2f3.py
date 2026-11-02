import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                target[a, b, c] = 1.0
    decomp = np.einsum('it,jt,kt->ijk', U, V, W)
    diff = np.abs(decomp - target)
    return np.max(diff) < tol

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen222():
    # Rank-7 Strassen decomposition of the 2x2x2 matrix multiplication tensor.
    # Term t contributes: sum over (a,b,c) of u_t[a]*v_t[b]*w_t[c] * e_a e_b e_c
    # where a = i*2+k (A row i, col k), b = k*2+j (B row k, col j), c = j*2+i (C col j, row i)
    # Strassen products:
    # M1=(A00+A11)(B00+B11)
    # M2=(A10+A11)B00
    # M3=A00(B01-B11)
    # M4=A11(B10-B00)
    # M5=(A00+A10)B10
    # M6=(A11-A01)(B00+B01)
    # M7=(A01-A11)(B10+B11)
    # C00=M1+M4-M5+M7 ; C01=M3+M5 ; C10=M2+M4 ; C11=M1-M2+M3+M6
    terms = []
    def vec(idx_map):
        # idx_map: list of (idx, coeff)
        v = [0.0] * 4
        for idx, c in idx_map:
            v[idx] += c
        return v
    # A indices: 00->0, 01->1, 10->2, 11->3
    A00, A01, A10, A11 = 0, 1, 2, 3
    # B indices: 00->0, 01->1, 10->2, 11->3
    B00, B01, B10, B11 = 0, 1, 2, 3
    # C indices: 00->0, 01->1, 10->2, 11->3
    C00, C01, C10, C11 = 0, 1, 2, 3

    M = {
        'M1': ([A00, A11], [B00, B11]),
        'M2': ([A10, A11], [B00]),
        'M3': ([A00], [B01, B11]),
        'M4': ([A11], [B10, B00]),
        'M5': ([A00, A10], [B10]),
        'M6': ([A11, A01], [B00, B01]),
        'M7': ([A01, A11], [B10, B11]),
    }
    # coefficients for M3: A00*(B01 - B11) -> B01:+1, B11:-1
    M['M3'] = ([A00], [(B01, 1), (B11, -1)])
    M['M4'] = ([A11], [(B10, 1), (B00, -1)])
    M['M6'] = ([A11, A01], [(B00, 1), (B01, 1)])  # (A11-A01)(B00+B01): A11:+1, A01:-1
    M['M6'] = ([(A11, 1), (A01, -1)], [(B00, 1), (B01, 1)])
    M['M7'] = ([(A01, 1), (A11, -1)], [(B10, 1), (B11, 1)])
    # fix M1, M2, M5 to tuple-of-lists-of-(idx,coeff) form
    M['M1'] = ([(A00, 1), (A11, 1)], [(B00, 1), (B11, 1)])
    M['M2'] = ([(A10, 1), (A11, 1)], [(B00, 1)])
    M['M3'] = ([(A00, 1)], [(B01, 1), (B11, -1)])
    M['M4'] = ([(A11, 1)], [(B10, 1), (B00, -1)])
    M['M5'] = ([(A00, 1), (A10, 1)], [(B10, 1)])
    M['M6'] = ([(A11, 1), (A01, -1)], [(B00, 1), (B01, 1)])
    M['M7'] = ([(A01, 1), (A11, -1)], [(B10, 1), (B11, 1)])

    C_contrib = {
        'M1': [(C00, 1), (C11, 1)],
        'M2': [(C10, 1), (C11, -1)],
        'M3': [(C01, 1), (C11, 1)],
        'M4': [(C00, 1), (C10, 1)],
        'M5': [(C00, -1), (C01, 1)],
        'M6': [(C11, 1)],
        'M7': [(C00, 1)],
    }

    for name in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7']:
        ua, ub = M[name]
        u = [0.0] * 4
        for idx, c in ua:
            u[idx] = c
        v = [0.0] * 4
        for idx, c in ub:
            v[idx] = c
        w = [0.0] * 4
        for idx, c in C_contrib[name]:
            w[idx] = c
        terms.append((u, v, w))
    return terms

def construct():
    n, m, p = 3, 5, 6
    deadline = time.time() + 130.0

    # Split n=3 -> 2+1, m=5 -> 2+2+1, p=6 -> 2+2+2
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    assert sum(n_parts) == n and sum(m_parts) == m and sum(p_parts) == p

    strassen = _strassen222()

    U = [[0] * 0 for _ in range(n * m)]  # placeholder, rebuilt
    # We'll accumulate terms as lists of (U_vec, V_vec, W_vec) then pack.
    terms = []  # each: (u_list len n*m, v_list len m*p, w_list len p*n)

    n_off = 0
    for a, n_a in enumerate(n_parts):
        m_off = 0
        for b, m_b in enumerate(m_parts):
            p_off = 0
            for c, p_c in enumerate(p_parts):
                # Block of size n_a x m_b x p_c
                # A rows: n_off..n_off+n_a-1, A cols: m_off..m_off+m_b-1
                # B rows: m_off..m_off+m_b-1, B cols: p_off..p_off+p_c-1
                # C rows: n_off..n_off+n_a-1, C cols: p_off..p_off+p_c-1
                if n_a == 2 and m_b == 2 and p_c == 2:
                    for (u, v, w) in strassen:
                        Uv = [0] * (n * m)
                        Vv = [0] * (m * p)
                        Wv = [0] * (p * n)
                        for i in range(2):
                            for k in range(2):
                                Uv[(n_off + i) * m + (m_off + k)] = u[i * 2 + k]
                        for k in range(2):
                            for j in range(2):
                                Vv[(m_off + k) * p + (p_off + j)] = v[k * 2 + j]
                        for j in range(2):
                            for i in range(2):
                                Wv[(p_off + j) * n + (n_off + i)] = w[j * 2 + i]
                        terms.append((Uv, Vv, Wv))
                else:
                    # schoolbook for this block
                    for i in range(n_a):
                        for k in range(m_b):
                            for j in range(p_c):
                                Uv = [0] * (n * m)
                                Vv = [0] * (m * p)
                                Wv = [0] * (p * n)
                                Uv[(n_off + i) * m + (m_off + k)] = 1
                                Vv[(m_off + k) * p + (p_off + j)] = 1
                                Wv[(p_off + j) * n + (n_off + i)] = 1
                                terms.append((Uv, Vv, Wv))
                p_off += p_c
            m_off += m_b
        n_off += n_a

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for col, (Uv, Vv, Wv) in enumerate(terms):
        for r in range(n * m):
            if Uv[r] != 0:
                U[r][col] = Uv[r]
        for r in range(m * p):
            if Vv[r] != 0:
                V[r][col] = Vv[r]
        for r in range(p * n):
            if Wv[r] != 0:
                W[r][col] = Wv[r]

    result = {"U": U, "V": V, "W": W}

    # Verify; fall back to schoolbook if it fails.
    if _verify(U, V, W, n, m, p):
        return result
    return _schoolbook(n, m, p)
