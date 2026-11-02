import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def _strassen_2x2x2():
    """
    Returns 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_sparse, v_sparse, w_sparse) where each is a dict {local_idx: coeff}.
    U local idx: i*2+j (i=row of A, j=col of A = row of B)
    V local idx: j*2+k (j=row of B, k=col of B)
    W local idx: k*2+i (k=col of C, i=row of C) -- trace convention W[k*n+i]
    """
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
    # M2 = (A21+A22)B11
    terms.append(({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}))
    # M3 = A11(B21-B22)
    terms.append(({0: 1}, {2: 1, 3: -1}, {2: 1, 3: 1}))
    # M4 = A22(B12-B11)
    terms.append(({3: 1}, {1: 1, 0: -1}, {0: 1, 1: 1}))
    # M5 = (A11+A12)B22
    terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}))
    # M6 = (A21-A11)(B11+B12)
    terms.append(({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}))
    # M7 = (A12-A22)(B21+B22)
    terms.append(({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}))
    return terms

def _schoolbook_fallback(n, m, p):
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}

def _verify_decomposition(U, V, W, n, m, p):
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        return np.allclose(T_decomp, target, atol=1e-10)
    except Exception:
        return False

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n, m, p = DIMENSIONS
    
    # Block decomposition: n=3 as [2,1], m=5 as [2,2,1], p=5 as [2,2,1]
    # This gives 4 Strassen 2x2x2 blocks (rank 7 each) + schoolbook for the rest
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    strassen_terms = _strassen_2x2x2()
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for ni, ni_sz in enumerate(n_parts):
        for mj, mj_sz in enumerate(m_parts):
            for pk, pk_sz in enumerate(p_parts):
                n_off = n_offsets[ni]
                m_off = m_offsets[mj]
                p_off = p_offsets[pk]
                
                if ni_sz == 2 and mj_sz == 2 and pk_sz == 2:
                    # Use Strassen: 7 rank-1 terms
                    for u_sp, v_sp, w_sp in strassen_terms:
                        u_col = [0] * (n * m)
                        v_col = [0] * (m * p)
                        w_col = [0] * (p * n)
                        
                        for u_rel, coeff in u_sp.items():
                            u_i = u_rel // 2
                            u_j = u_rel % 2
                            u_idx = (n_off + u_i) * m + (m_off + u_j)
                            u_col[u_idx] = coeff
                            
                        for v_rel, coeff in v_sp.items():
                            v_j = v_rel // 2
                            v_k = v_rel % 2
                            v_idx = (m_off + v_j) * p + (p_off + v_k)
                            v_col[v_idx] = coeff
                            
                        for w_rel, coeff in w_sp.items():
                            w_k = w_rel // 2
                            w_i = w_rel % 2
                            w_idx = (p_off + w_k) * n + (n_off + w_i)
                            w_col[w_idx] = coeff
                            
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                else:
                    # Schoolbook: ni_sz * mj_sz * pk_sz rank-1 terms
                    for i in range(ni_sz):
                        for j in range(mj_sz):
                            for k in range(pk_sz):
                                u_col = [0] * (n * m)
                                v_col = [0] * (m * p)
                                w_col = [0] * (p * n)
                                
                                u_idx = (n_off + i) * m + (m_off + j)
                                v_idx = (m_off + j) * p + (p_off + k)
                                w_idx = (p_off + k) * n + (n_off + i)
                                
                                u_col[u_idx] = 1
                                v_col[v_idx] = 1
                                w_col[w_idx] = 1
                                
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
    
    rank = len(U_cols)
    
    # Transpose: U is (n*m, rank), so U[row][col]
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return _schoolbook_fallback(n, m, p)
    
    if time.time() > deadline:
        return _schoolbook_fallback(n, m, p)
    
    return {"U": U, "V": V, "W": W}
