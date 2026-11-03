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
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    
    # We use 4 non-overlapping 2x2x2 Strassen blocks.
    # n=3, m=5, p=5.
    # Blocks cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    # Specifically:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    strassen_terms = _strassen_2x2x2()
    
    # Mark covered triples
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i0, j0, k0 in blocks:
        for u_sp, v_sp, w_sp in strassen_terms:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for u_rel, coeff in u_sp.items():
                u_i = u_rel // 2
                u_j = u_rel % 2
                u_idx = (i0 + u_i) * m + (j0 + u_j)
                u_col[u_idx] = coeff
                # Mark as covered
                covered[i0 + u_i][j0 + u_j][k0 + (u_rel % 2)] = True # This is tricky, u_rel doesn't map to k directly in U.
                # U is A_{i,j}. The block is defined by i0,j0,k0.
                # The Strassen term uses A_{i0+di, j0+dj}.
                # The corresponding B is B_{j0+dj, k0+dk}.
                # The corresponding C is C_{k0+dk, i0+di}.
                # The sparse dicts give the local indices.
                # u_rel = di*2 + dj.
                # v_rel = dj*2 + dk.
                # w_rel = dk*2 + di.
                # So for a given term, the set of (di, dj, dk) is not a single triple, but a set of interactions.
                # We should mark the entire 2x2x2 block as covered.
            
            # Simpler: just mark the whole 2x2x2 block as covered.
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[i0+di][j0+dj][k0+dk] = True
            
            for v_rel, coeff in v_sp.items():
                v_j = v_rel // 2
                v_k = v_rel % 2
                v_idx = (j0 + v_j) * p + (k0 + v_k)
                v_col[v_idx] = coeff
            
            for w_rel, coeff in w_sp.items():
                w_k = w_rel // 2
                w_i = w_rel % 2
                w_idx = (k0 + w_k) * n + (i0 + w_i)
                w_col[w_idx] = coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # Schoolbook for remaining triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    u_col = [0] * (n * m)
                    v_col = [0] * (m * p)
                    w_col = [0] * (p * n)
                    
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    u_col[u_idx] = 1
                    v_col[v_idx] = 1
                    w_col[w_idx] = 1
                    
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
    
    rank = len(U_cols)
    
    # Transpose to (rows, rank)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        return _schoolbook_fallback(n, m, p)
    
    if time.time() > deadline:
        return _schoolbook_fallback(n, m, p)
    
    return {"U": U, "V": V, "W": W}
