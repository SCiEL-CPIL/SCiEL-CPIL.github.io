import numpy as np
import time

DIMENSIONS = [3, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for the (i,j,k) triple."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    expected = _build_tensor(n, m, p)
    return np.allclose(T, expected, atol=tol)

def _naive_decomposition(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_222():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4,7) for A indices (i*2+j), V has shape (4,7) for B (j*2+k), W has shape (4,7) for C (k*2+i).
    """
    # M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-e), M5=(a+b)h, M6=(c-d)(f+g), M7=(b-d)(e+h)
    # C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    U = [
        [1, 0, 1, 0, 1, 0, 0],   # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, -1, -1], # d
    ]
    V = [
        [1, 1, 0, -1, 0, 0, 1],  # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 1, 0],   # g
        [1, 0, -1, 0, 1, 0, 1],  # h
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0],  # C22
    ]
    return U, V, W

def _try_strassen_block_decomposition(n, m, p):
    """Try to decompose n×m×p by splitting dimensions and using Strassen on 2×2×2 blocks."""
    # Split each dimension into parts of size <= 2
    def split_dim(d):
        parts = []
        while d > 2:
            parts.append(2)
            d -= 2
        parts.append(d)
        return parts
    
    ni = split_dim(n)
    mj = split_dim(m)
    pk = split_dim(p)
    
    total_rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    for ii, ni_size in enumerate(ni):
        for jj, mj_size in enumerate(mj):
            for kk, pk_size in enumerate(pk):
                if ni_size == 2 and mj_size == 2 and pk_size == 2:
                    # Use Strassen rank 7
                    U_s, V_s, W_s = _strassen_222()
                    # Map indices
                    i_off = sum(ni[:ii])
                    j_off = sum(mj[:jj])
                    k_off = sum(pk[:kk])
                    for r in range(7):
                        u_col = [0.0] * (n * m)
                        v_col = [0.0] * (m * p)
                        w_col = [0.0] * (p * n)
                        for a in range(2):
                            for b in range(2):
                                u_col[(i_off + a) * m + (j_off + b)] = U_s[a * 2 + b][r]
                        for b in range(2):
                            for c in range(2):
                                v_col[(j_off + b) * p + (k_off + c)] = V_s[b * 2 + c][r]
                        for c in range(2):
                            for a in range(2):
                                w_col[(k_off + c) * n + (i_off + a)] = W_s[c * 2 + a][r]
                        U_cols.append(u_col)
                        V_cols.append(v_col)
                        W_cols.append(w_col)
                    total_rank += 7
                else:
                    # Naive for this block
                    block_rank = ni_size * mj_size * pk_size
                    i_off = sum(ni[:ii])
                    j_off = sum(mj[:jj])
                    k_off = sum(pk[:kk])
                    for a in range(ni_size):
                        for b in range(mj_size):
                            for c in range(pk_size):
                                u_col = [0.0] * (n * m)
                                v_col = [0.0] * (m * p)
                                w_col = [0.0] * (p * n)
                                u_col[(i_off + a) * m + (j_off + b)] = 1.0
                                v_col[(j_off + b) * p + (k_off + c)] = 1.0
                                w_col[(k_off + c) * n + (i_off + a)] = 1.0
                                U_cols.append(u_col)
                                V_cols.append(v_col)
                                W_cols.append(w_col)
                    total_rank += block_rank
    
    U = [[U_cols[r][a] for r in range(total_rank)] for a in range(n * m)]
    V = [[V_cols[r][a] for r in range(total_rank)] for a in range(m * p)]
    W = [[W_cols[r][a] for r in range(total_rank)] for a in range(p * n)]
    
    return {"U": U, "V": V, "W": W}

def _try_random_search(n, m, p, max_rank, iterations, seed=42):
    """Try random search for a rank decomposition."""
    rng = np.random.RandomState(seed)
    for it in range(iterations):
        if time.time() > _START_TIME + TIME_LIMIT - 10:
            break
        rank = max_rank
        U = rng.randint(-2, 3, size=(n * m, rank)) * 0.5
        V = rng.randint(-2, 3, size=(m * p, rank)) * 0.5
        W = rng.randint(-2, 3, size=(p * n, rank)) * 0.5
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = _START_TIME + TIME_LIMIT
    
    # First verify Strassen decomposition works
    try:
        U_s, V_s, W_s = _strassen_222()
        # Quick verify 2x2x2
        T = np.zeros((4, 4, 4))
        for r in range(7):
            T += np.einsum('a,b,c->abc', np.array(U_s)[:, r], np.array(V_s)[:, r], np.array(W_s)[:, r])
        expected = _build_tensor(2, 2, 2)
        strassen_ok = np.allclose(T, expected, atol=1e-9)
    except Exception:
        strassen_ok = False
    
    if strassen_ok:
        try:
            result = _try_strassen_block_decomposition(n, m, p)
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                rank = len(result["U"][0])
                if rank < n * m * p:
                    return result
        except Exception:
            pass
    
    if time.time() > deadline - 15:
        return _naive_decomposition(n, m, p)
    
    # Try random search for lower ranks
    for target_rank in range(60, 84):
        if time.time() > deadline - 10:
            break
        result = _try_random_search(n, m, p, target_rank, iterations=100)
        if result is not None:
            return result
    
    return _naive_decomposition(n, m, p)
