import time
import numpy as np

DEADLINE_S = 110.0
N, M, P = 4, 5, 5


def _validate_fast(U, V, W, tol=1e-9):
    """Vectorized check using einsum."""
    n, m, p = N, M, P
    R = U.shape[1]
    U_r = U.reshape(n, m, R)
    V_r = V.reshape(m, p, R)
    W_r = W.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.all(np.abs(T - 1.0) <= tol)


def _strassen_2x2x2():
    """Return (U,V,W) for rank-7 Strassen decomposition of 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = -1; W[3, 0] = -1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[3, 4] = -1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[2, 6] = 1
    
    return U, V, W


def _strassen_4x4x4():
    """Return (U,V,W) for rank-49 Strassen decomposition of 4x4x4 matrix multiplication."""
    s2U, s2V, s2W = _strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    strassen_ops = [
        ({'A00': 1, 'A11': 1}, {'B00': 1, 'B11': 1}, {'C00': 1, 'C11': 1, 'C01': -1, 'C10': -1}),
        ({'A10': 1, 'A11': 1}, {'B00': 1}, {'C10': 1}),
        ({'A00': 1}, {'B10': 1, 'B11': -1}, {'C01': 1, 'C11': -1}),
        ({'A11': 1}, {'B01': 1, 'B00': -1}, {'C00': -1, 'C01': 1}),
        ({'A00': 1, 'A01': 1}, {'B11': 1}, {'C00': 1, 'C11': -1}),
        ({'A10': 1, 'A00': -1}, {'B00': 1, 'B10': 1}, {'C11': 1}),
        ({'A01': 1, 'A11': -1}, {'B01': 1, 'B11': 1}, {'C00': -1, 'C10': 1}),
    ]
    
    block_U = {'A00': (0, 4), 'A01': (2, 6), 'A10': (4, 8), 'A11': (6, 10)}
    block_V = {'B00': (0, 4), 'B01': (2, 6), 'B10': (4, 8), 'B11': (6, 10)}
    block_W = {'C00': (0, 4), 'C01': (2, 6), 'C10': (4, 8), 'C11': (6, 10)}
    
    for op_idx, (u_blocks, v_blocks, w_blocks) in enumerate(strassen_ops):
        base_r = op_idx * 7
        for r2 in range(7):
            r = base_r + r2
            for blk, c in u_blocks.items():
                s, e = block_U[blk]
                for i2 in range(2):
                    for j2 in range(2):
                        U[s + i2*2 + j2, r] += c * s2U[i2*2 + j2, r2]
            for blk, c in v_blocks.items():
                s, e = block_V[blk]
                for j2 in range(2):
                    for k2 in range(2):
                        V[s + j2*2 + k2, r] += c * s2V[j2*2 + k2, r2]
            for blk, c in w_blocks.items():
                s, e = block_W[blk]
                for k2 in range(2):
                    for i2 in range(2):
                        W[s + k2*2 + i2, r] += c * s2W[k2*2 + i2, r2]
    
    return U, V, W


def _make_schoolbook(n_b, m_b, p_b, i_off, j_off, k_off):
    """Create schoolbook for a sub-block embedded in full space."""
    n, m, p = N, M, P
    rank = n_b * m_b * p_b
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    c = 0
    for i in range(n_b):
        for j in range(m_b):
            for k in range(p_b):
                U[(i_off+i)*m + (j_off+j), c] = 1.0
                V[(j_off+j)*p + (k_off+k), c] = 1.0
                W[(k_off+k)*n + (i_off+i), c] = 1.0
                c += 1
    return U, V, W


def _combine_decompositions(decomps):
    """Combine a list of (U,V,W) decompositions into one."""
    if not decomps:
        return None
    n, m, p = N, M, P
    total_rank = sum(U.shape[1] for U, V, W in decomps)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    c = 0
    for U_i, V_i, W_i in decomps:
        r = U_i.shape[1]
        U[:, c:c+r] = U_i
        V[:, c:c+r] = V_i
        W[:, c:c+r] = W_i
        c += r
    return U, V, W


def _peel_fast(U, V, W, n, m, p, tol=1e-6, max_rounds=50, time_limit=30.0):
    """Fast peeling using vectorized validation."""
    t0 = time.time()
    U_cur, V_cur, W_cur = U.copy(), V.copy(), W.copy()
    R = U_cur.shape[1]
    
    for _ in range(max_rounds):
        if time.time() - t0 > time_limit:
            break
        if R < 2:
            break
        
        found = False
        for r_try in range(R):
            mask = [i for i in range(R) if i != r_try]
            U_new = U_cur[:, mask]
            V_new = V_cur[:, mask]
            W_new = W_cur[:, mask]
            
            if _validate_fast(U_new, V_new, W_new, tol=tol):
                U_cur, V_cur, W_cur = U_new, V_new, W_new
                R -= 1
                found = True
                break
        
        if not found:
            break
    
    return U_cur, V_cur, W_cur


def construct():
    t0 = time.time()
    n, m, p = N, M, P  # 4, 5, 5
    
    if time.time() - t0 > DEADLINE_S:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    decomps = []
    decomps.append(_make_schoolbook(4, 1, 1, 0, 0, 0))
    decomps.append(_make_schoolbook(4, 1, 4, 0, 0, 1))
    decomps.append(_make_schoolbook(4, 4, 1, 0, 1, 0))
    
    s4U, s4V, s4W = _strassen_4x4x4()
    U_embed = np.zeros((n*m, 49))
    V_embed = np.zeros((m*p, 49))
    W_embed = np.zeros((p*n, 49))
    for i in range(4):
        for j in range(4):
            U_embed[i*5 + (1+j), :] = s4U[i*4+j, :]
    for j in range(4):
        for k in range(4):
            V_embed[(1+j)*5 + (1+k), :] = s4V[j*4+k, :]
    for k in range(4):
        for i in range(4):
            W_embed[(1+k)*4 + i, :] = s4W[k*4+i, :]
    decomps.append((U_embed, V_embed, W_embed))
    
    U, V, W = _combine_decompositions(decomps)
    
    if time.time() - t0 > DEADLINE_S - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if not _validate_fast(U, V, W):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1.0
                    V[j*p+k, c] = 1.0
                    W[k*n+i, c] = 1.0
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    remaining_time = DEADLINE_S - 15 - (time.time() - t0)
    if remaining_time > 5:
        U_red, V_red, W_red = _peel_fast(U, V, W, n, m, p, tol=1e-6, max_rounds=100, time_limit=remaining_time)
        if U_red.shape[1] < U.shape[1]:
            U, V, W = U_red, V_red, W_red
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
