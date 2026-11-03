import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0

def _time_left():
    return _DEADLINE - time.time()

def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return bool(np.all(np.abs(T - 1.0) < tol))

def _build_strassen_2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a11+a22)*(b11+b22)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # P2 = (a21+a22)*b11
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    
    # P3 = a11*(b12-b22)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    # P4 = a22*(b21-b11)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # P5 = (a11+a12)*b22
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _build_strassen_4x4x2():
    U2, V2, W2 = _build_strassen_2x2()
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    def add_strassen_block(a_i_start, a_j_start, b_j_start, c_i_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row, t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = a_j_start + lj
                        u_row = global_i * 4 + global_j
                        U[u_row, col] += coeff_a
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row, t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 2 + global_k
                        V[v_row, col] += coeff_b
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row, t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = c_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row, col] += coeff_c

    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 2, 2, 0, 0, 7)
    add_strassen_block(2, 0, 0, 2, 0, 14)
    add_strassen_block(2, 2, 2, 2, 0, 21)
    
    return U, V, W

def _build_strassen_4x4x4():
    U2, V2, W2 = _build_strassen_2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    c1 = U2[i1 * 2 + j1, r1]
                    if c1 == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            c2 = U2[i2 * 2 + j2, r2]
                            if c2 == 0: continue
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j, col] += c1 * c2
            for j1 in range(2):
                for k1 in range(2):
                    c1 = V2[j1 * 2 + k1, r1]
                    if c1 == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            c2 = V2[j2 * 2 + k2, r2]
                            if c2 == 0: continue
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k, col] += c1 * c2
            for k1 in range(2):
                for i1 in range(2):
                    c1 = W2[k1 * 2 + i1, r1]
                    if c1 == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            c2 = W2[k2 * 2 + i2, r2]
                            if c2 == 0: continue
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i, col] += c1 * c2
    
    return U, V, W

def _build_4x5x6_improved():
    n, m, p = 4, 5, 6
    U444, V444, W444 = _build_strassen_4x4x4()
    U442, V442, W442 = _build_strassen_4x4x2()
    
    total_rank = 49 + 28 + 24
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    base = 0
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+49] = U444[i * 4 + j, :]
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, base:base+49] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, base:base+49] = W444[k * 4 + i, :]
    
    base = 49
    for i in range(4):
        for j in range(4):
            U[i * 5 + j, base:base+28] = U442[i * 4 + j, :]
    for j in range(4):
        for k_local in range(2):
            k_global = 4 + k_local
            V[j * 6 + k_global, base:base+28] = V442[j * 2 + k_local, :]
    for k_local in range(2):
        k_global = 4 + k_local
        for i in range(4):
            W[k_global * 4 + i, base:base+28] = W442[k_local * 4 + i, :]
    
    base = 49 + 28
    for i in range(4):
        for k in range(6):
            col = base + i * 6 + k
            U[i * 5 + 4, col] = 1.0
            V[4 * 6 + k, col] = 1.0
            W[k * 4 + i, col] = 1.0
    
    return U, V, W

def _build_schoolbook(n, m, p):
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _hypergraph_greedy_reduce(U, V, W, n, m, p, budget_s=20.0):
    """
    Treat each column r as a hyperedge { (i,j), (j,k), (k,i) : all nonzero in U,V,W }.
    Greedily remove columns whose contribution is redundant (residual still matches 1.0).
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T_full = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    removed = []
    for _ in range(R):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        found = False
        for c in range(R):
            if c in removed:
                continue
            T_c = np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
            T_remainder = T_full - T_c
            if np.all(np.abs(T_remainder - 1.0) < 1e-6):
                removed.append(c)
                T_full = T_remainder
                found = True
                break
        if not found:
            break
    
    if removed:
        mask = np.ones(R, dtype=bool)
        mask[removed] = False
        U = U[:, mask]
        V = V[:, mask]
        W = W[:, mask]
    
    return U, V, W

def _local_reduce(U, V, W, n, m, p, max_iters=30, budget_s=40.0):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    for _ in range(max_iters):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        if R < 2:
            break
        found = False
        cols = list(range(R))
        rng.shuffle(cols)
        for c in cols:
            if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
                break
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _verify_decomp(U2, V2, W2, n, m, p, 1e-6):
                U, V, W = U2, V2, W2
                R -= 1
                found = True
                break
        if not found:
            break
    return U, V, W

def _try_remove_fast(U, V, W, n, m, p, budget_s=20.0):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    start_time = time.time()
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T_full = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    for c in range(R):
        if _time_left() < 5.0 or (time.time() - start_time) > budget_s:
            break
        T_c = np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
        T_remainder = T_full - T_c
        if np.all(np.abs(T_remainder - 1.0) < 1e-6):
            U2 = np.delete(U, c, axis=1)
            V2 = np.delete(V, c, axis=1)
            W2 = np.delete(W, c, axis=1)
            if _verify_decomp(U2, V2, W2, n, m, p, 1e-9):
                return U2, V2, W2
    
    return U, V, W

def construct():
    n, m, p = 4, 5, 6
    best_rank = n * m * p
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    
    try:
        if _time_left() > 30.0:
            U, V, W = _build_4x5x6_improved()
            if _verify_decomp(U, V, W, n, m, p, 1e-9):
                if U.shape[1] < best_rank:
                    best_rank = U.shape[1]
                    best_U, best_V, best_W = U, V, W
    except Exception:
        pass
    
    # Hypergraph greedy reduction
    try:
        if _time_left() > 15.0:
            cand = _hypergraph_greedy_reduce(best_U, best_V, best_W, n, m, p, budget_s=20.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Fast column removal
    try:
        if _time_left() > 15.0:
            cand = _try_remove_fast(best_U, best_V, best_W, n, m, p, budget_s=15.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    # Local reduction
    try:
        if _time_left() > 10.0:
            cand = _local_reduce(best_U, best_V, best_W, n, m, p, max_iters=30, budget_s=30.0)
            if _verify_decomp(cand[0], cand[1], cand[2], n, m, p, 1e-9):
                if cand[0].shape[1] < best_rank:
                    best_rank = cand[0].shape[1]
                    best_U, best_V, best_W = cand
    except Exception:
        pass
    
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, 1e-9):
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": W.tolist()}
