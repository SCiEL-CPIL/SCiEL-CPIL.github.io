import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]
DEADLINE = 145.0

def _verify_decomposition(U, V, W):
    n, m, p = 2, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_36():
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1

    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _round_to_half_int(X):
    return np.round(2 * X) / 2.0

def _alternating_minimize(U, V, W, n_iter=100, tol=1e-10):
    n, m, p = 2, 4, 5
    r = U.shape[1]
    
    for it in range(n_iter):
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                b = np.ones(p)
                try:
                    U_new = np.linalg.lstsq(A, b, rcond=1e-10)[0]
                except:
                    continue
                U[i * m + j, :] = U_new
        
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, r))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                b = np.ones(n)
                try:
                    V_new = np.linalg.lstsq(A, b, rcond=1e-10)[0]
                except:
                    continue
                V[j * p + k, :] = V_new
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    W_new = np.linalg.lstsq(A, b, rcond=1e-10)[0]
                except:
                    continue
                W[k * n + i, :] = W_new
        
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        if max_err < tol:
            break
    
    return U, V, W, max_err

def _try_rank_reduction(U, V, W, target_rank, max_trials=50):
    n, m, p = 2, 4, 5
    rng = np.random.RandomState(42)
    
    for trial in range(max_trials):
        U_init = rng.randn(n * m, target_rank) * 0.1
        V_init = rng.randn(m * p, target_rank) * 0.1
        W_init = rng.randn(p * n, target_rank) * 0.1
        
        U_sol, V_sol, W_sol, err = _alternating_minimize(U_init, V_init, W_init, n_iter=200, tol=1e-12)
        
        if err > 1e-8:
            continue
        
        U_round = _round_to_half_int(U_sol)
        V_round = _round_to_half_int(V_sol)
        W_round = _round_to_half_int(W_sol)
        
        if _verify_decomposition(U_round, V_round, W_round):
            return U_round, V_round, W_round
    
    return None

def _build_kronecker_matrix(U, V, W):
    n, m, p = 2, 4, 5
    r = U.shape[1]
    A = np.zeros((n * m * p, r))
    for c in range(r):
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A[idx, c] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    idx += 1
    return A

def _check_column_redundant_fast(K, col_to_remove):
    r = K.shape[1]
    if r <= 1:
        return False
    keep_cols = [c for c in range(r) if c != col_to_remove]
    A = K[:, keep_cols]
    b = K[:, col_to_remove]
    try:
        solution, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=1e-10)
    except:
        return False
    A_sol = A @ solution
    if np.max(np.abs(A_sol - b)) > 1e-8:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    U, V, W = _build_strassen_36()
    
    if not _verify_decomposition(U, V, W):
        n, m, p = 2, 4, 5
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    if time.time() < deadline - 20:
        result = _try_rank_reduction(U, V, W, 35, max_trials=30)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = 35
    
    if best_rank == 35 and time.time() < deadline - 10:
        result = _try_rank_reduction(U, V, W, 34, max_trials=20)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = 34
    
    if best_rank == 36 and time.time() < deadline - 5:
        K = _build_kronecker_matrix(U, V, W)
        progress = True
        while progress and time.time() < deadline - 5:
            progress = False
            rank = U.shape[1]
            for col in range(rank):
                if time.time() > deadline - 5:
                    break
                if _check_column_redundant_fast(K, col):
                    mask = [i for i in range(rank) if i != col]
                    U_new = U[:, mask]
                    V_new = V[:, mask]
                    W_new = W[:, mask]
                    if _verify_decomposition(U_new, V_new, W_new):
                        U, V, W = U_new, V_new, W_new
                        K = _build_kronecker_matrix(U, V, W)
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = U.shape[1]
                        progress = True
                        break
    
    if not _verify_decomposition(U, V, W):
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
