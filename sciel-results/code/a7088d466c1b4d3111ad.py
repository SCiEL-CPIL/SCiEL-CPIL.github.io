import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    U_mat = U.reshape(n, m, rank)
    V_mat = V.reshape(m, p, rank)
    W_mat = W.reshape(p, n, rank)
    
    for i in range(n):
        for k in range(p):
            Uw = U_mat[i] * W_mat[k, i]
            M_ik = Uw @ V_mat[:, k].T
            if not np.allclose(M_ik, np.eye(m), atol=tol):
                return False
    return True

def _is_valid_entries(U, V, W):
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _build_strassen55(n, m, p):
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def _build_schoolbook(n, m, p):
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

def _try_drop_single(U55, V55, W55, n, m, p, deadline):
    """Try dropping each individual term and solving for corrections."""
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)
    rank55 = 55
    target_rank = 54
    
    num_eqs = n * m * p
    
    for drop_idx in range(rank55):
        if time.time() > deadline - 5:
            break
        
        keep_idx = [i for i in range(rank55) if i != drop_idx]
        U_keep = U55[:, keep_idx]
        V_keep = V55[:, keep_idx]
        W_keep = W55[:, keep_idx]
        
        num_unknowns = p * n * target_rank
        A = np.zeros((num_eqs, num_unknowns), dtype=np.float64)
        b = np.ones(num_eqs, dtype=np.float64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    for r in range(target_rank):
                        c_idx = k * n + i
                        col = c_idx * target_rank + r
                        A[eq_idx, col] = U_keep[a_idx, r] * V_keep[b_idx, r]
                    eq_idx += 1
        
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        valid = True
        for cv in coeffs:
            if abs(cv * 2 - round(cv * 2)) > 1e-6:
                valid = False
                break
        if not valid:
            continue
        
        W_new = np.zeros((p * n, target_rank), dtype=np.float64)
        for c in range(p * n):
            for r in range(target_rank):
                W_new[c, r] = coeffs[c * target_rank + r]
        
        U_new = U_keep
        V_new = V_keep
        
        if _verify_fast(U_new, V_new, W_new, n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None

def _try_random_subsets(U55, V55, W55, n, m, p, target_rank, deadline, seed=42):
    """Try random subsets of columns."""
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)
    
    rng = random.Random(seed)
    num_eqs = n * m * p
    num_unknowns = p * n * target_rank
    rank55 = 55
    
    max_attempts = 100
    attempts = 0
    
    while time.time() < deadline - 10 and attempts < max_attempts:
        attempts += 1
        
        if target_rank < rank55:
            cols = rng.sample(range(rank55), target_rank)
        else:
            cols = list(range(rank55))
        
        U_sub = U55[:, cols]
        V_sub = V55[:, cols]
        
        A = np.zeros((num_eqs, num_unknowns), dtype=np.float64)
        b = np.ones(num_eqs, dtype=np.float64)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    for r in range(target_rank):
                        c_idx = k * n + i
                        col = c_idx * target_rank + r
                        A[eq_idx, col] = U_sub[a_idx, r] * V_sub[b_idx, r]
                    eq_idx += 1
        
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        valid = True
        for cv in coeffs:
            if abs(cv * 2 - round(cv * 2)) > 1e-6:
                valid = False
                break
        if not valid:
            continue
        
        W_new = np.zeros((p * n, target_rank), dtype=np.float64)
        for c in range(p * n):
            for r in range(target_rank):
                W_new[c, r] = coeffs[c * target_rank + r]
        
        U_new = U_sub
        V_new = V_sub
        
        if _verify_fast(U_new, V_new, W_new, n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_verify_fast(U_best, V_best, W_best, n, m, p) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    U55 = np.array(U_best, dtype=np.float64)
    V55 = np.array(V_best, dtype=np.float64)
    W55 = np.array(W_best, dtype=np.float64)

    # Try dropping individual terms
    if time.time() < deadline - 30:
        result = _try_drop_single(U55, V55, W55, n, m, p, deadline)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 54
            return {"U": U_best, "V": V_best, "W": W_best}

    # Try random subsets for rank 54
    if time.time() < deadline - 15:
        result = _try_random_subsets(U55, V55, W55, n, m, p, 54, deadline)
        if result is not None:
            U_best, V_best, W_best = result
            best_rank = 54
            return {"U": U_best, "V": V_best, "W": W_best}

    return {"U": U_best, "V": V_best, "W": W_best}
