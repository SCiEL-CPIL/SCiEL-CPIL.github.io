"""Find rank-minimal tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 120.0
_SAFETY_MARGIN = 10.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _validate_numpy(U, V, W, n=3, m=3, p=5):
    """Check if U, V, W exactly decompose the matrix multiplication tensor using NumPy."""
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    for mat in [U, V, W]:
        if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
            return False
    return True

def _polish(U, V, W, n=3, m=3, p=5, max_iters=500):
    """High-precision polish: try to reduce error to zero with finer value set."""
    if _time_remaining() < 3.0:
        return None
    rng = random.Random(12345)
    rank = U.shape[1]
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    err = np.abs(T - 1.0)
    max_err = np.max(err)
    if max_err < 1e-9:
        if _validate_numpy(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    # Try small perturbations with half-integer values
    values = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
    for _ in range(max_iters):
        if _time_remaining() < 3.0:
            break
        mat_choice = rng.randint(0, 2)
        if mat_choice == 0:
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            r_idx = rng.randint(0, rank-1)
            old_val = U[i*m+j, r_idx]
            new_val = rng.choice(values)
            if new_val == old_val:
                continue
            delta = new_val - old_val
            v_vec = V_t[j, :, r_idx]
            w_vec = W_t[:, i, r_idx]
            delta_T = delta * (v_vec * w_vec)
            old_T_slice = T[i, j, :]
            new_T_slice = old_T_slice + delta_T
            old_sq = np.sum(np.abs(old_T_slice - 1.0)**2)
            new_sq = np.sum(np.abs(new_T_slice - 1.0)**2)
            if new_sq < old_sq:
                U[i*m+j, r_idx] = new_val
                U_t[i, j, r_idx] = new_val
                T[i, j, :] = new_T_slice
                if np.max(np.abs(T - 1.0)) < 1e-9:
                    if _validate_numpy(U, V, W, n, m, p):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        elif mat_choice == 1:
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            r_idx = rng.randint(0, rank-1)
            old_val = V[j*p+k, r_idx]
            new_val = rng.choice(values)
            if new_val == old_val:
                continue
            delta = new_val - old_val
            u_vec = U_t[:, j, r_idx]
            w_vec = W_t[k, :, r_idx]
            delta_T = delta * (u_vec * w_vec)
            old_T_slice = T[:, j, k]
            new_T_slice = old_T_slice + delta_T
            old_sq = np.sum(np.abs(old_T_slice - 1.0)**2)
            new_sq = np.sum(np.abs(new_T_slice - 1.0)**2)
            if new_sq < old_sq:
                V[j*p+k, r_idx] = new_val
                V_t[j, k, r_idx] = new_val
                T[:, j, k] = new_T_slice
                if np.max(np.abs(T - 1.0)) < 1e-9:
                    if _validate_numpy(U, V, W, n, m, p):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            k = rng.randint(0, p-1)
            i = rng.randint(0, n-1)
            r_idx = rng.randint(0, rank-1)
            old_val = W[k*n+i, r_idx]
            new_val = rng.choice(values)
            if new_val == old_val:
                continue
            delta = new_val - old_val
            u_vec = U_t[i, :, r_idx]
            v_vec = V_t[:, k, r_idx]
            delta_T = delta * (u_vec * v_vec)
            old_T_slice = T[i, :, k]
            new_T_slice = old_T_slice + delta_T
            old_sq = np.sum(np.abs(old_T_slice - 1.0)**2)
            new_sq = np.sum(np.abs(new_T_slice - 1.0)**2)
            if new_sq < old_sq:
                W[k*n+i, r_idx] = new_val
                W_t[k, i, r_idx] = new_val
                T[i, :, k] = new_T_slice
                if np.max(np.abs(T - 1.0)) < 1e-9:
                    if _validate_numpy(U, V, W, n, m, p):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _try_rank_numpy(r, n=3, m=3, p=5, max_iters=10000, seed=42):
    """Try to find a rank-r decomposition using NumPy-accelerated local search."""
    if _time_remaining() < 3.0:
        return None
    rng = random.Random(seed)
    values = [-2, -1, 0, 1, 2]
    n_restarts = 20
    for restart in range(n_restarts):
        if _time_remaining() < 5.0:
            break
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        for rr in range(r):
            nz_u = rng.randint(1, 3)
            nz_v = rng.randint(1, 3)
            nz_w = rng.randint(1, 3)
            for _ in range(nz_u):
                idx = rng.randint(0, n*m-1)
                U[idx, rr] = rng.choice(values)
            for _ in range(nz_v):
                idx = rng.randint(0, m*p-1)
                V[idx, rr] = rng.choice(values)
            for _ in range(nz_w):
                idx = rng.randint(0, p*n-1)
                W[idx, rr] = rng.choice(values)
        U_t = U.reshape(n, m, r)
        V_t = V.reshape(m, p, r)
        W_t = W.reshape(p, n, r)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        err = np.abs(T - 1.0)
        max_err = np.max(err)
        sq_err = np.sum(err**2)
        if max_err < 1e-9:
            if _validate_numpy(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        improved = True
        iters = 0
        temp = 1.0
        while improved and iters < max_iters and _time_remaining() > 5.0:
            improved = False
            iters += 1
            temp *= 0.99
            mat_choice = rng.randint(0, 2)
            if mat_choice == 0:
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                r_idx = rng.randint(0, r-1)
                old_val = U[i*m+j, r_idx]
                new_val = rng.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                v_vec = V_t[j, :, r_idx]
                w_vec = W_t[:, i, r_idx]
                delta_T = delta * (v_vec * w_vec)
                old_T_slice = T[i, j, :]
                new_T_slice = old_T_slice + delta_T
                old_sq_err_slice = np.sum(np.abs(old_T_slice - 1.0)**2)
                new_sq_err_slice = np.sum(np.abs(new_T_slice - 1.0)**2)
                delta_sq_err = new_sq_err_slice - old_sq_err_slice
                if delta_sq_err < 0 or (temp > 0 and rng.random() < np.exp(-delta_sq_err / temp)):
                    U[i*m+j, r_idx] = new_val
                    U_t[i, j, r_idx] = new_val
                    T[i, j, :] = new_T_slice
                    sq_err += delta_sq_err
                    improved = True
                    if np.max(np.abs(T - 1.0)) < 1e-9:
                        if _validate_numpy(U, V, W, n, m, p):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            elif mat_choice == 1:
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                r_idx = rng.randint(0, r-1)
                old_val = V[j*p+k, r_idx]
                new_val = rng.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                u_vec = U_t[:, j, r_idx]
                w_vec = W_t[k, :, r_idx]
                delta_T = delta * (u_vec * w_vec)
                old_T_slice = T[:, j, k]
                new_T_slice = old_T_slice + delta_T
                old_sq_err_slice = np.sum(np.abs(old_T_slice - 1.0)**2)
                new_sq_err_slice = np.sum(np.abs(new_T_slice - 1.0)**2)
                delta_sq_err = new_sq_err_slice - old_sq_err_slice
                if delta_sq_err < 0 or (temp > 0 and rng.random() < np.exp(-delta_sq_err / temp)):
                    V[j*p+k, r_idx] = new_val
                    V_t[j, k, r_idx] = new_val
                    T[:, j, k] = new_T_slice
                    sq_err += delta_sq_err
                    improved = True
                    if np.max(np.abs(T - 1.0)) < 1e-9:
                        if _validate_numpy(U, V, W, n, m, p):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                k = rng.randint(0, p-1)
                i = rng.randint(0, n-1)
                r_idx = rng.randint(0, r-1)
                old_val = W[k*n+i, r_idx]
                new_val = rng.choice(values)
                if new_val == old_val:
                    continue
                delta = new_val - old_val
                u_vec = U_t[i, :, r_idx]
                v_vec = V_t[:, k, r_idx]
                delta_T = delta * (u_vec * v_vec)
                old_T_slice = T[i, :, k]
                new_T_slice = old_T_slice + delta_T
                old_sq_err_slice = np.sum(np.abs(old_T_slice - 1.0)**2)
                new_sq_err_slice = np.sum(np.abs(new_T_slice - 1.0)**2)
                delta_sq_err = new_sq_err_slice - old_sq_err_slice
                if delta_sq_err < 0 or (temp > 0 and rng.random() < np.exp(-delta_sq_err / temp)):
                    W[k*n+i, r_idx] = new_val
                    W_t[k, i, r_idx] = new_val
                    T[i, :, k] = new_T_slice
                    sq_err += delta_sq_err
                    improved = True
                    if np.max(np.abs(T - 1.0)) < 1e-9:
                        if _validate_numpy(U, V, W, n, m, p):
                            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if not improved:
                for _ in range(10):
                    mat_choice = rng.randint(0, 2)
                    if mat_choice == 0:
                        idx = rng.randint(0, n*m-1)
                        r_idx = rng.randint(0, r-1)
                        U[idx, r_idx] = rng.choice(values)
                    elif mat_choice == 1:
                        idx = rng.randint(0, m*p-1)
                        r_idx = rng.randint(0, r-1)
                        V[idx, r_idx] = rng.choice(values)
                    else:
                        idx = rng.randint(0, p*n-1)
                        r_idx = rng.randint(0, r-1)
                        W[idx, r_idx] = rng.choice(values)
                U_t = U.reshape(n, m, r)
                V_t = V.reshape(m, p, r)
                W_t = W.reshape(p, n, r)
                T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                err = np.abs(T - 1.0)
                sq_err = np.sum(err**2)
                temp = 1.0
    return None

def _try_structured(r, n=3, m=3, p=5, seed=42):
    """Try structured initialization based on Strassen-like patterns."""
    if _time_remaining() < 3.0:
        return None
    rng = random.Random(seed)
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    # Use Strassen-inspired patterns: group indices
    # For 3x3, split into 1x1 blocks (trivial) or use algebraic identities
    # Try: for each rank term, assign U[i,j,r] based on (i,j) pattern
    patterns_u = [
        lambda i,j: 1 if i==j else 0,
        lambda i,j: 1 if i<j else 0,
        lambda i,j: 1 if i>j else 0,
        lambda i,j: (1 if i==0 else 0) + (1 if j==0 else 0),
        lambda i,j: 1,
    ]
    patterns_v = [
        lambda j,k: 1 if j==k else 0,
        lambda j,k: 1 if j<k else 0,
        lambda j,k: 1 if j>k else 0,
        lambda j,k: (1 if j==0 else 0) + (1 if k==0 else 0),
        lambda j,k: 1,
    ]
    patterns_w = [
        lambda k,i: 1 if k==i else 0,
        lambda k,i: 1 if k<i else 0,
        lambda k,i: 1 if k>i else 0,
        lambda k,i: (1 if k==0 else 0) + (1 if i==0 else 0),
        lambda k,i: 1,
    ]
    for rr in range(r):
        pu = rng.choice(patterns_u)
        pv = rng.choice(patterns_v)
        pw = rng.choice(patterns_w)
        su = rng.choice([-1, 1])
        sv = rng.choice([-1, 1])
        sw = rng.choice([-1, 1])
        for i in range(n):
            for j in range(m):
                U[i*m+j, rr] = su * pu(i, j)
        for j in range(m):
            for k in range(p):
                V[j*p+k, rr] = sv * pv(j, k)
        for k in range(p):
            for i in range(n):
                W[k*n+i, rr] = sw * pw(k, i)
    # Validate
    U_t = U.reshape(n, m, r)
    V_t = V.reshape(m, p, r)
    W_t = W.reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    if np.allclose(T, 1.0, atol=1e-9):
        if _validate_numpy(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    # Polish
    return _polish(U, V, W, n, m, p)

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
    global _START_TIME
    _START_TIME = time.time()
    n, m, p = 3, 3, 5
    # Build schoolbook as fallback (rank 45)
    U = [[0]*45 for _ in range(n*m)]
    V = [[0]*45 for _ in range(m*p)]
    W = [[0]*45 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    schoolbook = {"U": U, "V": V, "W": W}
    best_rank = 45
    best_sol = schoolbook
    # Try structured first for lower ranks
    for r in range(9, 20):
        if _time_remaining() < 5.0:
            break
        sol = _try_structured(r, n, m, p, seed=r)
        if sol is not None:
            best_rank = r
            best_sol = sol
            break
    # Then random search
    for r in range(best_rank, 46):
        if _time_remaining() < 5.0:
            break
        found = False
        for seed in [r, 100+r, 200+r, 300+r]:
            if _time_remaining() < 5.0:
                break
            sol = _try_rank_numpy(r, n, m, p, max_iters=5000, seed=seed)
            if sol is not None:
                best_rank = r
                best_sol = sol
                found = True
                break
        if found:
            break
    return best_sol
