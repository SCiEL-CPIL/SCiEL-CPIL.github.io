import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 140.0  # 10s margin before 150s kill

def _elapsed():
    return time.time() - _T0

def _remaining():
    return _DEADLINE - _elapsed()

def _build_tensors(n, m, p):
    idx = np.indices((n, m, m, p, n, p))
    ia = idx[0].flatten()
    ja = idx[1].flatten()
    jb = idx[2].flatten()
    kb = idx[3].flatten()
    i = idx[4].flatten()
    k = idx[5].flatten()
    
    U_idx = ia * m + ja
    V_idx = jb * p + kb
    W_idx = k * n + i
    
    expected = ((ia == i) & (ja == jb) & (kb == k)).astype(float)
    
    return U_idx, V_idx, W_idx, expected

def _bilinear_error(U, V, W, n, m, p, U_idx, V_idx, W_idx, expected):
    W_sel = W[W_idx, :]
    U_sel = U[U_idx, :]
    V_sel = V[V_idx, :]
    actual = np.einsum('nr,nr,nr->n', W_sel, U_sel, V_sel)
    err = actual - expected
    return float(np.sum(err * err))

def _verify_full(U, V, W, n, m, p, tol=1e-8):
    U_idx, V_idx, W_idx, expected = _build_tensors(n, m, p)
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    
    W_sel = W[W_idx, :]
    U_sel = U[U_idx, :]
    V_sel = V[V_idx, :]
    actual = np.einsum('nr,nr,nr->n', W_sel, U_sel, V_sel)
    return bool(np.all(np.abs(actual - expected) < tol))

def _rank22_list(n, m, p):
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    # Block 1: 7 Strassen terms for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[3][col] = 1
    col += 1

    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[3][col] = -1
    col += 1

    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[2][col] = 1; W[3][col] = 1
    col += 1

    U[4][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1

    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[2][col] = 1
    col += 1

    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[3][col] = 1
    col += 1

    U[1][col] = 1; U[4][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1
    col += 1

    # Block 2: 7 Strassen terms for A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[4][col] = 1; W[7][col] = 1
    col += 1

    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[5][col] = 1; W[7][col] = -1
    col += 1

    U[0][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1

    U[4][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1

    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[4][col] = -1; W[6][col] = 1
    col += 1

    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1

    U[1][col] = 1; U[4][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[4][col] = 1
    col += 1

    # Block 3: 8 schoolbook terms for A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    for i, ai in [(0, 2), (1, 5)]:
        for k, bk in [(0, 8), (1, 9), (2, 10), (3, 11)]:
            U[ai][col] = 1
            V[bk][col] = 1
            W[k * n + i][col] = 1
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def _schoolbook(n, m, p):
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
    return U, V, W

def _to_dict(U, V, W):
    return {"U": U, "V": V, "W": W}

def _snap_to_half_int(arr):
    return np.round(arr * 2) / 2

def _try_rank_optimize(r, n, m, p, U_idx, V_idx, W_idx, expected, max_restarts=50):
    """Try to find rank-r decomposition via L-BFGS-B local search."""
    nm, mp, pn = n * m, m * p, p * n
    best_err = float('inf')
    best_x = None
    
    bounds = [(-10, 10)] * (nm * r + mp * r + pn * r)
    
    def objective(x):
        U = x[:nm * r].reshape(nm, r)
        V = x[nm * r:nm * r + mp * r].reshape(mp, r)
        W = x[nm * r + mp * r:].reshape(pn, r)
        return _bilinear_error(U, V, W, n, m, p, U_idx, V_idx, W_idx, expected)

    np.random.seed(42)
    for i in range(max_restarts):
        if _remaining() < 10.0:
            break
        
        x0 = np.random.uniform(-2, 2, size=nm * r + mp * r + pn * r)
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x.copy()
                
                if best_err < 1e-10:
                    break
                    
        except Exception:
            pass

    if best_x is not None and best_err < 1e-8:
        U = best_x[:nm * r].reshape(nm, r)
        V = best_x[nm * r:nm * r + mp * r].reshape(mp, r)
        W = best_x[nm * r + mp * r:].reshape(pn, r)
        
        # Try snapping to half-integers
        U_snap = _snap_to_half_int(U)
        V_snap = _snap_to_half_int(V)
        W_snap = _snap_to_half_int(W)
        
        if _verify_full(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
            return U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), r
        elif _verify_full(U, V, W, n, m, p, tol=1e-8):
            return U.tolist(), V.tolist(), W.tolist(), r
        
    return None

def _structured_random_walk(start_U, start_V, start_W, n, m, p, U_idx, V_idx, W_idx, expected, 
                            target_rank, max_steps=5000, step_size=0.5):
    """
    Structured random walk over half-integer decompositions.
    Start from a valid rank-22 decomposition, try to reduce rank by zeroing columns.
    """
    nm, mp, pn = n * m, m * p, p * n
    current_U = np.array(start_U, dtype=float)
    current_V = np.array(start_V, dtype=float)
    current_W = np.array(start_W, dtype=float)
    current_rank = current_U.shape[1]
    
    # Track which columns are "active" (non-zero)
    active_cols = set(range(current_rank))
    
    rng = np.random.RandomState(123)
    
    for step in range(max_steps):
        if _remaining() < 15.0:
            break
        
        # Try to eliminate a random active column by adjusting others
        if len(active_cols) <= target_rank:
            # Check if we have a valid decomposition at target rank
            active_list = sorted(active_cols)
            if len(active_list) == target_rank:
                U_sub = current_U[:, active_list]
                V_sub = current_V[:, active_list]
                W_sub = current_W[:, active_list]
                
                # Snap to half-integers
                U_snap = _snap_to_half_int(U_sub)
                V_snap = _snap_to_half_int(V_sub)
                W_snap = _snap_to_half_int(W_sub)
                
                if _verify_full(U_snap, V_snap, W_snap, n, m, p, tol=1e-6):
                    return U_snap.tolist(), V_snap.tolist(), W_snap.tolist(), target_rank
                elif _verify_full(U_sub, V_sub, W_sub, n, m, p, tol=1e-8):
                    return U_sub.tolist(), V_sub.tolist(), W_sub.tolist(), target_rank
            
            continue
        
        # Pick a random active column to try to eliminate
        col_to_eliminate = rng.choice(list(active_cols))
        
        # Try small perturbations to other columns to compensate
        success = False
        for _ in range(100):
            if _remaining() < 15.0:
                break
            
            # Pick a random other active column to adjust
            other_cols = [c for c in active_cols if c != col_to_eliminate]
            if not other_cols:
                break
            col_to_adjust = rng.choice(other_cols)
            
            # Save state
            U_save = current_U.copy()
            V_save = current_V.copy()
            W_save = current_W.copy()
            
            # Apply small perturbation to the adjustment column
            delta = rng.uniform(-step_size, step_size, size=3)
            current_U[:, col_to_adjust] += delta[0]
            current_V[:, col_to_adjust] += delta[1]
            current_W[:, col_to_adjust] += delta[2]
            
            # Zero out the column we want to eliminate
            current_U[:, col_to_eliminate] = 0
            current_V[:, col_to_eliminate] = 0
            current_W[:, col_to_eliminate] = 0
            
            # Check if still valid
            err = _bilinear_error(current_U, current_V, current_W, n, m, p, U_idx, V_idx, W_idx, expected)
            
            if err < 1e-6:
                success = True
                active_cols.discard(col_to_eliminate)
                break
            else:
                # Revert
                current_U = U_save
                current_V = V_save
                current_W = W_save
        
        if not success:
            # Revert if failed
            pass
    
    return None

def construct():
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n

    U_idx, V_idx, W_idx, expected = _build_tensors(n, m, p)

    best_U, best_V, best_W = _rank22_list(n, m, p)
    best_rank = 22

    if not _verify_full(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = _schoolbook(n, m, p)
        best_rank = n * m * p

    if _remaining() < 20.0:
        return _to_dict(best_U, best_V, best_W)

    # Try rank 20 via optimization
    if _remaining() > 50.0:
        res20 = _try_rank_optimize(20, n, m, p, U_idx, V_idx, W_idx, expected, max_restarts=80)
        if res20 is not None:
            U20, V20, W20, r20 = res20
            return _to_dict(U20, V20, W20)

    # Try rank 21 via optimization
    if _remaining() > 30.0:
        res21 = _try_rank_optimize(21, n, m, p, U_idx, V_idx, W_idx, expected, max_restarts=80)
        if res21 is not None:
            U21, V21, W21, r21 = res21
            return _to_dict(U21, V21, W21)

    # Try structured random walk to reduce rank from 22
    if _remaining() > 20.0:
        res_walk = _structured_random_walk(best_U, best_V, best_W, n, m, p, 
                                           U_idx, V_idx, W_idx, expected,
                                           target_rank=21, max_steps=3000)
        if res_walk is not None:
            Uw, Vw, Ww, rw = res_walk
            return _to_dict(Uw, Vw, Ww)

    return _to_dict(best_U, best_V, best_W)
