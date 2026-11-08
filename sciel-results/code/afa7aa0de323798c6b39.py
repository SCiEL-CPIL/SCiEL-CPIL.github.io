"""Matrix multiplication 3x3x4 — search for rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product, combinations

_DIMS = (3, 3, 4)
_START = time.time()
_DEADLINE = _START + 130  # 20s safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _time_left():
    return _DEADLINE - time.time()


def _validate(U, V, W, n=3, m=3, p=4):
    """Check exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    if not np.allclose(T, target, atol=1e-9):
        return False
    return True


def _schoolbook(n=3, m=3, p=4):
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
    return {"U": U, "V": V, "W": W}


def _build_target(n=3, m=3, p=4):
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    return target


def _search_rank_small(n, m, p, target_rank, time_budget, rng):
    """Search for rank-`target_rank` decomposition using randomized coordinate descent."""
    nm, mp, pn = n * m, m * p, p * n
    target = _build_target(n, m, p)
    
    start = time.time()
    
    for attempt in range(100):
        if time.time() - start > time_budget - 5:
            break
        r = target_rank
        U = rng.normal(size=(nm, r))
        V = rng.normal(size=(mp, r))
        W = rng.normal(size=(pn, r))
        
        # Coordinate descent with adaptive step size
        best_err = np.inf
        stuck_count = 0
        
        for it in range(2000):
            if time.time() - start > time_budget - 5:
                break
            
            # Compute current error
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T - target) ** 2)
            
            if err < 1e-12:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            if err < best_err * (1 - 1e-6):
                best_err = err
                stuck_count = 0
            else:
                stuck_count += 1
            
            if stuck_count > 200:
                break
            
            # Pick a random entry to update
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                a = rng.randint(0, nm - 1)
                c = rng.randint(0, r - 1)
                v = V[:, c]
                w = W[:, c]
                vw_mat = np.outer(v, w)
                t_flat = target[a, :, :].reshape(-1)
                T_a = np.einsum('r,br,dr->bd', U[a], V, W)
                R_a = T_a - U[a, c] * vw_mat
                r_flat = R_a.reshape(-1)
                vw_flat = vw_mat.reshape(-1)
                denom = np.dot(vw_flat, vw_flat)
                if abs(denom) < 1e-15:
                    continue
                U[a, c] = np.dot(vw_flat, t_flat - r_flat) / denom
            elif mat == 'V':
                b = rng.randint(0, mp - 1)
                c = rng.randint(0, r - 1)
                u = U[:, c]
                w = W[:, c]
                uw_mat = np.outer(u, w)
                t_flat = target[:, b, :].reshape(-1)
                T_b = np.einsum('ar,r,dr->ad', U, V[b], W)
                R_b = T_b - V[b, c] * uw_mat
                r_flat = R_b.reshape(-1)
                uw_flat = uw_mat.reshape(-1)
                denom = np.dot(uw_flat, uw_flat)
                if abs(denom) < 1e-15:
                    continue
                V[b, c] = np.dot(uw_flat, t_flat - r_flat) / denom
            else:  # W
                d = rng.randint(0, pn - 1)
                c = rng.randint(0, r - 1)
                u = U[:, c]
                v = V[:, c]
                uv_mat = np.outer(u, v)
                t_flat = target[:, :, d].reshape(-1)
                T_d = np.einsum('ar,br,r->ab', U, V, W[d])
                R_d = T_d - W[d, c] * uv_mat
                r_flat = R_d.reshape(-1)
                uv_flat = uv_mat.reshape(-1)
                denom = np.dot(uv_flat, uv_flat)
                if abs(denom) < 1e-15:
                    continue
                W[d, c] = np.dot(uv_flat, t_flat - r_flat) / denom
    
        # Check if we converged
        T = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T, target, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def _search_rank_gf3(n, m, p, target_rank, time_budget, rng):
    """Search over GF(3) for a rank-`target_rank` decomposition, then lift to reals."""
    nm, mp, pn = n * m, m * p, p * n
    target = _build_target(n, m, p)
    target_gf3 = (target.astype(int) % 3)
    
    start = time.time()
    
    for attempt in range(200):
        if time.time() - start > time_budget - 5:
            break
        r = target_rank
        # Random initialization over GF(3): values in {0, 1, 2}
        U = rng.randint(0, 3, size=(nm, r))
        V = rng.randint(0, 3, size=(mp, r))
        W = rng.randint(0, 3, size=(pn, r))
        
        # Coordinate descent over GF(3)
        for it in range(3000):
            if time.time() - start > time_budget - 5:
                break
            
            # Compute T over GF(3)
            T = np.zeros((nm, mp, pn), dtype=int)
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        s = 0
                        for rr in range(r):
                            s += U[a, rr] * V[b, rr] * W[c, rr]
                        T[a, b, c] = s % 3
            
            # Check if exact
            if np.array_equal(T, target_gf3):
                # Found exact solution over GF(3)!
                # Lift to reals: map 0->0, 1->1, 2->-1 (or 2)
                U_real = np.where(U == 2, -1, U).astype(float)
                V_real = np.where(V == 2, -1, V).astype(float)
                W_real = np.where(W == 2, -1, W).astype(float)
                
                # Validate over reals
                if _validate(U_real, V_real, W_real, n, m, p):
                    return {"U": U_real.tolist(), "V": V_real.tolist(), "W": W_real.tolist()}
                # If not, try different lifting
                U_real = U.astype(float)
                V_real = V.astype(float)
                W_real = W.astype(float)
                if _validate(U_real, V_real, W_real, n, m, p):
                    return {"U": U_real.tolist(), "V": V_real.tolist(), "W": W_real.tolist()}
                continue
            
            # Pick a random entry to update
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                a = rng.randint(0, nm - 1)
                c = rng.randint(0, r - 1)
                # Try all 3 values, pick best
                best_val = U[a, c]
                best_err = np.inf
                for val in [0, 1, 2]:
                    U[a, c] = val
                    T = np.zeros((nm, mp, pn), dtype=int)
                    for aa in range(nm):
                        for bb in range(mp):
                            for cc in range(pn):
                                s = 0
                                for rr in range(r):
                                    s += U[aa, rr] * V[bb, rr] * W[cc, rr]
                                T[aa, bb, cc] = s % 3
                    err = np.sum((T - target_gf3) ** 2)
                    if err < best_err:
                        best_err = err
                        best_val = val
                U[a, c] = best_val
            elif mat == 'V':
                b = rng.randint(0, mp - 1)
                c = rng.randint(0, r - 1)
                best_val = V[b, c]
                best_err = np.inf
                for val in [0, 1, 2]:
                    V[b, c] = val
                    T = np.zeros((nm, mp, pn), dtype=int)
                    for aa in range(nm):
                        for bb in range(mp):
                            for cc in range(pn):
                                s = 0
                                for rr in range(r):
                                    s += U[aa, rr] * V[bb, rr] * W[cc, rr]
                                T[aa, bb, cc] = s % 3
                    err = np.sum((T - target_gf3) ** 2)
                    if err < best_err:
                        best_err = err
                        best_val = val
                V[b, c] = best_val
            else:  # W
                d = rng.randint(0, pn - 1)
                c = rng.randint(0, r - 1)
                best_val = W[d, c]
                best_err = np.inf
                for val in [0, 1, 2]:
                    W[d, c] = val
                    T = np.zeros((nm, mp, pn), dtype=int)
                    for aa in range(nm):
                        for bb in range(mp):
                            for cc in range(pn):
                                s = 0
                                for rr in range(r):
                                    s += U[aa, rr] * V[bb, rr] * W[cc, rr]
                                T[aa, bb, cc] = s % 3
                    err = np.sum((T - target_gf3) ** 2)
                    if err < best_err:
                        best_err = err
                        best_val = val
                W[d, c] = best_val
    
    return None


def construct():
    n, m, p = _DIMS
    nm, mp, pn = n * m, m * p, p * n
    
    # Always have schoolbook as fallback
    result = _schoolbook(n, m, p)
    
    rng = np.random.RandomState(42)
    
    # Try GF(3) search first for lower ranks
    for target_rank in [20, 21, 22, 23, 24, 25]:
        if _time_left() < 10:
            break
        time_budget = min(15, _time_left() - 10)
        if time_budget < 3:
            break
        
        res = _search_rank_gf3(n, m, p, target_rank, time_budget, rng)
        if res is not None:
            if _validate(res["U"], res["V"], res["W"], n, m, p):
                result = res
                break
    
    # Try real-valued coordinate descent if time remains
    if _time_left() > 15:
        for target_rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
            if _time_left() < 10:
                break
            time_budget = min(8, _time_left() - 10)
            if time_budget < 2:
                break
            
            res = _search_rank_small(n, m, p, target_rank, time_budget, rng)
            if res is not None:
                if _validate(res["U"], res["V"], res["W"], n, m, p):
                    result = res
                    break
    
    # Final validation
    if not _validate(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)
        if not _validate(result["U"], result["V"], result["W"], n, m, p):
            raise RuntimeError("Internal error: schoolbook decomposition failed validation")
    
    return result
