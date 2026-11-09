"""Matrix multiplication 3x3x4 — search for rank-minimal exact tensor decomposition."""
import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140  # 10s safety margin before 150s kill


def _time_left():
    return _DEADLINE - time.time()


def _build_target(n=3, m=3, p=4):
    """Build the target tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i."""
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
    return target


def _validate(U, V, W, n=3, m=3, p=4):
    """Check exact decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    target = _build_target(n, m, p)
    return np.allclose(T, target, atol=1e-9)


def _round_to_half_integer(x):
    """Round to nearest half-integer."""
    return np.round(2 * x) / 2


def _schoolbook(n=3, m=3, p=4):
    """Schoolbook decomposition with rank n*m*p."""
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


def _try_rank(n, m, p, target_rank, time_budget, rng):
    """Try to find a rank-`target_rank` decomposition using coordinate descent."""
    nm, mp, pn = n * m, m * p, p * n
    target = _build_target(n, m, p)
    r = target_rank
    
    for attempt in range(8):
        if _time_left() < time_budget + 5:
            break
        
        # Initialize with small random values
        U = rng.normal(size=(nm, r)) * 0.1
        V = rng.normal(size=(mp, r)) * 0.1
        W = rng.normal(size=(pn, r)) * 0.1
        
        for it in range(2000):
            if _time_left() < time_budget + 5:
                break
            
            # Compute current error
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T - target) ** 2)
            if err < 1e-12:
                # Try to round to half-integers
                U_rounded = _round_to_half_integer(U)
                V_rounded = _round_to_half_integer(V)
                W_rounded = _round_to_half_integer(W)
                if _validate(U_rounded, V_rounded, W_rounded, n, m, p):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                # Also try the unrounded version
                if _validate(U, V, W, n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
            
            # Pick a random matrix and entry to update
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
        
        # Final check after coordinate descent
        T = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T, target, atol=1e-9):
            U_rounded = _round_to_half_integer(U)
            V_rounded = _round_to_half_integer(V)
            W_rounded = _round_to_half_integer(W)
            if _validate(U_rounded, V_rounded, W_rounded, n, m, p):
                return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
            if _validate(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    n, m, p = 3, 3, 4
    
    # Start with schoolbook as guaranteed fallback
    result = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    rng = np.random.RandomState(42)
    
    for target_rank in range(35, 19, -1):  # Try 35 down to 20
        if _time_left() < 15:
            break
        time_budget = min(4, _time_left() - 10)
        if time_budget < 2:
            break
        
        res = _try_rank(n, m, p, target_rank, time_budget, rng)
        if res is not None:
            if _validate(res["U"], res["V"], res["W"], n, m, p):
                result = res
                break  # Found the lowest rank we tried
    
    # Final validation
    if not _validate(result["U"], result["V"], result["W"], n, m, p):
        result = _schoolbook(n, m, p)
        if not _validate(result["U"], result["V"], result["W"], n, m, p):
            raise RuntimeError("Internal error: schoolbook decomposition failed validation")
    
    return result
