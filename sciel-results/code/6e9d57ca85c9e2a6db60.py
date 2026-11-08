"""Matrix multiplication 3x3x4 — search for rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product, combinations

_DIMS = (3, 3, 4)
_START = time.time()
_DEADLINE = _START + 140  # 10s safety margin before 150s kill


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


def _try_solve_fixed_W(W, target, n=3, m=3, p=4):
    """Given a fixed W matrix (pn x r), try to solve for U and V."""
    nm, mp, pn = n * m, m * p, p * n
    r = W.shape[1]
    W = np.array(W, dtype=float)
    
    # For each a, we need: sum_c W[c, :] * (something) to reconstruct target row
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # For fixed W: T[a,b,:] = U[a,:] @ (V * W)  where (V*W)[b,c,r] = V[b,r]*W[c,r]
    # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Let M[a] be the mp x pn matrix with M[a][b,c] = T[a,b,c]
    # Then M[a] = sum_r U[a,r] * (V[:,r] outer W[:,r])
    # So M[a] = U[a,:] @ (V ⊙ W) where (V⊙W)[b,c,r] = V[b,r]*W[c,r]
    # Define G[b,c,r] = V[b,r]*W[c,r], shape (mp, pn, r)
    # Then M[a] = sum_r U[a,r] * G[:,:,r], i.e., M[a].reshape(mp*pn) = U[a,:] @ G.reshape(r, mp*pn).T
    
    # Build G: G[r, b, c] = V[b,r] * W[c,r]
    # But V is unknown! So this is coupled.
    
    # Alternative: treat it as a bilinear system.
    # For fixed W, the equations are:
    # For each (a,b,c): T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # If W[c,r] = 0 for all r, then we need T[a,b,c] = 0.
    # If W has some zero columns, those r don't contribute.
    
    # Let's identify which r are "active" (W[:,r] is not all zero)
    active = [r for r in range(r) if np.any(W[:, r] != 0)]
    if len(active) == 0:
        return None
    
    # For a given active set, we have a bilinear system in U and V.
    # This is hard to solve directly. Let's try a different approach:
    # Assume W is sparse with a specific pattern, and solve iteratively.
    
    # Actually, let's use a simpler strategy: 
    # Fix W to be a selection matrix (one 1 per column in specific rows),
    # then the problem becomes linear in U and V.
    
    # For now, return None and use a different search strategy.
    return None


def _search_rank_small(n, m, p, target_rank, time_budget, rng):
    """Search for rank-`target_rank` decomposition using structured random W patterns."""
    nm, mp, pn = n * m, m * p, p * n
    target = _build_target(n, m, p)
    
    # Strategy: try W matrices where each column has exactly one 1 (selection pattern)
    # W[c, r] = 1 if c == sel[r] for some permutation/selection sel
    # Then T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] = sum_{r: sel[r]==c} U[a,r] V[b,r]
    # So for each c, T[a,b,c] = sum_{r in group_c} U[a,r] V[b,r]
    # where group_c = {r : sel[r] == c}
    
    # This means T[:,:,c] is a sum of |group_c| rank-1 matrices.
    # The total rank is sum_c |group_c| = r.
    
    # For the matmul tensor, T[:,:,c] for c = k*n+i has a specific structure.
    # T[a,b,c] = 1 iff a%m == b//p and b%p == c//n and a//m == c%n
    # For fixed c = k*n+i: T[a,b,c] = 1 iff a = i*m+j, b = j*p+k for some j in range(m)
    # So T[:,:,c] has exactly m non-zero entries: at positions (i*m+j, j*p+k) for j=0..m-1
    
    # Each T[:,:,c] needs to be represented as sum of rank-1 matrices.
    # The minimum rank to represent T[:,:,c] is m (since the m non-zeros are in distinct rows and cols).
    # So we need at least m columns of W for each c, giving total rank >= p*n*m = 36? No...
    # Wait, we can share columns across different c values if the patterns align.
    
    # Actually, the schoolbook uses exactly m columns for each c, total p*n*m = 36.
    # To get lower rank, we need to use columns that contribute to multiple c values.
    # But W[c,r] is either 0 or 1 (in our pattern), so each column r contributes to exactly one c.
    # So this pattern can't beat schoolbook.
    
    # Better: allow W[c,r] to be 0, 1, or -1, and have each column contribute to multiple c.
    # This is getting complex. Let's try a direct linear algebra approach.
    
    # New strategy: Use the fact that the matmul tensor has symmetries.
    # For 3x3x4, try to find a rank-20 decomposition by solving a system of equations.
    
    # Let's try: fix U and V to have specific sparsity patterns, solve for W, or vice versa.
    
    # Actually, let's try a randomized approach with local optimization.
    # Start with random U, V, W of rank r, and do coordinate descent.
    
    best_result = None
    start = time.time()
    
    for attempt in range(50):
        if time.time() - start > time_budget - 10:
            break
        r = target_rank
        U = rng.normal(size=(nm, r))
        V = rng.normal(size=(mp, r))
        W = rng.normal(size=(pn, r))
        
        # Coordinate descent: update one entry at a time to reduce error
        for it in range(1000):
            if time.time() - start > time_budget - 10:
                break
            
            # Compute current error
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.sum((T - target) ** 2)
            if err < 1e-12:
                # Found exact solution!
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Pick a random entry to update
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                a = rng.randint(0, nm - 1)
                c = rng.randint(0, r - 1)
                # Optimize U[a,c] to minimize error
                # T[a,b,d] = sum_r U[a,r] V[b,r] W[d,r]
                # = U[a,c] V[b,c] W[d,c] + sum_{r!=c} U[a,r] V[b,r] W[d,r]
                # Let R[a,b,d] = sum_{r!=c} U[a,r] V[b,r] W[d,r]
                # Then T[a,b,d] = R[a,b,d] + U[a,c] * V[b,c] * W[d,c]
                # Minimize sum_{b,d} (target[a,b,d] - R[a,b,d] - U[a,c]*V[b,c]*W[d,c])^2
                # This is linear in U[a,c]: 
                # Let v = V[:,c], w = W[:,c], t = target[a,:,:].reshape(-1), r_flat = R[a,:,:].reshape(-1)
                # minimize ||t - r_flat - U[a,c]*(v*w)||^2
                # d/dU = -2 * (v*w)^T (t - r_flat - U[a,c]*(v*w)) = 0
                # U[a,c] = (v*w)^T (t - r_flat) / (v*w)^T (v*w)
                v = V[:, c]
                w = W[:, c]
                vw = v * w  # shape (mp, pn) no, v is (mp,), w is (pn,), so vw needs broadcasting
                # v[w] broadcast: v[:,None] * w[None,:]
                vw_mat = np.outer(v, w)  # (mp, pn)
                t_flat = target[a, :, :].reshape(-1)
                # Compute R[a,:,:] = T[a,:,:] - U[a,c]*vw_mat
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
                uw_mat = np.outer(u, w)  # (nm, pn)
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
                uv_mat = np.outer(u, v)  # (nm, mp)
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
    
    return best_result


def construct():
    n, m, p = _DIMS
    nm, mp, pn = n * m, m * p, p * n
    
    # Always have schoolbook as fallback
    result = _schoolbook(n, m, p)
    
    # Try to find lower rank decompositions
    # Known: rank for 3x3x4 is at most 20 (Lader 1977)
    # Try ranks from 20 up to 35
    
    rng = np.random.RandomState(42)
    
    for target_rank in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if _time_left() < 15:
            break
        time_budget = min(8, _time_left() - 10)
        if time_budget < 2:
            break
        
        res = _search_rank_small(n, m, p, target_rank, time_budget, rng)
        if res is not None:
            if _validate(res["U"], res["V"], res["W"], n, m, p):
                result = res
                break  # Found the lowest rank we tried
    
    # Final validation
    if not _validate(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
        result = _schoolbook(n, m, p)
        if not _validate(result["U"], result["V"], result["W"], n, m, p):
            raise RuntimeError("Internal error: schoolbook decomposition failed validation")
    
    return result
