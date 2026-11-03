import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """7 terms for 2x2x2 Strassen."""
    return [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

def _build_kronecker_strassen(n=4, m=4, p=5):
    """Kronecker-embedded Strassen baseline (rank 65)."""
    U_cols, V_cols, W_cols = [], [], []

    def add(u, v, w):
        U_cols.append(u); V_cols.append(v); W_cols.append(w)

    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i

    st = _strassen_2x2x2_terms()
    for (u1s, v1s, w1s) in st:
        for (u2s, v2s, w2s) in st:
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            for (a, va) in u1s:
                i1, j1 = a // 2, a % 2
                for (b, vb) in u2s:
                    i2, j2 = b // 2, b % 2
                    u[ui(i1 * 2 + i2, j1 * 2 + j2)] = va * vb
            for (a, va) in v1s:
                j1, k1 = a // 2, a % 2
                for (b, vb) in v2s:
                    j2, k2 = b // 2, b % 2
                    v[vi(j1 * 2 + j2, k1 * 2 + k2)] = va * vb
            for (a, va) in w1s:
                k1, i1 = a // 2, a % 2
                for (b, vb) in w2s:
                    k2, i2 = b // 2, b % 2
                    w[wi(k1 * 2 + k2, i1 * 2 + i2)] = va * vb
            add(u, v, w)

    # Rank-1 decomposition for the k=4 slice (4x4x1 all-ones tensor)
    u = np.ones(n * m)
    v = np.zeros(m * p); v[vi(0, 4):] = 1 # Wait, V index is j*p+k.
    # We need V[j, 4] = 1 for all j.
    for j in range(m):
        v[vi(j, 4)] = 1
    w = np.zeros(p * n)
    for i in range(n):
        w[wi(4, i)] = 1
    add(u, v, w)

    return np.array(U_cols).T, np.array(V_cols).T, np.array(W_cols).T

def _verify(U, V, W, n, m, p, tol=1e-7):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            row_u = U[i * m + j]
            for k in range(p):
                s = np.dot(row_u * V[j * p + k], W[k * n + i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_int(M, tol=1e-8):
    M = np.asarray(M, dtype=np.float64)
    return bool(np.all(np.abs(M * 2 - np.round(M * 2)) <= tol))

def _round_half(M):
    return np.round(np.asarray(M, dtype=np.float64) * 2) / 2.0

def _build_schoolbook(n=4, m=4, p=5):
    rank = n * m * p
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _to_list(M):
    return [[float(x) for x in row] for row in np.asarray(M)]

def _null_space_prune(U, V, W, n, m, p, max_iter=1000, tol=1e-8):
    """
    Uses null space of the tensor flattening to find linear dependencies
    and prune terms.
    """
    U = np.array(U, dtype=np.float64).copy()
    V = np.array(V, dtype=np.float64).copy()
    W = np.array(W, dtype=np.float64).copy()
    rank = U.shape[1]
    
    # Flatten tensor vectors
    # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Vector length n*m*p
    vecs = []
    for r in range(rank):
        vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                u_val = U[i*m+j, r]
                if u_val == 0: continue
                for k in range(p):
                    v_val = V[j*p+k, r]
                    if v_val == 0: continue
                    w_val = W[k*n+i, r]
                    if w_val == 0: continue
                    vec[i*m*p + j*p + k] = u_val * v_val * w_val
        vecs.append(vec)
        
    A = np.array(vecs).T # Shape (n*m*p, rank)
    t = np.sum(A, axis=1) # The target tensor vector
    
    # Solve A alpha = t
    # We know alpha = 1 is a solution.
    # Find null space N of A.
    N = null_space(A)
    
    if N.shape[1] == 0:
        return U, V, W
        
    # Current alpha
    alpha = np.ones(rank)
    
    # Greedy zeroing
    # We want to find z such that alpha = 1 + N z has many zeros.
    # And alpha_r must be half-integers.
    
    improved = True
    while improved:
        improved = False
        for k in range(rank):
            if alpha[k] == 0:
                continue
            # Try to set alpha[k] = 0
            # 1 + (N z)_k = 0 => (N z)_k = -1
            # Solve N[:, k] dot z = -1
            # This is one equation. We can pick z with minimal norm.
            # z = N[:, k] / ||N[:, k]||^2 * (-1)
            n_k = N[:, k]
            norm_sq = np.dot(n_k, n_k)
            if norm_sq < 1e-12:
                continue
            z = -n_k / norm_sq
            new_alpha = 1 + N @ z
            
            # Check if new_alpha has half-integer non-zeros
            valid = True
            for r in range(rank):
                if abs(new_alpha[r]) > 1e-8:
                    if not _is_half_int(np.array([[new_alpha[r]]])):
                        valid = False
                        break
                else:
                    # If it's close to 0, it should be exactly 0 in the final step
                    # But for now, we just check if we can round it
                    pass
            
            if valid:
                # Accept this change
                alpha = new_alpha
                # Round small values to 0
                alpha[np.abs(alpha) < 1e-6] = 0
                improved = True
                break # Restart loop with new alpha
    
    # Apply alpha
    # For each r, if alpha[r] != 0, scale U[:, r] by alpha[r]
    # We can distribute the scaling. Let's just scale U.
    # U[:, r] *= alpha[r]
    # Check if U entries are half-integers.
    # If not, try to distribute scaling to V or W.
    
    final_U = U.copy()
    final_V = V.copy()
    final_W = W.copy()
    
    active = []
    for r in range(rank):
        if abs(alpha[r]) > 1e-8:
            a = alpha[r]
            # Scale U
            final_U[:, r] *= a
            if _is_half_int(final_U[:, r]):
                active.append(r)
            else:
                # Try scaling V
                final_U[:, r] = U[:, r] # Reset
                final_V[:, r] *= a
                if _is_half_int(final_V[:, r]):
                    active.append(r)
                else:
                    # Try scaling W
                    final_V[:, r] = V[:, r] # Reset
                    final_W[:, r] *= a
                    if _is_half_int(final_W[:, r]):
                        active.append(r)
                    else:
                        # Try splitting? e.g. sqrt(a)
                        # If a is positive, sqrt(a) might work
                        if a > 0:
                            s = np.sqrt(a)
                            final_U[:, r] = U[:, r] * s
                            final_V[:, r] = V[:, r] * s
                            final_W[:, r] = W[:, r]
                            if _is_half_int(final_U[:, r]) and _is_half_int(final_V[:, r]):
                                active.append(r)
                            else:
                                # Fail, keep original? No, we need to remove it.
                                # If we can't scale, we can't use this alpha.
                                # This means the greedy step was bad.
                                pass
                                
    # If we failed to scale some, we might have a problem.
    # For safety, if the number of active terms is not less than rank, return original.
    if len(active) >= rank:
        return U, V, W
        
    # Construct new matrices
    new_U = final_U[:, active]
    new_V = final_V[:, active]
    new_W = final_W[:, active]
    
    # Verify
    if _verify(new_U, new_V, new_W, n, m, p):
        return new_U, new_V, new_W
        
    return U, V, W

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    n, m, p = DIMENSIONS

    best = None

    def consider(U, V, W, label=""):
        nonlocal best
        if time.time() > deadline:
            return
        U = _round_half(U); V = _round_half(V); W = _round_half(W)
        if not _verify(U, V, W, n, m, p):
            return
        if not (_is_half_int(U) and _is_half_int(V) and _is_half_int(W)):
            return
        r = U.shape[1]
        if best is None or r < best[0]:
            best = (r, U, V, W)

    # 1) Kronecker Strassen baseline
    try:
        U, V, W = _build_kronecker_strassen(n, m, p)
        # Apply null space pruning
        U2, V2, W2 = _null_space_prune(U, V, W, n, m, p)
        consider(U2, V2, W2, "null_space")
        # Also consider original if pruning failed or was worse
        consider(U, V, W, "kronecker")
    except Exception:
        pass

    # 2) Fallback schoolbook
    try:
        U, V, W = _build_schoolbook(n, m, p)
        consider(U, V, W, "schoolbook")
    except Exception:
        pass

    if best is None:
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    _, U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
# EVOLVE-BLOCK-END
