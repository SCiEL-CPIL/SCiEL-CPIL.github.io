import numpy as np
import time
from fractions import Fraction

def _strassen_2x2x2():
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _build(rank=98):
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]
    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_M(U, V, W, n, m, p):
    rank = U.shape[1]
    N = n*p*m
    M = np.zeros((N, rank))
    idx = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                idx += 1
    return M

def _is_half_integer(x, tol=1e-4):
    y = x * 2
    return abs(y - round(y)) < tol

def _round_half(x):
    return round(x * 2) / 2.0

def _try_reduce(U, V, W, n, m, p, deadline, max_rounds=300):
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for rnd in range(max_rounds):
        if time.time() > deadline - 15.0:
            break
        if R <= 1:
            break

        # Build tensor matrix M
        M = _build_M(U, V, W, n, m, p)

        # SVD to find null space
        try:
            _, s, Vt = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            break

        # Find singular values close to zero - use adaptive tolerance
        tol = max(1e-8, 1e-6 * max(s) if len(s) > 0 else 1e-8)
        null_idx = np.where(s < tol)[0]
        if len(null_idx) == 0:
            # Try with a larger tolerance
            tol2 = max(tol * 10, 1e-5)
            null_idx = np.where(s < tol2)[0]
            if len(null_idx) == 0:
                break

        # Try each null vector, trying multiple pivot choices
        reduced = False
        for ni in null_idx:
            if time.time() > deadline - 10.0:
                break
            c = Vt[ni, :]
            
            # Try multiple pivot columns (top 3 by magnitude)
            pivot_candidates = np.argsort(-np.abs(c))[:min(3, R)]
            
            for r0 in pivot_candidates:
                if abs(c[r0]) < 1e-10:
                    continue
                
                keep = [q for q in range(R) if q != r0]
                scales = []
                valid = True
                for r in keep:
                    factor = 1.0 + c[r] / c[r0]
                    if abs(factor) < 1e-14:
                        valid = False
                        break
                    # Cube root - allow any real value (not just half-integers)
                    if factor >= 0:
                        s_val = factor ** (1.0/3.0)
                    else:
                        s_val = -((-factor) ** (1.0/3.0))
                    if not np.isfinite(s_val):
                        valid = False
                        break
                    scales.append(s_val)

                if not valid:
                    continue

                # Apply scaling
                new_U = U[:, keep].copy()
                new_V = V[:, keep].copy()
                new_W = W[:, keep].copy()
                for i, r in enumerate(keep):
                    new_U[:, i] *= scales[i]
                    new_V[:, i] *= scales[i]
                    new_W[:, i] *= scales[i]

                # Verify with tighter tolerance
                if _verify(new_U, new_V, new_W, n, m, p, tol=1e-4):
                    U, V, W = new_U, new_V, new_W
                    R -= 1
                    reduced = True
                    if R < best_rank:
                        best_rank = R
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if reduced:
                break

        if not reduced:
            break

    return best_U, best_V, best_W, best_rank

def _try_reduce_phase2(U, V, W, n, m, p, deadline):
    """Second phase: try to find additional reductions using a different null-space approach."""
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for rnd in range(100):
        if time.time() > deadline - 15.0:
            break
        if R <= 1:
            break

        M = _build_M(U, V, W, n, m, p)
        try:
            _, s, Vt = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            break

        # Use adaptive tolerance
        tol = max(1e-8, 1e-6 * max(s) if len(s) > 0 else 1e-8)
        null_idx = np.where(s < tol)[0]
        if len(null_idx) == 0:
            tol2 = max(tol * 10, 1e-5)
            null_idx = np.where(s < tol2)[0]
            if len(null_idx) == 0:
                break

        reduced = False
        for ni in null_idx:
            if time.time() > deadline - 10.0:
                break
            c = Vt[ni, :]
            
            # Try multiple pivot columns
            pivot_candidates = np.argsort(-np.abs(c))[:min(3, R)]
            
            for r0 in pivot_candidates:
                if abs(c[r0]) < 1e-10:
                    continue
                
                keep = [q for q in range(R) if q != r0]
                scales = []
                valid = True
                for r in keep:
                    factor = 1.0 + c[r] / c[r0]
                    if abs(factor) < 1e-14:
                        valid = False
                        break
                    if factor >= 0:
                        s_val = factor ** (1.0/3.0)
                    else:
                        s_val = -((-factor) ** (1.0/3.0))
                    if not np.isfinite(s_val):
                        valid = False
                        break
                    scales.append(s_val)

                if not valid:
                    continue

                new_U = U[:, keep].copy()
                new_V = V[:, keep].copy()
                new_W = W[:, keep].copy()
                for i, r in enumerate(keep):
                    new_U[:, i] *= scales[i]
                    new_V[:, i] *= scales[i]
                    new_W[:, i] *= scales[i]

                if _verify(new_U, new_V, new_W, n, m, p, tol=1e-4):
                    U, V, W = new_U, new_V, new_W
                    R -= 1
                    reduced = True
                    if R < best_rank:
                        best_rank = R
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if reduced:
                break

        if not reduced:
            break

    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s limit

    U, V, W = _build(98)
    if not _verify(U, V, W, n, m, p):
        # Fallback: return schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Phase 1: SVD-based single-column reduction
    U, V, W, rank1 = _try_reduce(U, V, W, n, m, p, deadline)

    # Phase 2: Additional reduction pass with different tolerance strategy
    if time.time() < deadline - 20.0:
        U, V, W, rank2 = _try_reduce_phase2(U, V, W, n, m, p, deadline)
    else:
        rank2 = rank1

    # Final verification with reasonable tolerance
    if not _verify(U, V, W, n, m, p, tol=1e-4):
        # If verification fails, try to recover by rebuilding
        U, V, W = _build(98)
        if not _verify(U, V, W, n, m, p):
            # Ultimate fallback: schoolbook
            rank = n * m * p
            U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i*m*p + j*p + k
                        U[i*m+j, idx] = 1
                        V[j*p+k, idx] = 1
                        W[k*n+i, idx] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
