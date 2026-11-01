import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 2, 9]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_c U[:,c] outer V[:,c] outer W[:,c] equals the MM tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build tensor T[a,b,c] where a in A index (n*m), b in B index (m*p), c in C index (p*n)
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    # Target: C[i,k] = sum_j A[i,j]*B[j,k]
    # A index: i*m + j ; B index: j*p + k ; C index: k*n + i
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target[a,b,c] = 1.0
    diff = np.abs(T - target)
    return np.max(diff) < tol

def construct():
    start = time.time()
    deadline = 120.0
    n, m, p = DIMENSIONS
    Na, Nb, Nc = n*m, m*p, p*n
    # schoolbook fallback
    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(Na)]
        V = [[0]*rank for _ in range(Nb)]
        W = [[0]*rank for _ in range(Nc)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = best["U"][0].__len__() if best["U"] else n*m*p

    # Try to find lower rank via random bilinear decomposition + rounding.
    # Model: T = sum_r u_r v_r w_r^T (as 3-tensor). We search for rank r < 36.
    # Use a "bilinear" parameterization: we want T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    # This is exactly a rank-r CP-like decomposition but with the tensor being the MM tensor.
    # We'll use an alternating least squares (ALS) on the flattened tensor.

    target = np.zeros((Na, Nb, Nc))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    # Tensor flattening: for CP decomposition, we need mode unfoldings.
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Mode-0 unfolding: reshape T to (Na, Nb*Nc) -> = U * (V ⊙ W)^T
    # We'll do ALS: fix two, solve the third via SVD.

    def try_rank(r, iters=40, seed=0):
        rng = np.random.RandomState(seed)
        U = rng.randn(Na, r)
        V = rng.randn(Nb, r)
        W = rng.randn(Nc, r)
        # Scale to match norm
        for _ in range(iters):
            # Update U: T_(0) = U (V ⊙ W)^T
            VW = np.kron(V, W)  # (Nb*Nc, r)
            # T_(0) shape (Na, Nb*Nc)
            T0 = target.reshape(Na, Nb*Nc)
            U, _, _ = np.linalg.lstsq(VW.T, T0.T, rcond=None)
            # Update V: T_(1) = V (W ⊙ U)^T
            WU = np.kron(W, U)
            T1 = target.transpose(1,2,0).reshape(Nb, Nc*Na)
            V, _, _ = np.linalg.lstsq(WU.T, T1.T, rcond=None)
            # Update W: T_(2) = W (U ⊙ V)^T
            UV = np.kron(U, V)
            T2 = target.transpose(2,0,1).reshape(Nc, Na*Nb)
            W, _, _ = np.linalg.lstsq(UV.T, T2.T, rcond=None)
        # Reconstruct
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - target))
        return err, U, V, W

    # Try descending ranks
    for r in range(20, best_rank):
        if time.time() > deadline - 20:
            break
        found = False
        for seed in range(6):
            if time.time() > deadline - 15:
                break
            try:
                err, U, V, W = try_rank(r, iters=30, seed=seed)
            except Exception:
                continue
            if err < 1e-6:
                # Round to nearest half-integer
                Ur = np.round(U * 2) / 2.0
                Vr = np.round(V * 2) / 2.0
                Wr = np.round(W * 2) / 2.0
                if _verify(Ur.tolist(), Vr.tolist(), Wr.tolist(), n, m, p, tol=1e-6):
                    best = {"U": Ur.tolist(), "V": Vr.tolist(), "W": Wr.tolist()}
                    best_rank = r
                    found = True
                    break
        if found:
            # try to go even lower
            continue

    if time.time() < deadline - 5:
        # Final: try to polish best_rank with more iterations
        pass

    # Convert to list of lists of numbers (int or float)
    def to_clean(mat):
        out = []
        for row in mat:
            nr = []
            for v in row:
                if isinstance(v, float):
                    if abs(v - round(v)) < 1e-9:
                        nr.append(int(round(v)))
                    else:
                        nr.append(v)
                else:
                    nr.append(v)
            out.append(nr)
        return out

    best["U"] = to_clean(best["U"])
    best["V"] = to_clean(best["V"])
    best["W"] = to_clean(best["W"])
    return best
