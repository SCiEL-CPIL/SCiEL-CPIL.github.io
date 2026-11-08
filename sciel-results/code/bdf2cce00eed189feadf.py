"""Search for a low-rank exact tensor decomposition of the 4x4 * 4x7 multiplication tensor."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

_START = time.time()
_DEADLINE = 150.0 - 20.0   # safety margin

def _slices():
    """Return the 112 multiplication-tensor slices, each a (16,28) matrix."""
    n, m, p = DIMENSIONS
    sl = []
    for i in range(n):
        for k in range(p):
            M = np.zeros((n * m, m * p))
            for j in range(m):
                M[i * m + j, j * p + k] = 1.0
            sl.append(M)
    return sl

def _flat(slices):
    return [s.reshape(-1) for s in slices]

def _span_check(flat, cols):
    """Return True if all slice vectors lie in span of cols (tolerance-based)."""
    A = np.column_stack(cols)
    if A.size == 0:
        return False
    Q, _ = np.linalg.qr(A)
    for v in flat:
        resid = v - Q @ (Q.T @ v)
        if np.linalg.norm(resid) > 1e-6:
            return False
    return True

def _basis_from_sums(flat, cols, seed):
    """Build a candidate basis by random linear combinations of the slices."""
    rng = random.Random(seed)
    idxs = list(range(len(flat)))
    basis = []
    for _ in range(len(flat)):
        if not _span_check(flat, basis):
            basis.append(flat[rng.choice(idxs)])
        else:
            break
    return basis

def _try_rank(flat, r, rng):
    """Try to find r basis tensors whose span covers all slices."""
    for _ in range(400):
        if not _span_check(flat, []):
            pass
        # random linear combinations of r slices
        idxs = list(range(len(flat)))
        cols = []
        ok = True
        for _ in range(r):
            c = np.zeros(flat[0].shape)
            for _ in range(6):
                a = rng.uniform(-2.0, 2.0)
                c = c + a * flat[rng.choice(idxs)]
            if np.linalg.norm(c) > 1e-8:
                cols.append(c)
        if len(cols) < r:
            continue
        if _span_check(flat, cols):
            # reduce to a minimal spanning set of <= r
            basis = []
            for c in cols:
                if not _span_check(flat, basis):
                    basis.append(c)
                else:
                    break
            if len(basis) <= r and _span_check(flat, basis):
                return basis
    return None

def _build_result(basis, rank):
    n, m, p = DIMENSIONS
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c, mat in enumerate(basis):
        M = mat.reshape(n * m, m * p)
        for idx in range(n * m):
            for jdx in range(m * p):
                val = M[idx, jdx]
                if val == 0:
                    continue
                U[idx][c] = val
                V[jdx][c] = val
                # W row index k*n + i where idx=i*m+j, jdx=j*p+k
                i = idx // m
                j = idx % m
                jj = jdx // p
                k = jdx % p
                # only when j == jj (the product term)
                if j == jj:
                    W[k * n + i][c] = val
    return {"U": U, "V": V, "W": W}

def _verify(result, tol=1e-6):
    n, m, p = DIMENSIONS
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    for i in range(n):
        for k in range(p):
            # C[i,k] = sum_j A[i,j] B[j,k]
            # = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c])
            # Build the slice from U,V
            S = np.zeros((n * m, m * p))
            for c in range(U.shape[1]):
                S += np.outer(U[:, c], V[:, c])
            # check: for each j, S[i*m+j, j*p+k] should equal W[k*n+i,c] combined
            # Actually the decomposition: C[i,k] = sum_j A[i,j]B[j,k]
            # = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c])
            # and we need sum_j U[i*m+j,c]*V[j*p+k,c] = S[i*m+j, j*p+k] summed over j
            val = 0.0
            for j in range(m):
                val += S[i * m + j, j * p + k]
            expected = 1.0 if (i >= 0 and k >= 0) else 0.0
            # the tensor value at (i,k) is sum over c of W * (the contraction)
            # but our construction already bakes the contraction into S
            # So the slice value should be sum_j S[i*m+j, j*p+k]
            if abs(val - 1.0) > tol:
                return False
    return True

def construct():
    global _START
    flat = _flat(_slices())
    rng = random.Random(12345)

    # Start with schoolbook rank as a safe fallback
    n, m, p = DIMENSIONS
    best_rank = n * m * p
    best_basis = None

    # Build schoolbook basis as fallback
    school = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec = np.zeros(n * m * m * p)
                vec[(i * m + j) * (m * p) + (j * p + k)] = 1.0
                school.append(vec)
    best_basis = school

    # Try to find a lower rank via random linear-combination search
    for r in range(n * m, best_rank):
        if time.time() > _DEADLINE:
            break
        basis = _try_rank(flat, r, rng)
        if basis is not None:
            best_rank = r
            best_basis = basis
            break

    if best_basis is None:
        best_basis = school

    result = _build_result(best_basis, len(best_basis))
    return result
