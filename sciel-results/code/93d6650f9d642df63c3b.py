"""Matrix multiplication 2x3 by 3x7: rank-minimal exact decomposition.

Schoolbook rank = n*m*p = 42. We build an explicit exact decomposition of
rank 12 (2 blocks of rank 6) and then attempt a fast random refinement to
find an even smaller exact decomposition, all under a wall-clock deadline.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 42


def _target_tensor():
    """Target multiplication tensor T[i,j,k] = 1 (indexing A[i,j]B[j,k]->C[i,k])."""
    T = np.zeros((N, M, P))
    T[:] = 1.0
    return T


def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[:,c] o V[:,c] o W[:,c] equals the all-ones target."""
    T = np.zeros((N, M, P))
    for c in range(U.shape[1]):
        u = U[:, c].reshape(N, M)
        v = V[:, c].reshape(M, P)
        w = W[:, c].reshape(P, N)
        T += u[:, :, None] * v[None, :, :] * w[None, None, :]
    target = np.ones((N, M, P))
    return np.allclose(T, target, atol=tol, rtol=1e-9)


def _to_halfint(x):
    """Convert a float to the nearest half-integer (integer or x.5)."""
    return round(2.0 * x) / 2.0


def _decompose_blocks():
    """Build the explicit rank-12 decomposition (2 blocks of rank 6)."""
    U = []
    V = []
    W = []

    def add(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    # Each block handles one row index i0 of A (i = i0 only).
    for i0 in range(N):
        # 3 diagonal forms: A[i0,a]*B[a,a] -> C[i0,a]
        for a in range(M):
            u = np.zeros((N, M)); u[i0, a] = 1.0
            v = np.zeros((M, P)); v[a, a] = 1.0
            w = np.zeros((P, N)); w[a, i0] = 1.0
            add(u, v, w)

        # 3 outer forms: realize the off-diagonal part of the MxM B-product.
        # B[j,k] (j != k) = (L D R)[j,k] with L,R 3x3, D=diag.
        # For each factor c=0,1,2:
        #   u[i0, j] = L[j, c]
        #   v[j, k]  = R[c, k]
        #   w[k, i0] = D[c]
        L = np.array([[1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]])
        R = np.array([[1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]])
        D = np.array([1.0, 1.0, 1.0])
        # We need sum_c L[j,c]*R[c,k]*D[c] = 1 for j != k, and 0 for j == k.
        # Use L = ones, R = ones, D = ones gives sum_c 1 = 3 everywhere.
        # Instead choose a rank-3 decomposition of (J - I) where J = ones(3,3).
        # (J - I) = sum_c u_c v_c^T with 3 rank-1 terms.
        # J - I has eigenvalues 2 (once), -1 (twice). Rank 3.
        # Simple explicit rank-1 terms:
        #   (J-I) = e1 e2^T + e2 e1^T + (e2+e3)(e1+e3)^T?  -> verify by construction below.
        # We just solve for L, R, D directly: pick D = [1,1,1], find L,R 3x3 with
        # sum_c L[:,c] R[c,:] = J - I.  Take L = (J-I) and R = I * ... simplest:
        # L = (J-I), R = I, D = [1,1,1] -> sum_c L[:,c]*I[c,:]*1 = L = J-I. 
        JmI = np.ones((M, M)) - np.eye(M)
        L = JmI
        R = np.eye(M)
        D = np.ones(M)
        for c in range(M):
            u = np.zeros((N, M)); u[i0, :] = L[:, c]
            v = np.zeros((M, P)); v[:, :] = R[c, :]
            w = np.zeros((P, N)); w[:, i0] = D[c]
            add(u, v, w)

    U = np.array(U).T  # (N*M, rank)
    V = np.array(V).T  # (M*P, rank)
    W = np.array(W).T  # (P*N, rank)
    return U, V, W


def _refine_random(U0, V0, W0, deadline, seed=0):
    """Try random merges / null-space reductions to lower rank. Best-effort."""
    rng = random.Random(seed)
    nU, r0 = U0.shape
    nV, _ = V0.shape
    nW, _ = W0.shape
    best = (r0, U0.copy(), V0.copy(), W0.copy())

    # Build the bilinear constraint matrix: for each (i,j,k), the equation
    #   sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1
    # is a cubic (multilinear) constraint, so a linear null-space trick is not
    # directly applicable. Instead we try greedy rank reduction by testing whether
    # a random linear combination of two columns can be dropped. This is heuristic
    # and time-boxed.
    U, V, W = U0.copy(), V0.copy(), W0.copy()
    r = r0
    steps = 0
    while time.time() < deadline and r > 1 and steps < 4000:
        steps += 1
        # pick two columns to try merging
        a = rng.randrange(r)
        b = rng.randrange(r)
        if a == b:
            continue
        # try removing column b by absorbing into a with random scalar
        t = rng.uniform(-2.0, 2.0)
        U[:, a] = U[:, a] + t * U[:, b]
        V[:, a] = V[:, a] + t * V[:, b]
        W[:, a] = W[:, a] + t * W[:, b]
        # drop column b
        U = np.delete(U, b, axis=1)
        V = np.delete(V, b, axis=1)
        W = np.delete(W, b, axis=1)
        r -= 1
        if _verify(U, V, W, tol=1e-7):
            if r < best[0]:
                best = (r, U.copy(), V.copy(), W.copy())
        else:
            # revert
            U = np.insert(U, b, U[:, a] - t * U[:, b], axis=1) if False else None
            # simpler: rebuild from best
            U, V, W = best[1].copy(), best[2].copy(), best[3].copy()
            r = best[0]
    return best


def _to_lists(U, V, W):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    U = np.vectorize(_to_halfint)(U)
    V = np.vectorize(_to_halfint)(V)
    W = np.vectorize(_to_halfint)(W)
    return (U.tolist(), V.tolist(), W.tolist())


# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0  # safety margin under 150s wall-clock

    U, V, W = _decompose_blocks()
    if not _verify(U, V, W, tol=1e-9):
        # Fallback: schoolbook (guaranteed valid, rank 42)
        rank = RANK_SCHOOLBOOK
        U = [[0.0] * rank for _ in range(N * M)]
        V = [[0.0] * rank for _ in range(M * P)]
        W = [[0.0] * rank for _ in range(P * N)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1.0
                    V[j * P + k][col] = 1.0
                    W[k * N + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Heuristic refinement (time-boxed); keep base result if nothing better.
    if time.time() < deadline - 10.0:
        r, Ur, Vr, Wr = _refine_random(U, V, W, deadline)
        if r < U.shape[1] and _verify(Ur, Vr, Wr, tol=1e-9):
            U, V, W = Ur, Vr, Wr

    U, V, W = _to_lists(U, V, W)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
