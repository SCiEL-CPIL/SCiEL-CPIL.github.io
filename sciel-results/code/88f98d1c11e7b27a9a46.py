import numpy as np
import itertools
import time

D = 11
THRESH = 2  # conflict if dot > THRESH (for norm^2 = 4)


def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s limit

    # ---- Candidate generation ------------------------------------------------
    # Weight-4 vectors in {-1,0,1}^11  -> norm^2 = 4
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)

    # Weight-1 vectors in {-2,0,2}^11 -> norm^2 = 4
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)

    A4 = np.array(v4_list, dtype=np.int64)
    A1 = np.array(v1_list, dtype=np.int64)
    N4 = len(A4)

    if N4 == 0:
        return A1

    # ---- Conflict graph ------------------------------------------------------
    dots = A4 @ A4.T
    C = dots > THRESH
    np.fill_diagonal(C, False)
    C = C.astype(np.int8)

    # ---- Spectral (Fiedler) initial partition --------------------------------
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        degree = C.sum(axis=1).astype(np.float64)
        L = csr_matrix(np.diag(degree) - C.astype(np.float64))
        # smallest-magnitude eigenpairs (sigma=0 shift)
        eigvals, eigvecs = eigsh(L, k=4, sigma=0, which='LM', maxiter=400)
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx][1]
    except Exception:
        rng0 = np.random.default_rng(0)
        fiedler = rng0.standard_normal(N4)

    # ---- Local Kempe-chain flips (max-cut style) ----------------------------
    # Objective = number of "good" edges (non-conflict edges inside the selected side).
    # A vertex flip changes the objective by (deg_good - deg_bad).
    def flips(f, C, N, iters, rng, time_fn, deadline):
        deg_good = ((1 - C) * (f > 0)).sum(axis=1)
        deg_bad = (C * (f > 0)).sum(axis=1)
        gain = deg_good - deg_bad
        for _ in range(iters):
            if time_fn() > deadline - 12:
                break
            i = int(rng.integers(N))
            g = gain[i]
            if g <= 0:
                continue
            f[i] = -f[i]
            gain[i] = -g
            nb = np.where(C[i])[0]
            for j in nb:
                if f[j] == f[i]:  # edge (i,j) becomes good -> gain[j] +1
                    gain[j] += 1
                else:              # edge becomes bad -> gain[j] -1
                    gain[j] -= 1
        return f

    # ---- Multiple random restarts, keep best partition -----------------------
    rng = np.random.default_rng(42)
    best_f = None
    best_score = -1
    n_restarts = 60
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        if r == 0:
            f = np.where(fiedler >= 0, 1, -1).astype(np.int8)
        else:
            f = np.where(rng.random(N4) < 0.5, 1, -1).astype(np.int8)
        f = flips(f, C, N4, 4000, rng, time.time, deadline)
        # score = good edges
        s = int(((1 - C) * (f > 0)).sum())
        if s > best_score:
            best_score = s
            best_f = f.copy()

    if best_f is None:
        best_f = np.where(fiedler >= 0, 1, -1).astype(np.int8)

    # ---- Take the larger side as the independent set ------------------------
    pos = best_f > 0
    neg = ~pos
    if pos.sum() < neg.sum():
        selected = neg
    else:
        selected = pos

    # ---- Repair: drop any vertex that conflicts with an already-kept one ----
    kept = []
    for i in np.where(selected)[0]:
        if not (C[i, kept]).any():
            kept.append(i)

    result_v4 = A4[np.array(kept, dtype=np.int64)]

    # ---- Add all weight-1 vectors (compatible with each other and with V4) ---
    # Check compatibility of V1 with chosen V4 set: dot(v1, v4) <= 2 always holds
    # (v1 has a single 2, v4 has at most one +1/-1 at that coordinate => dot in {-2,0,2}).
    final_vectors = np.vstack([result_v4, A1])

    return final_vectors
