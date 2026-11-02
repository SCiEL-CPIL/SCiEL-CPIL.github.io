import time
import numpy as np
import random

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS

    # Build a rank-28 Strassen-style decomposition for 2x4 * 4x4
    R = 28
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))

    # Seven base 2x2 Strassen-like terms (each entry is a dict of local 2x2 indices -> coeff)
    base_terms = [
        ({0: 1, 3: 1},  {0: 1, 3: 1},  {0: 1, 3: 1}),
        ({2: 1, 3: 1},  {0: 1},         {2: 1, 3: -1}),
        ({0: 1},         {1: 1, 3: -1},  {1: 1, 3: 1}),
        ({3: 1},         {0: -1, 2: 1},  {0: 1, 2: 1}),
        ({0: 1, 1: 1},   {3: 1},         {-1 * 1: 0} | {0: -1, 1: 1}),
        ({-1 * 1: 0} | {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({-1 * 1: 0} | {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    def emit(a_off, b_row, b_col, c_col, col0):
        for t, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            cidx = col0 + t
            # U part
            for loc, cf in u_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = lr, a_off * 2 + lc
                U[gr * 4 + gc, cidx] += cf
            # V part
            for loc, cf in v_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = b_row * 2 + lr, b_col * 2 + lc
                V[gr * 4 + gc, cidx] += cf
            # W part (output trace convention W[k*n+i, r])
            for loc, cf in w_loc.items():
                lr, lc = divmod(loc, 2)
                gr, gc = lr, c_col * 2 + lc
                W[gc * 2 + gr, cidx] += cf

    # Four 2x2 blocks
    emit(0, 0, 0, 0, 0)
    emit(1, 1, 0, 0, 7)
    emit(0, 0, 1, 1, 14)
    emit(1, 1, 1, 1, 21)

    # Verification helper
    def verify(Um, Vm, Wm, seed=42):
        rs = np.random.RandomState(seed)
        A = rs.randn(n, m)
        B = rs.randn(m, p)
        Cexp = A @ B
        Ccomp = np.zeros((n, p))
        rank_k = Um.shape[1]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank_k):
                    for j in range(m):
                        s += Um[i * m + j, r] * Vm[j * p + k, r] * Wm[k * n + i, r]
                Ccomp[i, k] = s
        return np.allclose(Cexp, Ccomp, atol=1e-10)

    # Dependency reduction
    active = list(range(R))
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Shuffle the order of removal attempts each pass for variety
        order = active[:]
        random.Random(int(time.time() * 1000)).shuffle(order)
        for i in order:
            if i not in active:
                continue
            trial = [x for x in active if x != i]
            if len(trial) < 1:
                continue
            try:
                M = np.column_stack((U[:, trial], V[:, trial], W[:, trial]))
                Mi = np.column_stack((U[:, [i]], V[:, [i]], W[:, [i]]))
                if M.shape[1] <= Mi.shape[1]:
                    sol, *_ = np.linalg.lstsq(M, Mi, rcond=None)
                    resid = np.linalg.norm(M @ sol - Mi)
                else:
                    # Use SVD-based projection
                    Usvd, svals, Vt = np.linalg.svd(M, full_matrices=False)
                    # Project Mi onto column space of M
                    proj = Usvd @ (Usvd.T @ Mi)
                    resid = np.linalg.norm(proj - Mi)
                if resid < 1e-10:
                    active = trial
                    changed = True
                    break
            except Exception:
                continue

    # Polish: try to zero out small entries (should be exactly 0 already but clean up)
    for arr in (U, V, W):
        mask = np.abs(arr) < 1e-12
        arr[mask] = 0.0
        # Snap to half-integers
        arr *= 2
        np.round(arr, out=arr)
        arr *= 0.5

    # Final check
    if verify(U, V, W):
        return {"U": U[:, active].tolist(), "V": V[:, active].tolist(), "W": W[:, active].tolist()}

    # Fallback to full rank
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
