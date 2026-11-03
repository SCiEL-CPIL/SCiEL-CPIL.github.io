"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DL = t0 + 110  # safety margin before 120s hard kill

    n, m, p = DIMENSIONS
    Nr = n * m          # 16
    Mr = m * p          # 36
    Pr = p * n          # 36

    # Build the set of (a,b,c) triples where T[a,b,c] = 1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    Tset = set(triples)
    ntrip = len(triples)  # 144

    # Precompute index maps
    a_to_bc = {a: [] for a in range(Nr)}
    b_to_ac = {b: [] for b in range(Mr)}
    c_to_ab = {c: [] for c in range(Pr)}
    for (a, b, c) in triples:
        a_to_bc[a].append((b, c))
        b_to_ac[b].append((a, c))
        c_to_ab[c].append((a, b))

    # ---- schoolbook rank-144 baseline ----
    R0 = ntrip
    U0 = np.zeros((Nr, R0))
    V0 = np.zeros((Mr, R0))
    W0 = np.zeros((Pr, R0))
    for idx, (a, b, c) in enumerate(triples):
        U0[a, idx] = 1.0
        V0[b, idx] = 1.0
        W0[c, idx] = 1.0

    def _check_full(U, V, W, tol=1e-9):
        """Exact check over every (a,b,c) entry."""
        for (a, b, c) in triples:
            if abs(np.dot(U[a] * V[b], W[c]) - 1.0) > tol:
                return False
        for a in range(Nr):
            for b in range(Mr):
                for c in range(Pr):
                    if (a, b, c) in Tset:
                        continue
                    if abs(np.dot(U[a] * V[b], W[c])) > tol:
                        return False
        return True

    def _err(U, V, W):
        e = 0.0
        for (a, b, c) in triples:
            d = np.dot(U[a] * V[b], W[c]) - 1.0
            e += d * d
        return e

    def _als(U_init, V_init, W_init, rank, max_it=300, seed=0):
        rng = np.random.RandomState(seed)
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_e = float('inf')
        for it in range(max_it):
            if time.time() > DL - 5:
                break

            # --- update U ---
            B = np.stack([np.outer(V[:, r], W[:, r]).ravel() for r in range(rank)], axis=0)
            for a in range(Nr):
                tv = np.zeros(Mr * Pr)
                for (b, c) in a_to_bc[a]:
                    tv[b * Pr + c] = 1.0
                U[a] = np.linalg.lstsq(B.T, tv, rcond=None)[0]

            # --- update V ---
            C = np.stack([np.outer(U[:, r], W[:, r]).ravel() for r in range(rank)], axis=0)
            for b in range(Mr):
                tv = np.zeros(Nr * Pr)
                for (a, c) in b_to_ac[b]:
                    tv[a * Pr + c] = 1.0
                V[b] = np.linalg.lstsq(C.T, tv, rcond=None)[0]

            # --- update W ---
            D = np.stack([np.outer(U[:, r], V[:, r]).ravel() for r in range(rank)], axis=0)
            for c in range(Pr):
                tv = np.zeros(Nr * Mr)
                for (a, b) in c_to_ab[c]:
                    tv[a * Mr + b] = 1.0
                W[c] = np.linalg.lstsq(D.T, tv, rcond=None)[0]

            e = _err(U, V, W)
            if abs(prev_e - e) < 1e-12 * max(1.0, prev_e):
                break
            prev_e = e

        return U, V, W

    def _try_rank(rank, n_seeds=10):
        """Attempt to find a valid decomposition at a given rank."""
        if time.time() > DL - 15:
            return None

        for s in range(n_seeds):
            if time.time() > DL - 10:
                break

            # Initialization strategies
            if s % 2 == 0:
                # Structured init: subsample schoolbook
                rng0 = np.random.RandomState(1000 + rank + s)
                idx = rng0.choice(R0, size=min(rank, R0), replace=False)
                U = U0[:, idx].copy()
                V = V0[:, idx].copy()
                W = W0[:, idx].copy()
            else:
                # Random init
                rng0 = np.random.RandomState(2000 + rank + s)
                U = rng0.randn(Nr, rank) * 0.1
                V = rng0.randn(Mr, rank) * 0.1
                W = rng0.randn(Pr, rank) * 0.1

            U, V, W = _als(U, V, W, rank, max_it=200, seed=s)
            e = _err(U, V, W)
            
            if e < 1e-8:
                # Try half-integer rounding
                Ur = np.round(U * 2) / 2
                Vr = np.round(V * 2) / 2
                Wr = np.round(W * 2) / 2
                if _check_full(Ur, Vr, Wr, tol=1e-9):
                    return Ur, Vr, Wr
                # Try integer rounding
                Ui = np.round(U)
                Vi = np.round(V)
                Wi = np.round(W)
                if _check_full(Ui, Vi, Wi, tol=1e-9):
                    return Ui, Vi, Wi
                # If very close, try to nudge?
                # For now, if not rounded, skip.
        return None

    # Search ranks
    # Lower bound 36. Schoolbook 144.
    # Try to find something significantly better.
    candidate_ranks = [120, 100, 90, 80, 70, 60, 50]
    
    best_rank = R0
    best = (U0, V0, W0)

    for rk in candidate_ranks:
        if time.time() > DL - 20:
            break
        res = _try_rank(rk, n_seeds=8)
        if res is not None:
            U, V, W = res
            best_rank = rk
            best = (U, V, W)
            break  # Found a valid one, stop searching lower ranks? 
            # Actually, we might want to try lower if time permits, 
            # but finding one is a big jump from 144.
            # Let's stick with the first found to ensure validity.

    if best_rank < R0:
        Uo, Vo, Wo = best
        return {"U": Uo.tolist(), "V": Vo.tolist(), "W": Wo.tolist()}

    # Fallback: schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
