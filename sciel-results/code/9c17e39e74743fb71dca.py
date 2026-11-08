"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def construct():
    """Build the rank-minimal exact tensor decomposition for a 2×3·3×6 matmul.
    Strategy: construct an explicit Kronecker-like base, then use a small
    structured-search / algebraic elimination over the tensor entries to
    try to collapse the rank while keeping exactness. The search is bounded
    by an explicit deadline and the naive decomposition is always a valid
    fallback.
    """
    start_time = time.time()
    deadline = start_time + 145   # leave 5 s safety margin before 150 s wall-clock

    n, m, p = DIMENSIONS           # A: (n×m), B: (m×p), C: (n×p)
    nm, mp, pn = n*m, m*p, p*n     # flat sizes for U, V, W rows
    T_size = nm * mp * pn

    # ---------- Naive rank-36 base (always valid) ----------
    def naive_decomposition():
        U = np.zeros((nm, n*m*p), dtype=float)
        V = np.zeros((mp, n*m*p), dtype=float)
        W = np.zeros((pn, n*m*p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = (i*m + j)*p + k
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
        return U, V, W

    # ---------- Verify that a candidate decomposition exactly realizes C = A·B ----------
    def verify(Uc, Vc, Wc, atol=1e-8):
        Uc = np.asarray(Uc, dtype=float)
        Vc = np.asarray(Vc, dtype=float)
        Wc = np.asarray(Wc, dtype=float)
        # Test with several random integer / simple arrays (exactness required).
        for seed in [0, 1, 7, 123, 999]:
            rng = np.random.RandomState(seed)
            A = rng.randint(0, 3, (n, m)).astype(float)
            B = rng.randint(0, 3, (m, p)).astype(float)
            C_ref = A @ B
            # bilinear: C[i,k] = sum_r W[k*n+i, r] (U_r · A_flat) (V_r · B_flat)
            Af = A.reshape(-1)
            Bf = B.reshape(-1)
            Cv = np.einsum('kr,i,ki->', np.arange(0), np.arange(0), np.arange(0))  # no-op
            Cv = np.zeros((n, p), dtype=float)
            R = Uc.shape[1]
            uvals = Uc.T @ Af     # (R,)
            vvals = Vc.T @ Bf     # (R,)
            for r in range(R):
                Cv += np.outer(np.repeat(1, p), 1) * (Wc[:, r] * (uvals[r] * vvals[r]))
                Cv = Cv.reshape(n, p)
            if not np.allclose(Cv, C_ref, atol=atol):
                return False
        return True

    # ---------- Rank-33 attempt from seed (3 blocks: 7 Strassen terms + 4 scalar) ----------
    def seed_rank_33():
        U = [[0.0]*33 for _ in range(nm)]
        V = [[0.0]*33 for _ in range(mp)]
        W = [[0.0]*33 for _ in range(pn)]
        terms = []
        # local 2×2 sub-blocks on the A columns (j_local) and B rows (k_local)
        terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
        terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
        terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
        terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
        terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
        terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
        terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
        # third j-column (j=2) scalar products (K1..K4)
        terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
        terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
        terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
        terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
        for b in range(3):
            for ti, t in enumerate(terms):
                col = b*11 + ti
                for idx, val in t['u'].items():
                    U[idx][col] = val
                for li, val in t['v'].items():
                    jl = li // 2
                    kl = li % 2
                    gidx = jl*p + (b*2 + kl)
                    V[gidx][col] = val
                for li, val in t['w'].items():
                    il = li // 2
                    kl = li % 2
                    gidx = (b*2 + kl)*n + il
                    W[gidx][col] = val
        return np.array(U), np.array(V), np.array(W)

    # ---------- Build the naive tensor (as a dense ndarray of 0/1) ----------
    def tensor_from_decomp(U, V, W):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        R = U.shape[1]
        T = np.zeros((nm, mp, pn), dtype=float)
        for r in range(R):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return T

    # ---------- Try to reduce rank from a tensor via structured column merging ----------
    def try_reduce_rank(T, start_rank, budget_seconds):
        """Given a tensor T built from start_rank rank-1 terms, attempt a
        small structured search over signed / halved column merges that preserve
        the tensor exactly while lowering rank. Uses randomized linear
        elimination over the support of the tensor.
        Returns (best_rank, best_U, best_V, best_W) or (None,...) if none found.
        """
        best_rank = start_rank
        # We will operate directly on the rank-1 factor matrices (not on T)
        U0, V0, W0 = tensor_to_factors(T, start_rank)
        if U0 is None:
            return None, None, None, None
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        T0 = tensor_from_decomp(U0, V0, W0)

        def factors_to_tensor(Uf, Vf, Wf):
            return tensor_from_decomp(Uf, Vf, Wf)

        def rand_signed_halded_merge_factors(Uf, Vf, Wf, n_trials=200):
            """Attempt signed / halved linear combinations to collapse columns
            while preserving the tensor. This is a heuristic: for each pair of
            rank-1 terms (a,b), attempt a signed merge if the resulting column
            lies in the span of the tensor's linear image (which is guaranteed
            because we are reconstructing T)."""
            R0 = Uf.shape[1]
            T_cur = factors_to_tensor(Uf, Vf, Wf)
            best = (R0, Uf.copy(), Vf.copy(), Wf.copy())
            for _ in range(n_trials):
                if time.time() - start_time > budget_seconds:
                    break
                i, j = (np.random.default_rng().choice(R0, 2, replace=False))
                for s in (1.0, -1.0, 0.5, -0.5, 2.0, -2.0):
                    # new column r' = i + s*j, delete j
                    U2 = Uf.copy()
                    V2 = Vf.copy()
                    W2 = Wf.copy()
                    U2[:, i] = U2[:, i] + s * U2[:, j]
                    V2[:, i] = V2[:, i] + s * V2[:, j]
                    W2[:, i] = W2[:, i] + s * W2[:, j]
                    # remove column j
                    keep = [k for k in range(R0) if k != j]
                    U2 = U2[:, keep]
                    V2 = V2[:, keep]
                    W2 = W2[:, keep]
                    # check exact tensor preservation
                    if np.allclose(factors_to_tensor(U2, V2, W2), T0, atol=1e-10):
                        if R0 - 1 < best[0]:
                            best = (R0 - 1, U2, V2, W2)
                            break
                if time.time() - start_time > budget_seconds:
                    break
            return best[1], best[2], best[3], best[0]

        best_U, best_V, best_W, best_rank = rand_signed_halded_merge_factors(U0, V0, W0)
        return best_rank, best_U, best_V, best_W

    # ---------- Tensor → naive rank-1 factors (for a known rank count) ----------
    def tensor_to_factors(T, rank):
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        cnt = 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if abs(T[a, b, c]) > 1e-12:
                        if cnt >= rank:
                            return None, None, None
                        U[a, cnt] = 1.0
                        V[b, cnt] = 1.0
                        W[c, cnt] = 1.0
                        T[a, b, c] -= 1.0  # will be re-built from factors
                        cnt += 1
        if cnt < rank:
            return U, V, W
        return U, V, W

    # ---------- Main: try candidate strategies, always return the best valid ----------
    candidate_rank = None
    candidate_U = candidate_V = candidate_W = None

    # Strategy 0: seed rank-33
    U33, V33, W33 = seed_rank_33()
    if verify(U33, V33, W33):
        candidate_rank = 33
        candidate_U, candidate_V, candidate_W = U33, V33, W33

    # Strategy 1: try to reduce rank from seed rank-33
    if time.time() < deadline - 60:
        T33 = tensor_from_decomp(U33, V33, W33)
        r_red, U_red, V_red, W_red = try_reduce_rank(T33, 33, deadline - 75)
        if r_red is not None and r_red < 33 and verify(U_red, V_red, W_red):
            candidate_rank = r_red
            candidate_U, candidate_V, candidate_W = U_red, V_red, W_red

    # Strategy 2: naive rank-36 always valid fallback
    if candidate_rank is None:
        U6, V6, W6 = naive_decomposition()
        candidate_rank = 36
        candidate_U, candidate_V, candidate_W = U6, V6, W6

    # Final safety: if anything failed, return naive
    if not verify(candidate_U, candidate_V, candidate_W):
        U6, V6, W6 = naive_decomposition()
        candidate_rank = 36
        candidate_U, candidate_V, candidate_W = U6, V6, W6

    # Convert to plain lists to match the original interface
    U_list = candidate_U.tolist()
    V_list = candidate_V.tolist()
    W_list = candidate_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
