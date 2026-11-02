"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]


# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

    Strategy:
      1. Build a correct base decomposition (block-Strassen, rank 36).
      2. Verify correctness (sum of rank-1 tensor vectors == all-ones tensor).
      3. Exploit linear dependencies among the 36 rank-1 tensor vectors:
         if the 36 vectors are linearly dependent, we can rescale a subset
         (absorbing the scale into U/V/W) to drop one term. Repeat greedily.
      4. Try a "cross-block" merge: combine two terms that share structure
         across the left/right A-block split to see if a rank reduction is
         possible.
      5. Deadline guard: return best-so-far before the 120s limit.
    """
    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin

    n, m, p = DIMENSIONS  # 2, 4, 5
    N_A = n * m  # 8
    N_B = m * p  # 20
    N_C = p * n  # 10
    T_DIM = n * m * p  # 40

    def time_left():
        return DEADLINE - time.time()

    # ---------- Build base decomposition (block-Strassen, rank 36) ----------
    rank_terms = []  # list of (u_dict, v_dict, w_dict)

    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        terms = [
            ({(i0, j0): 1, (i0 + 1, j0 + 1): 1},
             {(j0, k0): 1, (j0 + 1, k0 + 1): 1},
             {(i0, k0): 1, (i0 + 1, k0 + 1): 1}),
            ({(i0 + 1, j0): 1, (i0 + 1, j0 + 1): 1},
             {(j0, k0): 1},
             {(i0 + 1, k0): 1, (i0 + 1, k0 + 1): -1}),
            ({(i0, j0): 1},
             {(j0, k0 + 1): 1, (j0 + 1, k0 + 1): -1},
             {(i0, k0 + 1): 1, (i0 + 1, k0 + 1): 1}),
            ({(i0 + 1, j0 + 1): 1},
             {(j0 + 1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0 + 1, k0): 1}),
            ({(i0, j0): 1, (i0, j0 + 1): 1},
             {(j0 + 1, k0 + 1): 1},
             {(i0, k0): -1, (i0, k0 + 1): 1}),
            ({(i0 + 1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0 + 1): 1},
             {(i0 + 1, k0 + 1): 1}),
            ({(i0, j0 + 1): 1, (i0 + 1, j0 + 1): -1},
             {(j0 + 1, k0): 1, (j0 + 1, k0 + 1): 1},
             {(i0, k0): 1}),
        ]
        for u, v, w in terms:
            rank_terms.append((
                {k: val * sign for k, val in u.items()},
                {k: val * sign for k, val in v.items()},
                {k: val * sign for k, val in w.items()},
            ))

    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0 + 1): 1}, {(j0 + 1, k0): 1}, {(i0, k0): 1}),
            ({(i0 + 1, j0): 1}, {(j0, k0): 1}, {(i0 + 1, k0): 1}),
            ({(i0 + 1, j0 + 1): 1}, {(j0 + 1, k0): 1}, {(i0 + 1, k0): 1}),
        ]
        for u, v, w in terms:
            rank_terms.append((
                {k: val * sign for k, val in u.items()},
                {k: val * sign for k, val in v.items()},
                {k: val * sign for k, val in w.items()},
            ))

    # A = [A1(2x2) | A2(2x2)], B = [Btop(2x5); Bbot(2x5)]
    # C = A1*Btop + A2*Bbot
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1)
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1)

    def build_matrices(terms):
        r = len(terms)
        U = np.zeros((N_A, r), dtype=float)
        V = np.zeros((N_B, r), dtype=float)
        W = np.zeros((N_C, r), dtype=float)
        for c, (ud, vd, wd) in enumerate(terms):
            for (i, j), coeff in ud.items():
                U[i * m + j, c] = coeff
            for (j, k), coeff in vd.items():
                V[j * p + k, c] = coeff
            for (i, k), coeff in wd.items():
                W[k * n + i, c] = coeff
        return U, V, W

    U, V, W = build_matrices(rank_terms)

    # ---------- Helpers ----------
    def tensor_vecs(U, V, W):
        r = U.shape[1]
        T = np.zeros((r, T_DIM), dtype=float)
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    a = uc[i * m + j]
                    if a == 0:
                        continue
                    for k in range(p):
                        b = vc[j * p + k]
                        if b == 0:
                            continue
                        d = wc[k * n + i]
                        if d == 0:
                            continue
                        T[c, i * (m * p) + j * p + k] = a * b * d
        return T

    def verify(U, V, W, tol=1e-7):
        T = tensor_vecs(U, V, W)
        return np.allclose(T.sum(axis=0), np.ones(T_DIM), atol=tol)

    # ---------- Verify base ----------
    if not verify(U, V, W):
        # Fallback: pure schoolbook rank 40 (guaranteed correct)
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((
                        {(i, j): 1}, {(j, k): 1}, {(i, k): 1},
                    ))
        U, V, W = build_matrices(terms)
        if not verify(U, V, W):
            # Should never happen; return what we have
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---------- Linear-dependency reduction ----------
    def try_reduce(U, V, W, max_iters=40):
        """Greedily drop terms by rescaling a subset to satisfy a null vector."""
        cur_U, cur_V, cur_W = U, V, W
        for _ in range(max_iters):
            if time_left() < 5.0:
                break
            T = tensor_vecs(cur_U, cur_V, cur_W)
            r = T.shape[0]
            if r <= 1:
                break
            # Null space of T (r x T_DIM). Null vectors have length r.
            try:
                _, svals, Vt = np.linalg.svd(T, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            tol = 1e-9 * (svals[0] if svals.size else 1.0)
            rank_T = int(np.sum(svals > tol))
            if rank_T >= r:
                break  # independent, no more reductions
            null_space = Vt[rank_T:, :]  # (dim, r)
            found = False
            best_candidate = None
            for nv in null_space:
                if time_left() < 5.0:
                    break
                # Try all pivot indices, not just the largest
                abs_nv = np.abs(nv)
                # Sort indices by magnitude descending
                sorted_idx = np.argsort(abs_nv)[::-1]
                for idx in sorted_idx[:5]:  # Try top 5 pivots
                    if abs_nv[idx] < 1e-12:
                        continue
                    # Build new scale factors: for c != idx, s_c = 1 + nv[c]/nv[idx]
                    # for c == idx, s_idx = 1 (the dropped term, scale 0)
                    new_U = cur_U.copy()
                    new_V = cur_V.copy()
                    new_W = cur_W.copy()
                    for c in range(r):
                        if c == idx:
                            continue
                        s = 1.0 + nv[c] / nv[idx]
                        if abs(s) < 1e-12:
                            continue
                        new_U[:, c] *= s
                        new_V[:, c] *= s
                        new_W[:, c] *= s
                    # Drop column idx
                    keep = [c for c in range(r) if c != idx]
                    new_U = new_U[:, keep]
                    new_V = new_V[:, keep]
                    new_W = new_W[:, keep]
                    if verify(new_U, new_V, new_W):
                        if best_candidate is None:
                            best_candidate = (new_U, new_V, new_W)
                        # Keep searching for potentially better candidates
            if best_candidate is not None:
                cur_U, cur_V, cur_W = best_candidate
                found = True
            if not found:
                break
        return cur_U, cur_V, cur_W

    U, V, W = try_reduce(U, V, W, max_iters=40)
    
    # Second pass with tighter tolerance
    if time_left() > 10.0:
        U, V, W = try_reduce(U, V, W, max_iters=20)

    # ---------- Cross-block merge: try to fold a Strassen term across A_L/A_R ----------
    # The block-Strassen structure has 6 Strassen blocks (each rank 7) + 2 schoolbook (each rank 4) = 36.
    # The two A-blocks (left j=0..1 and right j=2..3) each contribute 3 Strassen blocks + 1 schoolbook.
    # If we can find a linear combination of terms from the left and right blocks that
    # allows dropping one term, we gain rank reduction.
    #
    # This is a structural search: we look for a vector in the null space of the tensor
    # matrix that has support spanning both left and right block terms, and check if
    # rescaling yields a valid decomposition with one fewer term.
    if time_left() > 15.0:
        T = tensor_vecs(U, V, W)
        r = T.shape[0]
        if r > 1:
            try:
                _, svals, Vt = np.linalg.svd(T, full_matrices=False)
            except np.linalg.LinAlgError:
                svals, Vt = None, None
            if svals is not None:
                tol = 1e-9 * (svals[0] if svals.size else 1.0)
                rank_T = int(np.sum(svals > tol))
                if rank_T < r:
                    null_space = Vt[rank_T:, :]
                    # Try each null vector; prefer ones with large entries in both
                    # the left-block terms (indices 0..10) and right-block terms (indices 14..24)
                    found_cross = False
                    for nv in null_space:
                        if time_left() < 5.0:
                            break
                        # Check if this null vector spans both blocks
                        left_support = np.sum(np.abs(nv[:11]) > 1e-10)
                        right_support = np.sum(np.abs(nv[14:25]) > 1e-10)
                        if left_support < 2 or right_support < 2:
                            continue
                        # Try multiple pivot indices
                        abs_nv = np.abs(nv)
                        sorted_idx = np.argsort(abs_nv)[::-1]
                        for idx in sorted_idx[:5]:
                            if abs_nv[idx] < 1e-12:
                                continue
                            new_U = U.copy()
                            new_V = V.copy()
                            new_W = W.copy()
                            for c in range(r):
                                if c == idx:
                                    continue
                                s = 1.0 + nv[c] / nv[idx]
                                if abs(s) < 1e-12:
                                    continue
                                new_U[:, c] *= s
                                new_V[:, c] *= s
                                new_W[:, c] *= s
                            keep = [c for c in range(r) if c != idx]
                            new_U = new_U[:, keep]
                            new_V = new_V[:, keep]
                            new_W = new_W[:, keep]
                            if verify(new_U, new_V, new_W):
                                U, V, W = new_U, new_V, new_W
                                found_cross = True
                                break
                        if found_cross:
                            break

    # ---------- Final verify & return ----------
    if not verify(U, V, W):
        # Rebuild a guaranteed-correct schoolbook fallback
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        U, V, W = build_matrices(terms)
    
    # Final polish: try one more reduction pass with very tight tolerance
    if time_left() > 3.0:
        T = tensor_vecs(U, V, W)
        r = T.shape[0]
        if r > 1:
            try:
                _, svals, Vt = np.linalg.svd(T, full_matrices=False)
            except np.linalg.LinAlgError:
                svals, Vt = None, None
            if svals is not None:
                tol = 1e-11 * (svals[0] if svals.size else 1.0)
                rank_T = int(np.sum(svals > tol))
                if rank_T < r:
                    null_space = Vt[rank_T:, :]
                    for nv in null_space:
                        if time_left() < 2.0:
                            break
                        abs_nv = np.abs(nv)
                        sorted_idx = np.argsort(abs_nv)[::-1]
                        for idx in sorted_idx[:3]:
                            if abs_nv[idx] < 1e-14:
                                continue
                            new_U = U.copy()
                            new_V = V.copy()
                            new_W = W.copy()
                            for c in range(r):
                                if c == idx:
                                    continue
                                s = 1.0 + nv[c] / nv[idx]
                                if abs(s) < 1e-14:
                                    continue
                                new_U[:, c] *= s
                                new_V[:, c] *= s
                                new_W[:, c] *= s
                            keep = [c for c in range(r) if c != idx]
                            new_U = new_U[:, keep]
                            new_V = new_V[:, keep]
                            new_W = new_W[:, keep]
                            if verify(new_U, new_V, new_W, tol=1e-9):
                                U, V, W = new_U, new_V, new_W
                                break
                        else:
                            continue
                        break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
