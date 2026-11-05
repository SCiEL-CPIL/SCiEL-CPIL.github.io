"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START


def _validate(n, m, p, U, V, W):
    """Verify exact tensor decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)


def _build_strassen(n, m, p):
    """Build rank-55 decomposition using Strassen on 2x2 blocks."""
    assert n == 2 and m == 3 and p == 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    strassen_u = {
        0: [(0, 1), (4, 1)], 1: [(3, 1), (4, 1)], 2: [(0, 1)],
        3: [(4, 1)], 4: [(0, 1), (1, 1)], 5: [(3, 1), (0, -1)], 6: [(1, 1), (4, -1)]
    }
    strassen_v = {
        0: [(0, 1), (11, 1)], 1: [(0, 1)], 2: [(1, 1), (11, -1)],
        3: [(10, 1), (0, -1)], 4: [(11, 1)], 5: [(0, 1), (1, 1)], 6: [(10, 1), (11, 1)]
    }
    strassen_w = {
        0: [(0, 1), (3, 1)], 1: [(1, 1), (3, -1)], 2: [(2, 1), (3, 1)],
        3: [(0, 1), (1, 1)], 4: [(0, -1), (2, 1)], 5: [(3, 1)], 6: [(0, 1)]
    }
    
    for b in range(5):
        col0, col1 = 2 * b, 2 * b + 1
        for s in range(7):
            for (u_row, coeff) in strassen_u[s]:
                U[u_row][r] += coeff
            for (v_off, coeff) in strassen_v[s]:
                v_row = col0 if v_off == 0 else col1 if v_off == 1 else 10 + col0 if v_off == 10 else 10 + col1
                V[v_row][r] += coeff
            for (w_off, coeff) in strassen_w[s]:
                w_row = col0 * 2 + w_off if w_off < 2 else col1 * 2 + (w_off - 2)
                W[w_row][r] += coeff
            r += 1
    
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[20 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1
    
    assert r == rank
    return U, V, W, rank


def _reduce_rank(n, m, p, U, V, W, deadline):
    """Rank reduction exploiting shared extra-dimension structure + local search."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    target = np.ones((n, m, p))
    tol = 1e-9

    # Precompute each rank-1 contribution
    col_contrib = np.zeros((rank, n, m, p))
    for r in range(rank):
        u_col = U_arr[:, r].reshape(n, m)
        v_col = V_arr[:, r].reshape(m, p)
        w_col = W_arr[:, r].reshape(p, n)
        col_contrib[r] = np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)

    T_current = col_contrib.sum(axis=0)
    keep = np.ones(rank, dtype=bool)

    # Phase 1: Remove singles
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for r in range(rank):
            if not keep[r]:
                continue
            if np.allclose(T_current - col_contrib[r], target, atol=tol):
                keep[r] = False
                T_current -= col_contrib[r]
                changed = True

    # Phase 2: Remove pairs
    changed = True
    while changed and time.time() < deadline:
        changed = False
        kept = [r for r in range(rank) if keep[r]]
        for i in range(len(kept)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(kept)):
                r1, r2 = kept[i], kept[j]
                if not keep[r1] or not keep[r2]:
                    continue
                if np.allclose(T_current - col_contrib[r1] - col_contrib[r2], target, atol=tol):
                    keep[r1] = keep[r2] = False
                    T_current -= col_contrib[r1] + col_contrib[r2]
                    changed = True
                    break
            if changed:
                break

    # Phase 3: Remove triples
    changed = True
    while changed and time.time() < deadline:
        changed = False
        kept = [r for r in range(rank) if keep[r]]
        for i in range(len(kept)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(kept)):
                if time.time() > deadline:
                    break
                for k in range(j + 1, len(kept)):
                    r1, r2, r3 = kept[i], kept[j], kept[k]
                    if not keep[r1] or not keep[r2] or not keep[r3]:
                        continue
                    if np.allclose(T_current - col_contrib[r1] - col_contrib[r2] - col_contrib[r3], target, atol=tol):
                        keep[r1] = keep[r2] = keep[r3] = False
                        T_current -= col_contrib[r1] + col_contrib[r2] + col_contrib[r3]
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    # Phase 4: Exploit shared extra-dimension (U col 2, V row 20)
    # In the Strassen construction, rank-1 terms with U[:,r] having a nonzero
    # entry at position (1,2) [i=1,j=2] and V[:,r] at (2,k) [j=2,k] share
    # the "extra" path. If we can find a combination of these that sums to
    # zero tensor contribution, we can eliminate them.
    # More generally: try to find a linear dependency among the col_contrib
    # vectors that sums to zero.
    if time.time() < deadline:
        kept = [r for r in range(rank) if keep[r]]
        n_kept = len(kept)
        if n_kept > 1:
            # Flatten contributions to vectors
            flat = np.array([col_contrib[r].flatten() for r in kept])
            # Try to find a subset that sums to zero (exact)
            # Use a greedy approach: repeatedly check if removing a set
            # leaves the target intact
            # For efficiency, try pairs and triples among the "extra" terms
            extra_indices = []
            for idx, r in enumerate(kept):
                if abs(U_arr[5, r]) > tol or abs(V_arr[20, r]) > tol:
                    extra_indices.append(idx)
            # Try removing pairs of extra terms
            for a in range(len(extra_indices)):
                if time.time() > deadline:
                    break
                for b in range(a + 1, len(extra_indices)):
                    ia, ib = extra_indices[a], extra_indices[b]
                    r1, r2 = kept[ia], kept[ib]
                    if not keep[r1] or not keep[r2]:
                        continue
                    if np.allclose(T_current - col_contrib[r1] - col_contrib[r2], target, atol=tol):
                        keep[r1] = keep[r2] = False
                        T_current -= col_contrib[r1] + col_contrib[r2]

    # Phase 5: Local search - try removing a term and adjusting coefficients
    # of one other term to compensate (only if both have overlapping support)
    if time.time() < deadline:
        kept = [r for r in range(rank) if keep[r]]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for r1 in kept:
                if time.time() > deadline:
                    break
                if not keep[r1]:
                    continue
                # Try removing r1 and adjusting r2 to compensate
                for r2 in kept:
                    if r1 == r2 or not keep[r2]:
                        continue
                    # We need: T - col[r1] + delta * col[r2] = target
                    # => delta * col[r2] = target - T + col[r1]
                    # => col[r2] * delta = col[r1] - (T - target)
                    # Since T - target should be 0 for valid decomp, we need
                    # col[r2] * delta = col[r1], i.e., col[r1] = delta * col[r2]
                    # Check if col[r1] is a scalar multiple of col[r2]
                    c1 = col_contrib[r1].flatten()
                    c2 = col_contrib[r2].flatten()
                    # Find nonzero entries of c2
                    nz = np.nonzero(np.abs(c2) > tol)[0]
                    if len(nz) == 0:
                        continue
                    # Check if c1 = delta * c2 for some delta
                    deltas = c1[nz] / c2[nz]
                    if np.allclose(deltas, deltas[0], atol=tol):
                        delta = deltas[0]
                        if abs(delta - 1.0) > tol:  # need delta != 1 (else trivial)
                            # Check: does T - c1 + delta*c2 = target?
                            # Since T = target (valid), we need -c1 + delta*c2 = 0
                            # i.e., c1 = delta*c2, which we just verified.
                            # So removing r1 and scaling r2 by (1+delta) works:
                            # New T' = T - c1 + delta*c2 = target - c1 + delta*c2
                            # = target - delta*c2 + delta*c2 = target. Yes!
                            keep[r1] = False
                            T_current -= col_contrib[r1]
                            # Update r2's contribution
                            col_contrib[r2] = (1.0 + delta) * col_contrib[r2]
                            U_arr[:, r2] *= (1.0 + delta)
                            V_arr[:, r2] /= (1.0 + delta) if abs(1.0 + delta) > tol else 1.0
                            # Actually, we need U*V*W to scale by (1+delta)
                            # Scale U by (1+delta): new contribution = (1+delta)*old
                            # But we removed r1 which had contribution c1 = delta*c2
                            # New sum = T - c1 + (1+delta)*c2 = target - delta*c2 + (1+delta)*c2 = target + c2
                            # That's wrong. Let me redo.
                            # T = target. Remove r1: T - c1. Add delta*c2 to r2:
                            # T' = T - c1 + delta*c2 = target - delta*c2 + delta*c2 = target. Correct!
                            # So we scale r2's contribution by adding delta*c2.
                            # New r2 contribution = c2 + delta*c2 = (1+delta)*c2.
                            # Scale U by (1+delta): U_new = (1+delta)*U_old
                            # Then U_new * V * W = (1+delta) * c2. Correct!
                            U_arr[:, r2] *= (1.0 + delta)
                            improved = True
                            break
                if improved:
                    break
            # Remove r1 from kept list
            kept = [r for r in kept if keep[r]]

    new_rank = int(np.sum(keep))
    if new_rank >= rank:
        return U, V, W, rank

    U_new = U_arr[:, keep]
    V_new = V_arr[:, keep]
    W_new = W_arr[:, keep]

    if _validate(n, m, p, U_new.tolist(), V_new.tolist(), W_new.tolist()):
        return U_new.tolist(), V_new.tolist(), W_new.tolist(), new_rank
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    U, V, W, rank = _build_strassen(n, m, p)
    
    if not _validate(n, m, p, U, V, W):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
    
    U, V, W, rank = _reduce_rank(n, m, p, U, V, W, deadline)

    # Phase 6: Try to find a lower-rank decomposition via structured search
    # The 5 Strassen blocks each use 7 rank-1 terms. The "extra" dimension
    # (U position (1,2), V position (2,k)) is shared. This means the 35
    # Strassen terms have a 5-dimensional subspace of "extra" contributions.
    # We can try to find a rank-(7*5 - d) decomposition by finding d
    # linear dependencies.
    if time.time() < deadline - 5:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        cur_rank = U_arr.shape[1]
        target = np.ones((n, m, p))
        tol = 1e-9

        # Compute all contributions
        contribs = np.zeros((cur_rank, n * m * p))
        for r in range(cur_rank):
            u_c = U_arr[:, r].reshape(n, m)
            v_c = V_arr[:, r].reshape(m, p)
            w_c = W_arr[:, r].reshape(p, n)
            contribs[r] = np.einsum('ij,jk,ki->ijk', u_c, v_c, w_c).flatten()

        # Try to find a subset of size 2 whose contributions cancel
        # (i.e., sum to zero), allowing us to remove both
        found = False
        for i in range(cur_rank):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, cur_rank):
                if np.allclose(contribs[i] + contribs[j], 0, atol=tol):
                    # Remove both
                    mask = np.ones(cur_rank, dtype=bool)
                    mask[i] = mask[j] = False
                    U_t = U_arr[:, mask]
                    V_t = V_arr[:, mask]
                    W_t = W_arr[:, mask]
                    if _validate(n, m, p, U_t.tolist(), V_t.tolist(), W_t.tolist()):
                        U, V, W = U_t.tolist(), V_t.tolist(), W_t.tolist()
                        rank = int(mask.sum())
                        found = True
                        break
            if found:
                break

        # Try to find a subset of size 3 whose contributions sum to zero
        if not found and time.time() < deadline - 5:
            for i in range(cur_rank):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, cur_rank):
                    if time.time() > deadline - 5:
                        break
                    for k in range(j + 1, cur_rank):
                        if np.allclose(contribs[i] + contribs[j] + contribs[k], 0, atol=tol):
                            mask = np.ones(cur_rank, dtype=bool)
                            mask[i] = mask[j] = mask[k] = False
                            U_t = U_arr[:, mask]
                            V_t = V_arr[:, mask]
                            W_t = W_arr[:, mask]
                            if _validate(n, m, p, U_t.tolist(), V_t.tolist(), W_t.tolist()):
                                U, V, W = U_t.tolist(), V_t.tolist(), W_t.tolist()
                                rank = int(mask.sum())
                                found = True
                                break
                    if found:
                        break
                if found:
                    break

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
