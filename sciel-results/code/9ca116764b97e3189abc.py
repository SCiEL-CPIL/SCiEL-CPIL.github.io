"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    rng = np.random.default_rng(12345)

    # --- Build a large structured candidate pool from a lattice ---
    # Use the D_11 lattice: integer vectors with even coordinate sum,
    # and half-integer vectors (all coords in {±0.5, ±1.5}) with even sum.
    # We work in doubled coordinates to keep integers: D_11 doubled = {v in Z^11 : sum even} union {v in (2Z+1)^11 : sum even}.
    # We restrict to small norms to maximize packing density.

    cand_list = []

    # Integer part of D_11 (doubled): coords in {-3,-2,-1,0,1,2,3}, sum even
    vals_int = np.array([-3, -2, -1, 0, 1, 2, 3])
    # Enumerate by weight to keep count bounded
    # We'll do a bounded random sampling over the lattice ball instead of full enumeration.

    # Strategy: sample many random lattice vectors with bounded entries, dedup.
    seen = set()
    target_pool = 400000

    def add_v(v):
        # v is int array length D, must be nonzero
        if np.any(v != 0):
            key = v.tobytes()
            if key not in seen:
                seen.add(key)
                cand_list.append(v)

    # Integer D_11 (doubled): even sum
    while len(cand_list) < target_pool and time.time() < deadline - 60:
        w = int(rng.integers(2, 6))  # number of nonzero coords
        pos = rng.choice(D, size=w, replace=False)
        # random nonzero values from small set
        vals = rng.choice([-3, -2, -1, 1, 2, 3], size=w)
        v = np.zeros(D, dtype=np.int64)
        v[pos] = vals
        s = int(v.sum())
        if s % 2 == 0:
            add_v(v)

    # Half-integer D_11 (doubled): all coords odd, even sum
    while len(cand_list) < target_pool and time.time() < deadline - 60:
        # all coords in {-3,-1,1,3} (odd), even sum
        v = np.array(rng.choice([-3, -1, 1, 3], size=D), dtype=np.int64)
        s = int(v.sum())
        if s % 2 == 0:
            add_v(v)

    if len(cand_list) == 0:
        # fallback baseline
        vecs = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    cand = np.array(cand_list, dtype=np.int64)
    n_cand = len(cand)
    cand_norm_sq = np.sum(cand * cand, axis=1)

    # --- Greedy independent set, sorted by norm ---
    order = np.argsort(cand_norm_sq)
    accepted_idx = []
    accepted_norms_sq = []

    for idx in order:
        if time.time() > deadline - 30:
            break
        cv = cand[idx]
        cvn = cand_norm_sq[idx]
        if len(accepted_idx) == 0:
            accepted_idx.append(int(idx))
            accepted_norms_sq.append(int(cvn))
            continue
        acc = cand[accepted_idx]
        dots = acc @ cv
        accn = np.array(accepted_norms_sq, dtype=np.int64)
        min_norms = np.minimum(cvn, accn)
        if np.all(min_norms >= 2 * dots):
            accepted_idx.append(int(idx))
            accepted_norms_sq.append(int(cvn))

    if len(accepted_idx) == 0:
        vecs = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        vecs.append(v)
        return np.array(vecs, dtype=np.int64)

    # --- Local search: swap-based improvement ---
    # Maintain accepted set; try to swap out one accepted vector for a candidate
    # that is compatible with the rest, increasing count (we keep count same but
    # may enable more later). Actually to increase count, try adding a candidate
    # that is compatible with all but one accepted, then swap.

    acc_set = set(accepted_idx)
    acc_idx = np.array(accepted_idx, dtype=np.int64)
    acc_norms = np.array(accepted_norms_sq, dtype=np.int64)

    def is_compatible_with_all_but(cv, cvn, exclude):
        # check cv against all accepted except the one at position `exclude`
        # returns True if compatible with all others
        mask = np.ones(len(acc_idx), dtype=bool)
        mask[exclude] = False
        if not np.any(mask):
            return True
        sub = acc_idx[mask]
        subn = acc_norms[mask]
        dots = cand[sub] @ cv
        min_norms = np.minimum(cvn, cand_norm_sq[sub])
        return bool(np.all(min_norms >= 2 * dots))

    # Precompute for each candidate the count of accepted it conflicts with
    # conflict count = number of accepted a where min(cvn, an) < 2*dot
    # We want candidates with conflict count <= 1 to swap.

    # Vectorize: build matrix of dots between all candidates and accepted.
    # This may be large; do it in chunks.
    acc_mat = cand[acc_idx]            # (k, D)
    acc_norms_mat = cand_norm_sq[acc_idx]  # (k,)

    # For each candidate c, conflict with a_j if min(cand_norm_sq[c], acc_norms_mat[j]) < 2*dot(c, a_j)
    # Compute in chunks over candidates.
    conflict_counts = np.zeros(n_cand, dtype=np.int64)
    chunk = 2000
    for start in range(0, n_cand, chunk):
        end = min(start + chunk, n_cand)
        sub = cand[start:end]
        subn = cand_norm_sq[start:end]
        dots = sub @ acc_mat.T  # (chunk, k)
        minn = np.minimum(subn[:, None], acc_norms_mat[None, :])
        conflict = (minn < 2 * dots).astype(np.int64)
        conflict_counts[start:end] = conflict.sum(axis=1)
        if time.time() > deadline - 15:
            break

    # Candidates with conflict count 0 that are not already accepted -> can add directly
    # (greedy should have caught most, but order was by norm; some may now be addable)
    not_acc = np.ones(n_cand, dtype=bool)
    not_acc[acc_idx] = False
    addable = np.where((conflict_counts == 0) & not_acc)[0]

    # Add them (re-check each individually to be safe)
    for idx in addable:
        if time.time() > deadline - 10:
            break
        cv = cand[idx]
        cvn = cand_norm_sq[idx]
        dots = acc_mat @ cv
        minn = np.minimum(cvn, acc_norms_mat)
        if np.all(minn >= 2 * dots):
            acc_idx = np.append(acc_idx, int(idx))
            acc_norms_mat = np.append(acc_norms_mat, int(cvn))
            acc_mat = np.vstack([acc_mat, cv[None, :]])
            not_acc[int(idx)] = False

    # Swap: candidates with conflict count == 1 -> remove the conflicting accepted, add candidate
    # This keeps count same but can enable further additions. Do a few rounds.
    for round_ in range(3):
        if time.time() > deadline - 8:
            break
        # recompute conflict counts after additions
        acc_mat = cand[acc_idx]
        acc_norms_mat = cand_norm_sq[acc_idx]
        conflict_counts = np.zeros(n_cand, dtype=np.int64)
        for start in range(0, n_cand, chunk):
            end = min(start + chunk, n_cand)
            sub = cand[start:end]
            subn = cand_norm_sq[start:end]
            dots = sub @ acc_mat.T
            minn = np.minimum(subn[:, None], acc_norms_mat[None, :])
            conflict = (minn < 2 * dots).astype(np.int64)
            conflict_counts[start:end] = conflict.sum(axis=1)
            if time.time() > deadline - 5:
                break
        not_acc = np.ones(n_cand, dtype=bool)
        not_acc[acc_idx] = False
        swap_cands = np.where((conflict_counts == 1) & not_acc)[0]
        # For each, find which accepted conflicts, try swap, then re-add
        improved = False
        for idx in swap_cands:
            if time.time() > deadline - 5:
                break
            cv = cand[idx]
            cvn = cand_norm_sq[idx]
            dots = acc_mat @ cv
            minn = np.minimum(cvn, acc_norms_mat)
            bad = np.where(minn < 2 * dots)[0]
            if len(bad) != 1:
                continue
            b = int(bad[0])
            # remove accepted b, add idx
            new_acc_idx = np.delete(acc_idx, b)
            new_acc_idx = np.append(new_acc_idx, int(idx))
            # verify full compatibility of new set (cheap-ish)
            nm = cand[new_acc_idx]
            nn = cand_norm_sq[new_acc_idx]
            # pairwise check
            ok = True
            k = len(nm)
            # vectorized pairwise
            dotsm = nm @ nm.T
            nnm = nn[:, None]
            nnT = nn[None, :]
            minm = np.minimum(nnm, nnT)
            # diagonal is 0 distance to self, ignore
            np.fill_diagonal(dotsm, 0)
            if not np.all(minm >= 2 * dotsm):
                ok = False
            if ok:
                acc_idx = new_acc_idx
                acc_norms_mat = cand_norm_sq[acc_idx]
                acc_mat = nm
                not_acc = np.ones(n_cand, dtype=bool)
                not_acc[acc_idx] = False
                improved = True
                break  # recompute after a swap

    return cand[acc_idx]
# EVOLVE-BLOCK-END
