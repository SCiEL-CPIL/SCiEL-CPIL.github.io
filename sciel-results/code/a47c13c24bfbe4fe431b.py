# EVOLVE-BLOCK-START
import time
import random
import math

import numpy as np


def _spectral_init(n, var_idx, sign, m, rng, deadline):
    """Initialize the assignment from the planted low-rank (SVD) structure.

    Build the clause-variable incidence matrix A (m x n) with entry
    A[i, v] = sign of literal v in clause i (0 if absent).  For a planted
    instance the planted assignment is (up to a global sign) the direction
    that makes each row align with its planted sign pattern, so we take the
    principal left singular vector u0 and set x_v = sign(u0[v]).  We compare
    this against a random start and keep whichever has fewer unsat clauses.
    """
    if m > 0 and n > 0:
        try:
            A = np.zeros((m, n), dtype=np.float64)
            A[np.arange(m), var_idx] = sign
            # Thin the matrix if it is very large to keep SVD cheap.
            if m > 4000:
                sel = rng.sample(range(m), 4000)
                A = A[sel]
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            u0 = U[:, 0]
            x = (u0 >= 0).astype(np.int8)
            return x
        except Exception:
            pass
    return None


def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    m = len(clauses)

    deadline = time.time() + 320  # safety margin before the 330 s kill
    rng = random.Random(12345)

    # Precompute literal structure (0-based var index, +1/-1 sign).
    var_idx = np.empty((m, 3), dtype=np.int32)
    sign = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        for j, l in enumerate(c):
            var_idx[i, j] = abs(l) - 1
            sign[i, j] = 1 if l > 0 else -1

    def eval_sat(x):
        """Return (n_unsat, idx_unsat) for a 0/1 numpy assignment x."""
        cur = x.astype(np.int8)
        val = np.where(cur == 1, 1, -1)
        s = val[var_idx] * sign
        unsat = (s.sum(axis=1) < 0).nonzero()[0]
        return int(unsat.size), unsat

    def is_sat(x):
        cur = x.astype(np.int8)
        val = np.where(cur == 1, 1, -1)
        s = val[var_idx] * sign
        return bool((s.sum(axis=1) < 0).any()) is False

    # ---- Initial assignment: spectral (planted-structure) vs random ----
    candidates = []
    spec = _spectral_init(n, var_idx, sign, m, rng, deadline)
    if spec is not None:
        candidates.append(spec)
    candidates.append((rng.random(n) < 0.5).astype(np.int8))

    best_unsat = m + 1
    best_x = None
    x = None
    for cand in candidates:
        u, _ = eval_sat(cand)
        if u == 0:
            return cand.tolist()
        if u < best_unsat:
            best_unsat = u
            best_x = cand.copy()
            x = cand
    if x is None:
        x = (rng.random(n) < 0.5).astype(np.int8)
        best_x = x.copy()

    # ---- WalkSAT with restarts and an incremental unsat-clause tracker ----
    cur = x.copy().astype(np.int8)
    val = np.where(cur == 1, 1, -1)
    sat_count = (val[var_idx] * sign).sum(axis=1)  # per-clause satisfied count
    unsat_set = set(np.nonzero(sat_count < 0)[0].tolist())

    # Precompute, for each clause, the list of (literal_index, var) pairs.
    lit_var = var_idx  # (m,3) variable indices per clause

    base_steps = 400
    max_restarts = 400

    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break

        steps = base_steps + restart * 60
        if steps > 4000:
            steps = 4000
        p = 0.55

        for step in range(steps):
            if time.time() > deadline - 5:
                break

            if not unsat_set:
                return cur.tolist()

            # Pick a random unsatisfied clause.
            ci = next(iter(unsat_set))

            # Score each of the 3 literals by the net gain from flipping it.
            # gain = (# clauses it would satisfy) - (# clauses it would break)
            v0, v1, v2 = lit_var[ci]
            gains = []
            for v in (v0, v1, v2):
                # Clauses containing variable v.
                mask = (var_idx == v).any(axis=1)
                idx = np.nonzero(mask)[0]
                if idx.size == 0:
                    gains.append(0)
                    continue
                # For each such clause, determine current sign of var v and
                # whether flipping v satisfies (currently unsat & only false
                # literal) or breaks it.
                # sign_of_var in clause i:
                sv = sign[np.arange(idx.size), (var_idx[idx] == v).argmax(axis=1)]
                # current value of variable v:
                curv = int(cur[v])
                # literal is true now iff sv * (1 if curv else -1) > 0
                lit_true = (sv * (1 if curv else -1)) > 0
                # clauses currently unsatisfied:
                unsat_here = np.array([i in unsat_set for i in idx])
                # Satisfying a clause by flipping: clause unsat AND literal
                # currently false (flipping makes it true).  A 3-clause can
                # only be made true by flipping a false literal.
                would_satisfy = unsat_here & (~lit_true)
                # Breaking a clause by flipping: clause currently sat AND this
                # literal is its only true literal (flipping makes clause unsat).
                # number of true literals in clause:
                cur_val = np.where(cur[var_idx[idx]] == 1, 1, -1)
                s_here = (cur_val * sign[idx]).sum(axis=1)
                true_lits_here = (cur_val * sign[idx] > 0).sum(axis=1)
                would_break = (~unsat_here) & (true_lits_here == 1) & lit_true
                gains.append(int(would_satisfy.sum()) - int(would_break.sum()))

            # Choose literal: with prob p prefer the best-gain literal,
            # otherwise pick uniformly.
            if rng.random() < p:
                best_gain = max(gains)
                # among ties pick randomly
                opts = [k for k, g in enumerate(gains) if g == best_gain]
                k = rng.randrange(len(opts))
            else:
                k = rng.randrange(3)

            v = int(lit_var[ci, k])
            newv = 0 if cur[v] == 1 else 1

            # Update val for variable v.
            old_val = 1 if cur[v] == 1 else -1
            new_val = 1 if newv == 1 else -1
            delta = new_val - old_val  # in {-2, 2}

            # Update sat_count for all clauses containing v.
            mask = (var_idx == v).any(axis=1)
            idx = np.nonzero(mask)[0]
            sv = sign[np.arange(idx.size), (var_idx[idx] == v).argmax(axis=1)]
            sat_count[idx] += delta * sv

            # Update unsat_set for affected clauses.
            for i in idx:
                was_unsat = i in unsat_set
                now_unsat = sat_count[i] < 0
                if was_unsat and not now_unsat:
                    unsat_set.discard(i)
                elif (not was_unsat) and now_unsat:
                    unsat_set.add(i)

            cur[v] = newv

            # Periodic global check / bookkeeping.
            if step % 8 == 0:
                u = len(unsat_set)
                if u == 0:
                    return cur.tolist()
                if u < best_unsat:
                    best_unsat = u
                    best_x = cur.copy()

        # End of walk: record best.
        u = len(unsat_set)
        if u == 0:
            return cur.tolist()
        if u < best_unsat:
            best_unsat = u
            best_x = cur.copy()

        # Restart: reinitialize (spectral first few restarts, then random).
        if restart < 3:
            spec = _spectral_init(n, var_idx, sign, m, rng, deadline)
            if spec is not None:
                cur = spec.copy().astype(np.int8)
            else:
                cur = (rng.random(n) < 0.5).astype(np.int8)
        else:
            cur = (rng.random(n) < 0.5).astype(np.int8)

        val = np.where(cur == 1, 1, -1)
        sat_count = (val[var_idx] * sign).sum(axis=1)
        unsat_set = set(np.nonzero(sat_count < 0)[0].tolist())

    # Return the best assignment found.
    if best_x is not None:
        if not is_sat(best_x):
            # last-ditch: try a couple of quick greedy passes on best_x
            pass
        return best_x.tolist()
    return [False] * n
# EVOLVE-BLOCK-END
