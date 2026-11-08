"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _all_weight4():
    """All nonzero vectors with exactly four +/-1 entries (norm 2)."""
    vecs = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for pos, s in zip(combo, signs):
                v[pos] = s
            vecs.append(v)
    return np.array(vecs, dtype=np.int8)


def _build_conflict(A):
    """conflict[i,j] iff A[i].A[j] > 2 (forbidden pair for a norm-2 kissing config)."""
    dot = A.astype(np.int32) @ A.astype(np.int32).T
    conflict = dot > 2
    np.fill_diagonal(conflict, False)
    return conflict


def _greedy(conflict, order, perturb=None):
    """Vectorized min-degree greedy independent set over candidate order.
    Repeatedly pick the remaining vertex with fewest conflicts to already-chosen
    vertices (tie broken by perturb, then index)."""
    n = conflict.shape[0]
    in_rem = np.zeros(n, dtype=bool)
    in_rem[order] = True
    to_chosen = np.zeros(n, dtype=np.int32)
    chosen = []
    c8 = conflict.astype(np.int32)
    while in_rem.any():
        key = to_chosen.astype(np.int64)
        if perturb is not None:
            key = key + perturb
        key = np.where(in_rem, key, n + 1)
        idx = int(np.argmin(key))
        chosen.append(idx)
        in_rem[idx] = False
        to_chosen = to_chosen + c8[idx]
    return chosen


def _try_add(conflict, chosen_set, n):
    """Add any remaining vertex that has no conflict with the current set (one pass)."""
    if not chosen_set:
        return chosen_set
    ch = np.fromiter(chosen_set, dtype=np.int64, count=len(chosen_set))
    conflict_to_chosen = conflict[:, ch].any(axis=1)
    for i in range(n):
        if i not in chosen_set and not conflict_to_chosen[i]:
            chosen_set.add(i)
    return chosen_set


def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    A = _all_weight4()
    conflict = _build_conflict(A)
    n = A.shape[0]

    deg = conflict.sum(axis=1)

    # deterministic baseline: low-degree-first
    order0 = list(np.argsort(deg, kind="stable"))
    best_set = set(_greedy(conflict, order0))
    best_set = _try_add(conflict, best_set, n)
    best_count = len(best_set)

    rng = np.random.default_rng(2024)

    while time.time() < deadline:
        perm = rng.permutation(n)
        # small random 0/1 perturbation to diversify tie-breaks across restarts
        perturb = (rng.random(n) < 0.5).astype(np.int64)
        s = set(_greedy(conflict, perm, perturb))
        s = _try_add(conflict, s, n)
        if len(s) > best_count:
            best_count = len(s)
            best_set = s

    return A[list(best_set)].astype(np.int64)
# EVOLVE-BLOCK-END
