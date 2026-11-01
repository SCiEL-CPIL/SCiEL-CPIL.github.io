import time
import random
import itertools
import numpy as np

# ---------------------------------------------------------------------------
# Precomputation cache (independent of the seed / restarts).
# For the task parameters v=19, k=7, t=4:
#   n_ksubs = C(19,7) = 50388 candidate blocks
#   n_tsubs = C(19,4) = 3927 total 4-subsets
# ---------------------------------------------------------------------------
_CACHE = {}


def _get_cache(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]

    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # For each candidate block, the list of t-subset indices it covers.
    block_tsubs = []
    for ks in all_ksubs:
        block_tsubs.append([tsub_to_idx[sub] for sub in itertools.combinations(ks, t)])
    block_tsubs = np.asarray(block_tsubs, dtype=np.int32)
    block_tsubs_list = [list(row) for row in block_tsubs]

    # For each t-subset, the list of candidate blocks that cover it (reverse index).
    # Used by the swap polish to find candidates that cover a specific rare t-subset.
    tsub_blocks = [[] for _ in range(n_tsubs)]
    for b, idxs in enumerate(block_tsubs_list):
        for s in idxs:
            tsub_blocks[s].append(b)
    tsub_blocks = [np.asarray(lst, dtype=np.int32) for lst in tsub_blocks]

    cache = {
        "all_ksubs": all_ksubs,
        "block_tsubs": block_tsubs,
        "block_tsubs_list": block_tsubs_list,
        "tsub_blocks": tsub_blocks,
        "n_ksubs": n_ksubs,
        "n_tsubs": n_tsubs,
    }
    _CACHE[key] = cache
    return cache


def construct(v, k, t):
    start = time.time()
    deadline = start + 140.0  # explicit guard, safety margin under 150s wall clock

    cache = _get_cache(v, k, t)
    all_ksubs = cache["all_ksubs"]
    block_tsubs = cache["block_tsubs"]          # shape (n_ksubs, k choose t)
    block_tsubs_list = cache["block_tsubs_list"]
    tsub_blocks = cache["tsub_blocks"]
    n_ksubs = cache["n_ksubs"]
    n_tsubs = cache["n_tsubs"]

    def time_left():
        return deadline - time.time()

    # ------------------------------------------------------------------
    # Greedy selection with a (possibly reweighted) coverage objective.
    # `w` is a per-t-subset weight vector.  The objective of a candidate
    # block is sum over its t-subsets of w[s] * (count[s] < 1), i.e. it
    # only rewards t-subsets that are still uncovered, scaled by w[s].
    # This is the subgradient / Lagrangian surrogate of max coverage.
    # ------------------------------------------------------------------
    def greedy_select(w, max_blocks):
        count = np.zeros(n_tsubs, dtype=np.int32)
        uncovered = n_tsubs
        chosen = []
        for _ in range(max_blocks):
            if uncovered == 0 or time_left() < 1.0:
                break
            # reward[b] = sum_{s in block b} w[s] * (count[s] == 0)
            reward = np.sum(w[block_tsubs] * (count[block_tsubs] == 0), axis=1)
            b = int(np.argmax(reward))
            if reward[b] <= 0:
                break
            for s in block_tsubs_list[b]:
                if count[s] == 0:
                    count[s] = 1
                    uncovered -= 1
            chosen.append(b)
        return chosen, uncovered

    # ------------------------------------------------------------------
    # Lagrangian relaxation: dual weights w updated by subgradient descent.
    # Each greedy pass is a "primal" solve; the subgradient of the
    # Lagrangian with respect to the uncovered indicators is
    #   g[s] = w[s] * (count[s] < 1) - lambda   (per selected block)
    # which pushes w up for t-subsets that keep staying uncovered and
    # down for well-covered ones, dynamically balancing coverage density
    # against block count.
    # ------------------------------------------------------------------
    def lagrangian_greedy(seed, max_blocks):
        rng = random.Random(seed)
        w = np.ones(n_tsubs, dtype=np.float64)
        lam = 0.0
        best = None
        best_len = 10 ** 9
        for it in range(60):
            if time_left() < 1.0:
                break
            chosen, unc = greedy_select(w, max_blocks)
            if unc == 0 and len(chosen) < best_len:
                best = chosen
                best_len = len(chosen)
            # subgradient step
            count = np.zeros(n_tsubs, dtype=np.int32)
            for b in chosen:
                for s in block_tsubs_list[b]:
                    count[s] += 1
            g = w * (count == 0) - lam
            norm = float(np.linalg.norm(g))
            if norm > 1e-12:
                step = 2.0 / (it + 1)
                w = np.maximum(w + step * g, 0.05)
                lam = max(0.0, lam + step * (len(chosen) - 100.0))
            w = w / (float(np.sum(w)) / n_tsubs)  # keep mean weight ~ 1
        return best

    # ------------------------------------------------------------------
    # Redundancy removal + 1-for-1 swap polish.  All operations are O(35)
    # per candidate using the per-t-subset count array.
    # ------------------------------------------------------------------
    def polish(chosen, max_blocks):
        chosen = list(chosen)
        count = np.zeros(n_tsubs, dtype=np.int32)
        for b in chosen:
            for s in block_tsubs_list[b]:
                count[s] += 1
        in_set = set(chosen)
        changed = True
        while changed and time_left() > 1.0:
            changed = False
            # (a) remove redundant blocks
            new_list = []
            for b in chosen:
                for s in block_tsubs_list[b]:
                    count[s] -= 1
                    if count[s] == 0:
                        break
                else:
                    # b covered no unique t-subset -> redundant, drop it
                    in_set.discard(b)
                    changed = True
                    continue
                for s in block_tsubs_list[b]:
                    count[s] += 1
                new_list.append(b)
            if changed:
                chosen = new_list
                continue
            # (b) try 1-for-1 swaps that reduce the count
            for i, b_out in enumerate(chosen):
                if time_left() < 1.0:
                    return chosen
                for s in block_tsubs_list[b_out]:
                    count[s] -= 1
                for b_in in tsub_blocks[s]:
                    if b_in in in_set:
                        continue
                    # net change = (# t-subsets newly covered by b_in)
                    gain = 0
                    for s2 in block_tsubs_list[b_in]:
                        if count[s2] == 0:
                            gain += 1
                    if gain >= 1:  # reduces total uncovered (or keeps it, but here strictly improves)
                        # commit the swap
                        for s2 in block_tsubs_list[b_in]:
                            count[s2] += 1
                        for s2 in block_tsubs_list[b_out]:
                            count[s2] += 1  # restore b_out's t-subsets (now uncovered -> 0)
                        in_set.discard(b_out)
                        in_set.add(b_in)
                        chosen[i] = b_in
                        changed = True
                        break
                for s in block_tsubs_list[b_out]:
                    count[s] += 1  # restore
                if changed:
                    break
        return chosen

    # ------------------------------------------------------------------
    # Build an initial valid covering (plain max-coverage greedy).
    # ------------------------------------------------------------------
    init, unc = greedy_select(np.ones(n_tsubs, dtype=np.float64), 400)
    if unc > 0:
        # extremely unlikely; top up with pure greedy
        init, unc = greedy_select(np.ones(n_tsubs, dtype=np.float64), 2000)
    best = polish(init, 400)
    best_len = len(best)

    # ------------------------------------------------------------------
    # Multi-restart with randomized Lagrangian dual weights.
    # ------------------------------------------------------------------
    n_restarts = 12
    for r in range(n_restarts):
        if time_left() < 4.0:
            break
        rng = random.Random(1000 + r)
        w0 = np.exp(rng.uniform(-0.8, 0.8, size=n_tsubs))
        w0 = w0 / (float(np.sum(w0)) / n_tsubs)
        res = lagrangian_greedy(1000 + r, 400)
        if res is not None:
            cand = polish(res, 400)
            if len(cand) < best_len:
                best = cand
                best_len = len(cand)

    # Final safety polish on the current best.
    best = polish(best, 400)

    return [list(all_ksubs[b]) for b in best]
