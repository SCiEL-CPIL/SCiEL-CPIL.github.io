"""Covering design C(v,k,t) via LP-relaxation dual + randomized rounding + subgradient polish.

Strategy family (kept): greedy / local-search covering, but with a materially different
inner engine:
  * Represent coverage as a 0/1 vector over all t-subsets (N = C(v,t)).
  * Smooth surrogate objective: soft-min over uncovered t-sets (log-sum-exp), a smooth
    surrogate for the nonsmooth 'min coverage count' objective.
  * Dual variables (prices) for each t-subset, updated by subgradient (the gradient of
    the soft surrogate). These prices drive a randomized rounding pass (randomized
    thresholding guided by the dual) that breaks symmetries.
  * Polish with redundant-block pruning.
"""
import itertools
import random
import time
import math
import sys
from collections import defaultdict

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    _HAVE_NUMPY = False


def _all_tsubsets(v, t):
    return list(itertools.combinations(range(v), t))


def _build_index(v, k, t):
    """Return (tsets, tset_index, block_to_tset_indices, n_tsets)."""
    tsets = _all_tsubsets(v, t)
    tset_index = {ts: i for i, ts in enumerate(tsets)}
    n_tsets = len(tsets)
    # For each k-subset, precompute the list of t-subset indices it covers.
    block_to_tset_indices = {}
    for blk in itertools.combinations(range(v), k):
        lst = []
        for ts in itertools.combinations(blk, t):
            lst.append(tset_index[ts])
        block_to_tset_indices[blk] = lst
    return tsets, tset_index, block_to_tset_indices, n_tsets


def _subgradient_prices(tsets, block_to_tset_indices, n_tsets, n_blocks, deadline, rng,
                        max_iters=60, step_scale=1.0):
    """Run subgradient on the LP relaxation of the set cover (min sum y s.t. coverage >= 1).

    Returns dual price vector u (len n_tsets), one per t-subset.
    Uses the soft surrogate: f = logsumexp(-u_j) over uncovered, gradient = softmax(-u).
    """
    if n_tsets == 0:
        return [0.0] * 0
    # Initialize prices small positive.
    u = [0.1] * n_tsets
    # For speed, keep a running coverage count per t-subset from the y variables.
    # y_b = 1 if block b is selected. We start with all y=0.
    # Actually for LP dual subgradient we maintain y (primal) and update u (dual).
    # Standard: u_j <- max(0, u_j - alpha*(sum_b A_jb y_b - 1))  (dual ascent on max)
    # Here we do primal-dual: start y=0, coverage=0, so each t-subset 'deficit'=1.
    # Update u_j += alpha * deficit_j  (raise price where uncovered).
    # Then set y_b = 1 if sum_j A_jb u_j > threshold (rounding via dual).
    # We only need the final u for the rounding pass.
    cov = [0] * n_tsets  # coverage count
    # Precompute for each block the list of t-indices (already have).
    # Iterate: update u, then recompute y from u (threshold), recompute cov.
    for it in range(max_iters):
        if time.time() > deadline:
            break
        # deficit = max(0, 1 - cov_j)
        # step
        # compute gradient of soft surrogate: g_j = exp(-u_j)/Z over uncovered-ish.
        # Simplify: use deficit as subgradient.
        max_neg = min(u)  # for numeric stability in softmax
        exps = []
        for j in range(n_tsets):
            exps.append(math.exp(-(u[j] - max_neg)))
        Z = sum(exps)
        if Z <= 0:
            Z = 1e-12
        alpha = step_scale / max(1.0, math.sqrt(it + 1))
        for j in range(n_tsets):
            deficit = max(0.0, 1.0 - cov[j])
            # combine: raise price by alpha*(deficit + small softmax weight)
            u[j] = max(0.0, u[j] + alpha * (deficit + 0.01 * exps[j] / Z))
        # Recompute y from u: select block if its dual value is high.
        # dual value of block b = sum_{j in block} u_j
        # threshold: pick blocks with top dual values until coverage improves.
        # To keep it cheap, recompute cov from a greedy selection driven by u.
        # We'll do: sort blocks by dual value desc, take those with dual > tau.
        # But that's O(B log B). Instead, do a pass: select block if its sum u > current best uncovered need.
        # Simpler: recompute cov by selecting all blocks whose dual value > 0.5 * max_dual.
        duals = []
        for b, lst in block_to_tset_indices.items():
            s = 0.0
            for j in lst:
                s += u[j]
            duals.append((s, b))
        if not duals:
            break
        max_dual = max(d for d, _ in duals)
        if max_dual <= 0:
            break
        tau = 0.4 * max_dual
        cov = [0] * n_tsets
        for d, b in duals:
            if d > tau:
                for j in block_to_tset_indices[b]:
                    cov[j] += 1
    return u


def _rounding_pass(tsets, tset_index, block_to_tset_indices, n_tsets, u, deadline, rng,
                   max_blocks=None):
    """Randomized rounding guided by dual prices u.

    Repeatedly pick a random uncovered t-subset j, then among k-sets containing it,
    pick one with probability proportional to exp(sum_{j' in block} u_{j'}).
    This breaks symmetries and tends to pick high-price (hard) t-sets' blocks.
    """
    uncovered = set(range(n_tsets))
    blocks = []
    seen = set()
    # Precompute for each t-subset, the list of blocks containing it.
    tset_to_blocks = defaultdict(list)
    for b, lst in block_to_tset_indices.items():
        for j in lst:
            tset_to_blocks[j].append(b)
    # Precompute block dual values.
    block_dual = {}
    for b, lst in block_to_tset_indices.items():
        s = 0.0
        for j in lst:
            s += u[j]
        block_dual[b] = s
    while uncovered:
        if time.time() > deadline:
            break
        if max_blocks is not None and len(blocks) >= max_blocks:
            break
        # Pick a random uncovered t-subset.
        j = rng.choice(list(uncovered))
        cands = tset_to_blocks[j]
        if not cands:
            # Should not happen; skip.
            uncovered.discard(j)
            continue
        # Weighted selection by exp(block_dual).
        weights = []
        maxd = max(block_dual[b] for b in cands)
        for b in cands:
            weights.append(math.exp(block_dual[b] - maxd))
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        chosen = cands[0]
        for b, w in zip(cands, weights):
            acc += w
            if r <= acc:
                chosen = b
                break
        # Add block if new.
        if chosen in seen:
            # Still, remove the t-sets it covers from uncovered.
            for jj in block_to_tset_indices[chosen]:
                uncovered.discard(jj)
            continue
        seen.add(chosen)
        blocks.append(list(chosen))
        for jj in block_to_tset_indices[chosen]:
            uncovered.discard(jj)
    return blocks


def _greedy_fallback(v, k, t, deadline, rng):
    """Plain greedy to guarantee a valid covering if rounding fails."""
    tsets = _all_tsubsets(v, t)
    uncovered = set(tsets)
    blocks = []
    while uncovered:
        if time.time() > deadline:
            break
        start = min(uncovered)
        block = list(start)
        while len(block) < k:
            best_gain = -1
            best_point = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    cand = tuple(sorted(rest + (x,)))
                    if cand in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point == -1:
                for x in range(v):
                    if x not in block:
                        best_point = x
                        break
            block.append(best_point)
        block.sort()
        blocks.append(block)
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    return blocks


def _prune_blocks(blocks, v, k, t, deadline):
    """Remove redundant blocks (those whose every t-subset is covered by others)."""
    tsets = _all_tsubsets(v, t)
    cover_count = defaultdict(int)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            cover_count[ts] += 1
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        i = 0
        while i < len(blocks):
            redundant = True
            for ts in itertools.combinations(blocks[i], t):
                if cover_count[ts] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts in itertools.combinations(blocks[i], t):
                    cover_count[ts] -= 1
                blocks.pop(i)
                changed = True
            else:
                i += 1
    return blocks


def _is_valid(blocks, v, k, t):
    """Check every t-subset is covered by at least one block."""
    tsets = _all_tsubsets(v, t)
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    return all(ts in covered for ts in tsets)


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s kill
    rng = random.Random(0)

    # Build index structures.
    tsets, tset_index, block_to_tset_indices, n_tsets = _build_index(v, k, t)
    n_blocks = len(block_to_tset_indices)

    best_blocks = None
    best_len = float('inf')

    # --- Phase 1: subgradient LP-relaxation to get dual prices, then randomized rounding ---
    # Budget ~60% of time for the dual+rounding, rest for polish/fallback.
    phase1_deadline = time.time() + 66.0
    u = _subgradient_prices(tsets, block_to_tset_indices, n_tsets, n_blocks,
                            deadline=phase1_deadline, rng=rng, max_iters=80, step_scale=1.0)

    # Multiple randomized rounding restarts (different seeds) guided by the same dual.
    for seed in range(6):
        if time.time() > phase1_deadline:
            break
        r = random.Random(seed * 1000 + 7)
        blocks = _rounding_pass(tsets, tset_index, block_to_tset_indices, n_tsets,
                                u, deadline=phase1_deadline, rng=r)
        # Ensure validity; if invalid, top up with greedy.
        if not _is_valid(blocks, v, k, t):
            # Top up: find uncovered t-sets and greedily add blocks.
            covered = set()
            for block in blocks:
                for ts in itertools.combinations(block, t):
                    covered.add(ts)
            uncovered = set(tsets) - covered
            while uncovered:
                if time.time() > phase1_deadline:
                    break
                start = min(uncovered)
                block = list(start)
                while len(block) < k:
                    best_gain = -1
                    best_point = -1
                    for x in range(v):
                        if x in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            cand = tuple(sorted(rest + (x,)))
                            if cand in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    if best_point == -1:
                        for x in range(v):
                            if x not in block:
                                best_point = x
                                break
                    block.append(best_point)
                block.sort()
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    # --- Phase 2: also run a few pure greedy restarts as a safety net / comparison ---
    phase2_deadline = time.time() + 90.0
    for seed in range(4):
        if time.time() > phase2_deadline:
            break
        r = random.Random(seed * 31 + 11)
        blocks = _greedy_fallback(v, k, t, deadline=phase2_deadline, rng=r)
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    # --- Phase 3: polish with redundant-block pruning ---
    if best_blocks is not None:
        polished = _prune_blocks(best_blocks, v, k, t, deadline=deadline)
        if len(polished) < best_len:
            best_blocks = polished
            best_len = len(polished)

    # --- Final validity guard: if invalid, fall back to full greedy ---
    if best_blocks is None or not _is_valid(best_blocks, v, k, t):
        r = random.Random(999)
        best_blocks = _greedy_fallback(v, k, t, deadline=deadline, rng=r)
        if not _is_valid(best_blocks, v, k, t):
            # Last resort: exhaustive-ish greedy without deadline break.
            tsets = _all_tsubsets(v, t)
            uncovered = set(tsets)
            blocks = []
            while uncovered:
                start = min(uncovered)
                block = list(start)
                while len(block) < k:
                    best_gain = -1
                    best_point = -1
                    for x in range(v):
                        if x in block:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block, t - 1):
                            cand = tuple(sorted(rest + (x,)))
                            if cand in uncovered:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                    if best_point == -1:
                        for x in range(v):
                            if x not in block:
                                best_point = x
                                break
                    block.append(best_point)
                block.sort()
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
            best_blocks = blocks

    return best_blocks
