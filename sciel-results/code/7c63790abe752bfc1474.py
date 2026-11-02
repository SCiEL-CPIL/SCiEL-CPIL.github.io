import time
import random
import itertools
import numpy as np
from math import comb as math_comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin: evaluator kills at 150s

    # Precompute all k-blocks and t-subsets
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Build incidence: block -> list of t-subset indices it covers
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t)]
    for j, blk in enumerate(all_k_blocks):
        idxs = []
        for ts in itertools.combinations(blk, t):
            idx = ts_index[ts]
            idxs.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(idxs)

    # Bitset-based coverage tracking
    # Each block gets a numpy boolean vector of length num_t
    # But that's 75582 * 11628 booleans ≈ 880MB - too large
    # Instead, use a different approach: track coverage counts per t-subset

    # For the greedy, we need fast gain updates.
    # gain[b] = number of uncovered t-subsets in block b
    # We track uncovered as a boolean array and gain as an int array.

    def run_greedy(seed, randomized=True, time_budget=None):
        """Run randomized greedy set cover. Returns list of block indices."""
        rng = np.random.RandomState(seed)
        gain = np.full(num_k, math_comb(k, t), dtype=np.int32)
        uncovered = np.ones(num_t, dtype=bool)
        selected = np.zeros(num_k, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 15.0:
                return None
            mg = int(np.max(gain))
            if mg <= 0:
                break
            cands = np.where((gain == mg) & (~selected))[0]
            if len(cands) == 0:
                break
            if randomized and len(cands) > 1:
                # Random tie-breaking with slight perturbation
                w = np.ones(len(cands)) + rng.rand(len(cands)) * 0.5
                p = w / w.sum()
                b = cands[rng.choice(len(cands), p=p)]
            else:
                b = cands[rng.randint(0, len(cands))] if randomized else cands[0]
            sol.append(int(b))
            selected[b] = True
            gain[b] = -1
            for ti in block_to_ts[b]:
                if uncovered[ti]:
                    uncovered[ti] = False
                    for nb in ts_to_blocks[ti]:
                        if not selected[nb]:
                            gain[nb] -= 1
        return sol

    def idx_to_blocks(idxs):
        return [list(all_k_blocks[i]) for i in idxs]

    def coverage_counts(blks):
        """Compute coverage count for each t-subset given a list of block indices."""
        cc = np.zeros(num_t, dtype=np.int32)
        for b in blks:
            for ti in block_to_ts[b]:
                cc[ti] += 1
        return cc

    def remove_redundant(blk_idxs, time_limit=None):
        """Remove redundant blocks. blk_idxs is a list of block indices."""
        if not blk_idxs:
            return blk_idxs
        cc = coverage_counts(blk_idxs)
        changed = True
        while changed:
            changed = False
            for i in range(len(blk_idxs)):
                if time_limit and time.time() > time_limit:
                    return blk_idxs
                # Check if all t-subsets in this block are covered by other blocks
                redundant = True
                for ti in block_to_ts[blk_idxs[i]]:
                    if cc[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    for ti in block_to_ts[blk_idxs[i]]:
                        cc[ti] -= 1
                    blk_idxs.pop(i)
                    changed = True
                    break
        return blk_idxs

    def local_search_single_swap(blk_idxs, max_passes=100, time_limit=None):
        """Try to replace blocks to reduce redundancy, maintaining coverage."""
        if not blk_idxs:
            return blk_idxs
        cc = coverage_counts(blk_idxs)
        sol_set = set(blk_idxs)
        improved = False

        for _ in range(max_passes):
            if time_limit and time.time() > time_limit:
                break
            found = False
            for i in range(len(blk_idxs)):
                if time_limit and time.time() > time_limit:
                    break
                # Find t-subsets only covered by block i
                ut = [ti for ti in block_to_ts[blk_idxs[i]] if cc[ti] == 1]
                if not ut:
                    continue
                # Find blocks that cover all t-subsets in ut
                cands = set(ts_to_blocks[ut[0]])
                for ti in ut[1:]:
                    cands.intersection_update(ts_to_blocks[ti])
                    if not cands:
                        break
                if not cands:
                    continue
                valid = [c for c in cands if c not in sol_set]
                if not valid:
                    continue
                # Pick a candidate that minimizes additional overlap
                best_c = None
                best_overlap = float('inf')
                for c in valid[:min(len(valid), 20)]:  # limit candidates
                    overlap = sum(1 for ti in block_to_ts[c] if cc[ti] > 0)
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_c = c
                if best_c is not None:
                    # Remove old block, add new
                    for ti in block_to_ts[blk_idxs[i]]:
                        cc[ti] -= 1
                    for ti in block_to_ts[best_c]:
                        cc[ti] += 1
                    sol_set.discard(blk_idxs[i])
                    sol_set.add(best_c)
                    blk_idxs[i] = best_c
                    improved = True
                    found = True
                    break
            if not found:
                break

        # After swaps, try to remove redundant blocks
        blk_idxs = remove_redundant(blk_idxs, time_limit)
        return blk_idxs

    def local_search_two_exchange(blk_idxs, max_passes=50, time_limit=None):
        """Try to replace two blocks with one, or restructure pairs."""
        if len(blk_idxs) < 2:
            return blk_idxs
        cc = coverage_counts(blk_idxs)
        sol_set = set(blk_idxs)
        improved = False
        n = len(blk_idxs)

        for _ in range(max_passes):
            if time_limit and time.time() > time_limit:
                break
            found = False
            for i in range(n):
                if time_limit and time.time() > time_limit:
                    break
                for j in range(i + 1, min(i + 10, n)):  # limit search
                    if time_limit and time.time() > time_limit:
                        break
                    # Union of uniquely covered t-subsets by blocks i and j
                    ut = []
                    for ti in block_to_ts[blk_idxs[i]]:
                        if cc[ti] == 1:
                            ut.append(ti)
                    for ti in block_to_ts[blk_idxs[j]]:
                        if cc[ti] == 1:
                            ut.append(ti)
                    if not ut:
                        continue
                    # Find a single block covering all t-subsets in ut
                    cands = set(ts_to_blocks[ut[0]])
                    for ti in ut[1:]:
                        cands.intersection_update(ts_to_blocks[ti])
                        if not cands:
                            break
                    if cands:
                        valid = [c for c in cands if c not in sol_set]
                        if valid:
                            best_c = random.choice(valid[:min(len(valid), 10)])
                            # Remove both blocks, add one
                            for ti in block_to_ts[blk_idxs[i]]:
                                cc[ti] -= 1
                            for ti in block_to_ts[blk_idxs[j]]:
                                cc[ti] -= 1
                            for ti in block_to_ts[best_c]:
                                cc[ti] += 1
                            sol_set.discard(blk_idxs[i])
                            sol_set.discard(blk_idxs[j])
                            sol_set.add(best_c)
                            blk_idxs.pop(j)
                            blk_idxs[i] = best_c
                            improved = True
                            found = True
                            n = len(blk_idxs)
                            break
                    # Try replacing both with two new blocks
                    # For each of the two blocks, find replacements
                    ci = [ti for ti in block_to_ts[blk_idxs[i]] if cc[ti] == 1]
                    cj = [ti for ti in block_to_ts[blk_idxs[j]] if cc[ti] == 1]
                    if ci:
                        vi = set(ts_to_blocks[ci[0]])
                        for ti in ci[1:]:
                            vi.intersection_update(ts_to_blocks[ti])
                    else:
                        vi = set(range(num_k))
                    if cj:
                        vj = set(ts_to_blocks[cj[0]])
                        for ti in cj[1:]:
                            vj.intersection_update(ts_to_blocks[ti])
                    else:
                        vj = set(range(num_k))
                    vi = [c for c in vi if c not in sol_set]
                    vj = [c for c in vj if c not in sol_set and c not in vi]
                    if vi and vj:
                        ni = random.choice(vi[:min(len(vi), 10)])
                        nj = random.choice(vj[:min(len(vj), 10)])
                        for ti in block_to_ts[blk_idxs[i]]:
                            cc[ti] -= 1
                        for ti in block_to_ts[blk_idxs[j]]:
                            cc[ti] -= 1
                        for ti in block_to_ts[ni]:
                            cc[ti] += 1
                        for ti in block_to_ts[nj]:
                            cc[ti] += 1
                        sol_set.discard(blk_idxs[i])
                        sol_set.discard(blk_idxs[j])
                        sol_set.add(ni)
                        sol_set.add(nj)
                        blk_idxs[i] = ni
                        blk_idxs[j] = nj
                        improved = True
                        found = True
                        break
                if found:
                    break
            if not found:
                break

        blk_idxs = remove_redundant(blk_idxs, time_limit)
        return blk_idxs

    # Phase 1: Multiple greedy runs
    best_blk_idxs = None
    best_len = float('inf')

    # Deterministic greedy
    s0 = run_greedy(0, randomized=False)
    if s0 is not None:
        best_blk_idxs = s0
        best_len = len(s0)

    # Randomized greedy runs
    for seed in range(1, 50):
        if time.time() > deadline - 30.0:
            break
        sol = run_greedy(seed, randomized=True)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_blk_idxs = sol

    # Phase 2: Local search on best solution
    if best_blk_idxs is not None:
        # Remove redundant first
        best_blk_idxs = remove_redundant(best_blk_idxs, deadline - 10.0)
        best_len = len(best_blk_idxs)

        # Iterative local search
        for iteration in range(50):
            if time.time() > deadline - 10.0:
                break

            # Single swap
            improved = False
            prev_len = len(best_blk_idxs)
            best_blk_idxs = local_search_single_swap(
                best_blk_idxs, max_passes=80, time_limit=deadline - 10.0
            )
            if len(best_blk_idxs) < prev_len:
                improved = True
                best_len = len(best_blk_idxs)

            # Two exchange
            if time.time() > deadline - 10.0:
                break
            prev_len = len(best_blk_idxs)
            best_blk_idxs = local_search_two_exchange(
                best_blk_idxs, max_passes=40, time_limit=deadline - 10.0
            )
            if len(best_blk_idxs) < prev_len:
                improved = True
                best_len = len(best_blk_idxs)

            if not improved:
                # Try more aggressive single swap
                prev_len = len(best_blk_idxs)
                best_blk_idxs = local_search_single_swap(
                    best_blk_idxs, max_passes=200, time_limit=deadline - 8.0
                )
                if len(best_blk_idxs) < prev_len:
                    best_len = len(best_blk_idxs)
                else:
                    break

        # Final polish
        if time.time() < deadline - 5.0:
            for _ in range(20):
                if time.time() > deadline - 4.0:
                    break
                prev_len = len(best_blk_idxs)
                best_blk_idxs = local_search_single_swap(
                    best_blk_idxs, max_passes=150, time_limit=deadline - 4.0
                )
                if len(best_blk_idxs) < prev_len:
                    best_len = len(best_blk_idxs)
                else:
                    break

    # Convert to blocks
    if best_blk_idxs is None:
        # Fallback: just return greedy solution
        best_blk_idxs = run_greedy(0, randomized=False)
        if best_blk_idxs is None:
            best_blk_idxs = []

    blocks = idx_to_blocks(best_blk_idxs)

    # Ensure no duplicates
    seen = set()
    unique_blocks = []
    for b in blocks:
        tb = tuple(b)
        if tb not in seen:
            seen.add(tb)
            unique_blocks.append(b)

    return unique_blocks
