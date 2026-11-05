import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def now():
        return time.time()

    # ---- Precomputation ----
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)

    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}

    # For each block, the list of subset-indices it covers
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)

    # For each subset, list of blocks that cover it
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)

    base_potential = np.array([len(s) for s in block_subsets], dtype=np.int32)

    # ---- Greedy with optional random tie-breaking and optional "rarity bias" ----
    def run_greedy(seed=None, rarity_bias=False, randomize=True):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random

        cover_count = np.zeros(N_subsets, dtype=np.int32)
        potential = base_potential.copy()
        selected = []

        uncovered_mask = np.ones(N_subsets, dtype=bool)
        n_uncovered = N_subsets

        while n_uncovered > 0:
            if now() > deadline:
                return None

            max_pot = int(np.max(potential))
            if max_pot == 0:
                break

            if randomize:
                candidates = np.where(potential == max_pot)[0]
                if len(candidates) > 0:
                    # Optional rarity bias: among ties, prefer blocks covering
                    # subsets that are currently "rare" (low cover_count among
                    # blocks that still need coverage). Simplified: just random.
                    best_idx = candidates[rng.randint(0, len(candidates) - 1)]
                else:
                    break
            else:
                best_idx = int(np.argmax(potential))

            selected.append(best_idx)

            for s in block_subsets[best_idx]:
                old = cover_count[s]
                cover_count[s] = old + 1
                if old == 0:
                    n_uncovered -= 1
                    for b in blocks_covering_subset[s]:
                        potential[b] -= 1

        return selected, cover_count

    def remove_redundant(selected, cover_count):
        """Remove any block whose every subset has cover_count > 1."""
        if selected is None:
            return None, None
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                b = selected[i]
                redundant = True
                for s in block_subsets[b]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_subsets[b]:
                        cover_count[s] -= 1
                    selected.pop(i)
                    changed = True
        return selected, cover_count

    def swap_local_search(selected, cover_count, max_iters=200):
        """Try to replace a block with another that allows net removal >= 1."""
        if selected is None:
            return None, None
        selected = list(selected)
        cover_count = cover_count.copy()
        n = len(selected)
        improved = True
        iters = 0
        while improved and iters < max_iters and now() < deadline:
            improved = False
            iters += 1
            # Random order of selected blocks to try as removal candidates
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if now() > deadline:
                    return selected, cover_count
                b = selected[i]
                # Check: if we remove b, which subsets become uncovered?
                uncovered = []
                for s in block_subsets[b]:
                    if cover_count[s] == 1:
                        uncovered.append(s)
                if not uncovered:
                    # b is redundant; just remove
                    for s in block_subsets[b]:
                        cover_count[s] -= 1
                    selected.pop(i)
                    n -= 1
                    improved = True
                    break
                # Try to find a block c (not in selected) that covers ALL uncovered
                # and also covers at least one subset that is covered only by b
                # (so that adding c lets us remove b, and we may remove others too).
                # Simpler: find c covering all `uncovered`. If found, check net gain.
                # Build set of uncovered for fast membership
                unc_set = set(uncovered)
                best_c = -1
                best_gain = 0
                # Candidate blocks: those covering the first uncovered subset
                first_unc = uncovered[0]
                for c in blocks_covering_subset[first_unc]:
                    if c in selected:
                        continue
                    # Check c covers all uncovered
                    ok = True
                    for s in unc_set:
                        if s not in set(block_subsets[c]):
                            ok = False
                            break
                    if not ok:
                        continue
                    # If we add c and remove b, how many additional blocks become removable?
                    # Approximate: count subsets that are covered only by b (after removing b)
                    # that c also covers -> those were the reason b was needed.
                    # Net change: -1 (remove b) + 1 (add c) = 0, but maybe other blocks
                    # become redundant. We'll just accept if it doesn't increase size,
                    # and rely on remove_redundant to clean up.
                    # To be safe, only accept if c covers at least one "b-only" subset
                    # that is also covered by some other selected block (so b truly
                    # becomes redundant after adding c).
                    # Simplify: accept the swap, then run remove_redundant.
                    best_c = c
                    break
                if best_c >= 0:
                    # Apply swap
                    for s in block_subsets[b]:
                        cover_count[s] -= 1
                    selected.pop(i)
                    n -= 1
                    selected.append(best_c)
                    for s in block_subsets[best_c]:
                        cover_count[s] += 1
                    improved = True
                    # Run redundancy removal
                    selected, cover_count = remove_redundant(selected, cover_count)
                    n = len(selected)
                    break
        return selected, cover_count

    # ---- Main: multi-start ----
    best_selected = None
    best_count = None
    best_len = float('inf')

    # First: deterministic greedy (seed=0, no randomize) as baseline
    res = run_greedy(seed=0, randomize=False)
    if res is not None:
        sel, cc = res
        sel, cc = remove_redundant(sel, cc)
        if sel is not None:
            best_selected = sel
            best_count = cc
            best_len = len(sel)

    # Now do randomized restarts with swap local search
    n_restarts = 0
    max_restarts = 30
    for trial in range(max_restarts):
        if now() > deadline - 10:
            break
        seed = 1000 + trial
        res = run_greedy(seed=seed, randomize=True)
        if res is None:
            break
        sel, cc = res
        sel, cc = remove_redundant(sel, cc)
        if sel is None:
            continue
        # Apply swap local search
        sel, cc = swap_local_search(sel, cc, max_iters=50)
        if sel is not None and len(sel) < best_len:
            best_len = len(sel)
            best_selected = sel
            best_count = cc
            n_restarts += 1
            # If we've found something very good, we can stop early
            if best_len <= 280:
                break

    # If we still have time, do a few more swap-only iterations on the best
    if best_selected is not None and now() < deadline - 5:
        best_selected, best_count = swap_local_search(best_selected, best_count, max_iters=100)
        if best_selected is not None:
            best_len = len(best_selected)

    if best_selected is None:
        # Fallback: return a valid (but possibly large) cover from a fresh greedy
        res = run_greedy(seed=0, randomize=False)
        if res is not None:
            best_selected, best_count = res
            best_selected, best_count = remove_redundant(best_selected, best_count)
        if best_selected is None:
            # Last resort: return empty (invalid) -- but this shouldn't happen
            return []

    final_blocks = [list(all_blocks[i]) for i in best_selected]
    return final_blocks
