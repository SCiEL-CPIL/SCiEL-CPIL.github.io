"""Covering design C(v,k,t) via scarcity-weighted greedy + deletion, vectorized."""
import time
import numpy as np
from itertools import combinations


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin before 120s kill

    # Precompute all t-subsets and their indices
    all_t_list = list(combinations(range(v), t))
    N = len(all_t_list)
    all_t = np.array(all_t_list, dtype=np.int32)
    t_index = {tuple(r): i for i, r in enumerate(all_t_list)}

    best_blocks = None
    best_score = float('inf')

    def run_pass(seed):
        rng = np.random.default_rng(seed)
        covered = np.zeros(N, dtype=bool)
        weights = np.ones(N, dtype=np.float64)
        blocks = []

        while not covered.all():
            if time.time() > deadline:
                return None

            unc_idx = np.where(~covered)[0]
            if len(unc_idx) == 0:
                break

            # Stochastic seed selection: sample candidates, pick highest weight
            if len(unc_idx) <= 8:
                cands = unc_idx
            else:
                cands = rng.choice(unc_idx, size=8, replace=False)
            cands_w = weights[cands]
            seed_subset = all_t[cands[int(np.argmax(cands_w))]]
            block = list(seed_subset)

            while len(block) < k:
                if time.time() > deadline:
                    return None
                block_set = set(block)
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break

                # Compute gain for each candidate
                gains = np.zeros(len(candidates), dtype=np.int32)
                cand_arr = np.array(candidates, dtype=np.int32)
                block_arr = np.array(block, dtype=np.int32)

                # For each (t-1)-subset of block, and each candidate x,
                # check if sorted(subset + x) is an uncovered t-subset
                for ts in combinations(block_arr, t - 1):
                    ts_arr = np.array(ts, dtype=np.int32)
                    for ci, x in enumerate(cand_arr):
                        # Merge ts and x in sorted order
                        merged = list(ts)
                        pos = np.searchsorted(ts_arr, x, side='right')
                        merged = list(ts[:pos]) + [x] + list(ts[pos:])
                        key = tuple(merged)
                        idx = t_index.get(key, -1)
                        if idx >= 0 and not covered[idx]:
                            gains[ci] += 1

                max_gain = np.max(gains)
                best_cands = candidates[gains == max_gain]
                if len(best_cands) > 1:
                    chosen = int(rng.choice(best_cands))
                else:
                    chosen = int(best_cands[0])
                block.append(chosen)

            block.sort()
            blocks.append(block)

            # Update covered
            for comb in combinations(block, t):
                idx = t_index.get(comb, -1)
                if idx >= 0:
                    covered[idx] = True

            # Update weights: uncovered subsets get slightly higher weight
            weights[~covered] *= 1.05

        # Deletion phase
        blocks = list(blocks)
        for _ in range(5):
            if time.time() > deadline:
                break
            changed = True
            while changed:
                changed = False
                if time.time() > deadline:
                    break
                # Try removing each block (from end to start)
                for i in range(len(blocks) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    saved = blocks[i]
                    blocks[i] = None
                    # Check coverage
                    covered2 = np.zeros(N, dtype=bool)
                    for b in blocks:
                        if b is None:
                            continue
                        for comb in combinations(b, t):
                            idx = t_index.get(comb, -1)
                            if idx >= 0:
                                covered2[idx] = True
                    if covered2.all():
                        del blocks[i]
                        changed = True
                        break
                    else:
                        blocks[i] = saved

        return blocks

    # Run multiple passes
    for seed in range(40):
        if time.time() > deadline:
            break
        result = run_pass(seed)
        if result is not None and len(result) < best_score:
            best_score = len(result)
            best_blocks = result

    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(combinations(block, t))
        return blocks

    return best_blocks
