import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    t_count = nCr(k, t)

    # Precompute: for each block, which t-subsets it covers (as array of indices)
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]

    # Build inverse: for each t-subset, list of blocks that cover it
    # Use a list of lists for memory efficiency
    ts_to_blocks = [[] for _ in range(n_ts)]
    for b_idx in range(n_ks):
        for s_idx in block_t_indices[b_idx]:
            ts_to_blocks[s_idx].append(b_idx)

    best_solution = None
    best_score = float('inf')

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)

        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts

        # cover_freq tracks how many times each t-subset has been covered
        cover_freq = np.zeros(n_ts, dtype=np.int32)

        # For efficiency, maintain a "gain" array: for each block, how many NEW t-subsets it covers
        # gain[b] = t_count - (number of already-covered t-subsets in block b)
        # We update this incrementally when we add a block

        # Precompute which blocks cover which t-subsets for fast gain updates
        # When we add a block, we need to decrement gain for all blocks that share t-subsets with it
        # This is expensive, so instead we recompute gains periodically or use a different strategy

        # Alternative: use a "potential" function that we update lazily
        # For now, recompute gains each iteration but use numpy efficiently

        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None

            # Compute gain for each block: number of new t-subsets it covers
            # covered[block_t_indices] gives a (n_ks, t_count) boolean matrix
            # sum along axis 1 gives count of already-covered per block
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts

            max_gain = int(np.max(gains))
            if max_gain == 0:
                break

            # Among blocks with max gain, pick one with good "rarity" score
            max_gain_blocks = np.where(gains == max_gain)[0]

            if len(max_gain_blocks) > 50:
                # Sample from top candidates to keep it fast
                sample_size = min(50, len(max_gain_blocks))
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), sample_size)]
            else:
                candidates = max_gain_blocks

            # Score: prefer blocks covering rarer (less frequently covered) t-subsets
            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
                    # Bonus for covering rare subsets (those with low cover_freq)
                    bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]]))
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus * 0.1
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = cand

            blocks_indices.append(best_candidate)

            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = int(np.sum(newly_covered))
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)

        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1

        # Aggressive deletion with multiple passes
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []

            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                other_covers = coverage_count[subs_in_block] - 1
                if np.all(other_covers > 0):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)

            current_blocks = new_blocks

            # Second pass: try removing blocks where >= 95% of subsets are over-covered
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                other_covers = coverage_count[subs_in_block] - 1
                if np.mean(other_covers > 0) >= 0.95:
                    uncovered_mask = other_covers <= 0
                    if not np.any(uncovered_mask):
                        improved = True
                        coverage_count[subs_in_block] -= 1
                        continue
                    # Check if uncovered subsets are covered by other blocks
                    uncovered_subs = subs_in_block[uncovered_mask]
                    can_remove = True
                    for unc_sub in uncovered_subs:
                        found = False
                        for other_b in new_blocks:
                            if other_b != b_idx:
                                if np.any(block_t_indices[other_b] == unc_sub):
                                    found = True
                                    break
                        if not found:
                            can_remove = False
                            break
                    if can_remove:
                        improved = True
                        coverage_count[subs_in_block] -= 1
                        continue
                new_blocks.append(b_idx)
            current_blocks = new_blocks

        return current_blocks, len(current_blocks)

    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)

        if blocks is None:
            break

        if score < best_score:
            best_score = score
            best_solution = blocks

        restart_count += 1

    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]

    return final_blocks
