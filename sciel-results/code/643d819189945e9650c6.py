import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    points = list(range(v))
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

    # Precompute: for each block, which t-subsets it covers (stored as sorted array)
    block_t_indices = []
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        idxs = np.array([idx_map[sub] for sub in subs], dtype=np.int32)
        block_t_indices.append(idxs)
    
    block_t_indices_arr = np.array(block_t_indices, dtype=np.int32)

    # Precompute, for each point x, the list of t-subset indices that contain x.
    point_ts = [[] for _ in range(v)]
    for i in range(n_ts):
        comb = all_ts[i]
        for x in comb:
            point_ts[x].append(i)
    point_ts = [np.array(lst, dtype=np.int32) for lst in point_ts]

    best_solution = None
    best_score = float('inf')

    def rebuild_coverage(block_list):
        cc = np.zeros(n_ts, dtype=np.int32)
        for b in block_list:
            cc[block_t_indices[b]] += 1
        return cc

    def aggressive_deletion(current_blocks, coverage_count):
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
        return current_blocks

    # Precompute swap data: for each block b and each point x in it,
    # the lost and gained triple indices for swapping x with each outside point y
    # This is expensive, so we compute on demand and cache
    swap_cache = {}

    def get_swap_data(b_idx, old_x):
        """Get lost/gained triples for swapping old_x in block b_idx with any outside point."""
        cache_key = (b_idx, old_x)
        if cache_key in swap_cache:
            return swap_cache[cache_key]
        
        block = all_ks[b_idx]
        other = [p for p in block if p != old_x]
        # For each (t-1)-subset of other, compute the triple with old_x
        rest_list = list(itertools.combinations(other, t - 1))
        lost_base = np.array([idx_map[tuple(sorted(rest + (old_x,)))] for rest in rest_list], dtype=np.int32)
        
        # For each outside point y, gained triples are formed by (t-1)-subsets of other + y
        # We need to compute this for each y not in block
        outside_points = [x for x in range(v) if x not in block]
        gained_dict = {}
        for y in outside_points:
            gained_list = [idx_map[tuple(sorted(rest + (y,)))] for rest in rest_list]
            gained_dict[y] = np.array(gained_list, dtype=np.int32)
        
        if len(swap_cache) < 80000:
            swap_cache[cache_key] = (lost_base, gained_dict)
        return lost_base, gained_dict

    def local_search(blocks, coverage_count, max_iters):
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 4.0:
                break
            improved = False

            # Pass 1: try to delete any redundant blocks
            new_blocks = []
            for b_idx in blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.all(coverage_count[subs_in_block] > 1):
                    coverage_count[subs_in_block] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks

            if len(blocks) == 0:
                break

            # Pass 2: point-swap perturbations
            block_set = set(blocks)
            
            for b_idx in list(blocks):
                if time.time() - start_time > time_limit - 4.0:
                    break
                block = all_ks[b_idx]
                for old_x in block:
                    lost_base, gained_dict = get_swap_data(b_idx, old_x)
                    outside_points = [x for x in range(v) if x not in block]
                    random.shuffle(outside_points)
                    
                    found_swap = False
                    for new_x in outside_points[:12]:
                        gained = gained_dict[new_x]
                        # Check if swap is valid: all lost triples must have coverage > 1
                        if np.all(coverage_count[lost_base] > 1):
                            # Perform swap
                            coverage_count[lost_base] -= 1
                            coverage_count[gained] += 1
                            improved = True
                            found_swap = True
                            break
                    
                    if found_swap:
                        break
                if improved:
                    break

            # After any swap, try deleting again
            if improved:
                new_blocks = []
                for b_idx in blocks:
                    subs_in_block = block_t_indices[b_idx]
                    if np.all(coverage_count[subs_in_block] > 1):
                        coverage_count[subs_in_block] -= 1
                    else:
                        new_blocks.append(b_idx)
                blocks = new_blocks

            if not improved:
                break

        return blocks

    def run_greedy(seed, strategy='standard'):
        random.seed(seed)
        np.random.seed(seed)

        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        cover_freq = np.zeros(n_ts, dtype=np.int32)

        # Random weights per point for randomized greedy
        if strategy == 'weighted':
            point_weights = np.random.uniform(0.3, 3.0, size=v).astype(np.float64)
        elif strategy == 'exponential':
            point_weights = np.random.exponential(1.0, size=v).astype(np.float64) + 0.1
        else:
            point_weights = None

        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None

            # Compute gain: number of new t-subsets covered
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            max_gain_blocks = np.where(gains == max_gain)[0]

            if len(max_gain_blocks) > 40:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(40, len(max_gain_blocks)))]
            else:
                candidates = list(max_gain_blocks)

            # Score candidates
            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered_mask = ~covered[subs]
                if np.any(newly_covered_mask):
                    # Prefer blocks covering rare triples
                    bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered_mask]]))
                else:
                    bonus = 0.0

                # Add point-weight bonus if using weighted strategy
                if point_weights is not None:
                    block_pts = all_ks[cand]
                    pw_bonus = np.sum(point_weights[list(block_pts)]) / k * 0.1
                else:
                    pw_bonus = 0.0

                score_cand = max_gain + bonus * 0.2 + pw_bonus
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = cand

            blocks_indices.append(best_candidate)

            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)
        coverage_count = rebuild_coverage(current_blocks)

        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        coverage_count = rebuild_coverage(current_blocks)

        # Local search with point swaps
        if time.time() - start_time < time_limit - 8.0:
            current_blocks = local_search(current_blocks, coverage_count, max_iters=200)
            coverage_count = rebuild_coverage(current_blocks)

        # Final deletion pass
        current_blocks = aggressive_deletion(current_blocks, coverage_count)

        return current_blocks, len(current_blocks)

    def run_greedy_with_late_swaps(seed, strategy='standard'):
        """Greedy that starts performing swaps earlier to escape local optima."""
        random.seed(seed)
        np.random.seed(seed)

        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        cover_freq = np.zeros(n_ts, dtype=np.int32)

        if strategy == 'weighted':
            point_weights = np.random.uniform(0.3, 3.0, size=v).astype(np.float64)
        else:
            point_weights = None

        # Do greedy until 70% of time is used, then start mixing in swaps
        greedy_fraction = 0.6
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                break

            elapsed = time.time() - start_time
            total_time = time_limit - 5.0
            progress = elapsed / total_time

            # Compute gain
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            max_gain_blocks = np.where(gains == max_gain)[0]

            if len(max_gain_blocks) > 40:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(40, len(max_gain_blocks)))]
            else:
                candidates = list(max_gain_blocks)

            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered_mask = ~covered[subs]
                if np.any(newly_covered_mask):
                    bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered_mask]]))
                else:
                    bonus = 0.0

                if point_weights is not None:
                    block_pts = all_ks[cand]
                    pw_bonus = np.sum(point_weights[list(block_pts)]) / k * 0.1
                else:
                    pw_bonus = 0.0

                score_cand = max_gain + bonus * 0.2 + pw_bonus
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = cand

            blocks_indices.append(best_candidate)

            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)
        coverage_count = rebuild_coverage(current_blocks)

        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        coverage_count = rebuild_coverage(current_blocks)

        # Local search
        if time.time() - start_time < time_limit - 6.0:
            current_blocks = local_search(current_blocks, coverage_count, max_iters=400)
            coverage_count = rebuild_coverage(current_blocks)
            current_blocks = aggressive_deletion(current_blocks, coverage_count)

        return current_blocks, len(current_blocks)

    # Phase 1: Standard greedy with multiple seeds
    restart_count = 0
    while time.time() - start_time < time_limit - 25.0:
        seed = restart_count
        blocks, score = run_greedy(seed, strategy='standard')

        if blocks is None:
            break

        if score < best_score:
            best_score = score
            best_solution = list(blocks)

        restart_count += 1

    # Phase 2: Weighted greedy
    if time.time() - start_time < time_limit - 15.0:
        for seed in range(6):
            if time.time() - start_time > time_limit - 15.0:
                break
            blocks, score = run_greedy(1000 + seed, strategy='weighted')
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = list(blocks)

    # Phase 3: Exponential weighted greedy
    if time.time() - start_time < time_limit - 10.0:
        for seed in range(4):
            if time.time() - start_time > time_limit - 10.0:
                break
            blocks, score = run_greedy(2000 + seed, strategy='exponential')
            if blocks is None:
                continue
            if score < best_score:
                best_score = score
                best_solution = list(blocks)

    # Phase 4: Final high-precision polish on the best solution found
    if best_solution is not None and time.time() - start_time < time_limit - 5.0:
        coverage_count = rebuild_coverage(best_solution)
        best_solution = aggressive_deletion(best_solution, coverage_count)
        coverage_count = rebuild_coverage(best_solution)

        if time.time() - start_time < time_limit - 4.0:
            best_solution = local_search(best_solution, coverage_count, max_iters=600)
            coverage_count = rebuild_coverage(best_solution)
            best_solution = aggressive_deletion(best_solution, coverage_count)

    if best_solution is None:
        blocks, score = run_greedy(0, strategy='standard')
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    return final_blocks
