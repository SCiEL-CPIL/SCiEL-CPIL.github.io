import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 150s kill

    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}

    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)

    def nCr(n, r):
        if r > n:
            return 0
        if r == 0:
            return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res

    t_count = nCr(k, t)

    # Precompute block_t_indices: for each k-subset, which t-subsets it covers
    # Using a sparse representation for efficiency
    block_t_indices = []
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        block_t_indices.append([idx_map[sub] for sub in subs])

    best_solution = None
    best_score = float('inf')

    def run_greedy(seed, aggressive_local_search=True):
        """Run one greedy construction with given random seed."""
        rng = random.Random(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts

        # Track coverage counts for local search
        coverage_count = np.zeros(n_ts, dtype=np.int32)

        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None

            # Calculate gains: how many new t-subsets each block would cover
            # Use vectorized approach for speed
            gains = np.zeros(n_ks, dtype=np.int32)
            for i in range(n_ks):
                if not covered[block_t_indices[i]].all():
                    gains[i] = np.sum(~covered[block_t_indices[i]])

            max_gain = np.max(gains)
            if max_gain == 0:
                break

            # Pick from top candidates with some randomness
            top_k = min(100, n_ks)
            top_indices = np.argpartition(gains, -top_k)[-top_k:]
            
            # Weighted random selection favoring higher gains
            top_gains = gains[top_indices]
            top_gains = top_gains - np.min(top_gains) + 1  # Ensure positive
            probs = top_gains / np.sum(top_gains)
            
            best_idx = top_indices[rng.choices(range(len(top_indices)), weights=probs, k=1)[0]]

            blocks_indices.append(best_idx)

            # Update covered
            subs_to_cover = block_t_indices[best_idx]
            new_covers = np.sum(~covered[subs_to_cover])
            remaining -= new_covers
            covered[subs_to_cover] = True
            coverage_count[subs_to_cover] += 1

        current_blocks = list(blocks_indices)

        if not aggressive_local_search:
            return current_blocks, len(current_blocks)

        # Local search phase 1: Remove redundant blocks (multiple passes)
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                return current_blocks, len(current_blocks)

            improved = False
            new_blocks = []

            # Recalculate coverage
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1

            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.min(coverage_count[subs_in_block]) > 1:
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)

            current_blocks = new_blocks

        # Local search phase 2: Try swaps to reduce block count
        # For each block, try replacing it with another block that covers
        # the same "critical" t-subsets (those only covered by this block)
        for swap_pass in range(5):
            if time.time() - start_time > time_limit - 2.0:
                break

            improved_swap = False
            new_blocks = list(current_blocks)

            # Recalculate coverage
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in new_blocks:
                coverage_count[block_t_indices[b_idx]] += 1

            for i in range(len(new_blocks)):
                if time.time() - start_time > time_limit - 2.0:
                    break

                b_idx = new_blocks[i]
                subs_in_block = block_t_indices[b_idx]

                # Find t-subsets only covered by this block
                critical = coverage_count[subs_in_block] == 1
                if not np.any(critical):
                    # This block is redundant, remove it
                    new_blocks.pop(i)
                    improved_swap = True
                    i -= 1
                    # Update coverage
                    coverage_count[subs_in_block] -= 1
                    continue

                # Try to find a replacement block that covers all critical subsets
                critical_set = set(subs_in_block[critical])
                
                best_cand = -1
                best_score_cand = -1

                for cand_idx in range(n_ks):
                    if cand_idx == b_idx:
                        continue
                    cand_subs = block_t_indices[cand_idx]
                    cand_set = set(cand_subs)
                    
                    # Check if candidate covers all critical subsets
                    if not critical_set.issubset(cand_set):
                        continue

                    # Count how many of the current block's subsets are not covered by candidate
                    # (these would become uncovered if we swap)
                    lost = len([s for s in subs_in_block if s not in cand_set and coverage_count[s] == 1])
                    
                    # Count new coverage from candidate
                    new_cover = len([s for s in cand_subs if coverage_count[s] == 0])
                    
                    score = new_cover - lost
                    if score > best_score_cand:
                        best_score_cand = score
                        best_cand = cand_idx

                if best_cand != -1 and best_score_cand >= 0:
                    # Accept the swap
                    new_blocks[i] = best_cand
                    improved_swap = True
                    # Update coverage
                    coverage_count[subs_in_block] -= 1
                    coverage_count[block_t_indices[best_cand]] += 1

            current_blocks = new_blocks
            if not improved_swap:
                break

        # Local search phase 3: Try adding blocks that enable removal of multiple existing blocks
        if time.time() - start_time < time_limit - 1.0:
            for add_pass in range(3):
                improved_add = False
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current_blocks:
                    coverage_count[block_t_indices[b_idx]] += 1

                for cand_idx in range(n_ks):
                    if time.time() - start_time > time_limit - 1.0:
                        break
                    if cand_idx in current_blocks:
                        continue
                    
                    cand_subs = block_t_indices[cand_idx]
                    
                    # Count how many blocks become redundant if we add this block
                    redundant_count = 0
                    for b_idx in current_blocks:
                        subs_in_block = block_t_indices[b_idx]
                        # Check if all subsets in this block are covered by other blocks + candidate
                        for s in subs_in_block:
                            if coverage_count[s] == 1 and s not in cand_subs:
                                break
                        else:
                            redundant_count += 1
                    
                    # Net benefit: remove redundant_count blocks, add 1
                    if redundant_count > 1:
                        # Accept
                        current_blocks.append(cand_idx)
                        for b_idx in list(current_blocks[:-1]):
                            subs_in_block = block_t_indices[b_idx]
                            all_covered = True
                            for s in subs_in_block:
                                if coverage_count[s] == 1 and s not in cand_subs:
                                    all_covered = False
                                    break
                            if all_covered:
                                current_blocks.remove(b_idx)
                        improved_add = True
                        break

                if not improved_add:
                    break

        return current_blocks, len(current_blocks)

    # Run multiple random restarts
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        blocks, score = run_greedy(restart_count)

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
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]

    return final_blocks
