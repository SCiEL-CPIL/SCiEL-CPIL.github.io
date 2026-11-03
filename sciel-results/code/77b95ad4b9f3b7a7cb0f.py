import itertools
import time
import random
import numpy as np
import heapq

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(points, t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Bitset representation
    tsub_mask = [0] * n_blocks
    tsub_blocks_mask = [0] * n_tsubsets
    block_tsubs = [[] for _ in range(n_blocks)]

    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = tsub_idx[ts]
            mask |= (1 << idx)
            tsub_blocks_mask[idx] |= (1 << b_idx)
            block_tsubs[b_idx].append(idx)
        tsub_mask[b_idx] = mask

    full_tsub_mask = (1 << n_tsubsets) - 1

    def greedy_cover(seed):
        """
        Standard greedy using bitsets for gain calculation.
        """
        rng = random.Random(seed)
        uncovered_mask = full_tsub_mask
        selected = []

        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break

            best_block = -1
            best_gain = 0

            for b_idx in range(n_blocks):
                if not (tsub_mask[b_idx] & uncovered_mask):
                    continue
                gain = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    if best_gain == n_tsubsets:
                        break

            if best_block == -1:
                break

            selected.append(best_block)
            uncovered_mask &= ~tsub_mask[best_block]

        return selected

    def local_search(blocks, seed):
        """
        Local search with swaps and deletions.
        """
        rng = random.Random(seed)
        if not blocks:
            return blocks

        block_set = set(blocks)

        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False

            # Phase 1: Remove redundant blocks
            to_remove = []
            for b in block_set:
                redundant = True
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(b)

            if to_remove:
                for b in to_remove:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                improved = True
                continue

            # Phase 2: Try swaps
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            for b in current_blocks:
                if time.time() > deadline - 2.0:
                    break
                if b not in block_set:
                    continue

                unique_tsubs = []
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        unique_tsubs.append(ts_idx)

                if not unique_tsubs:
                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
                    improved = True
                    break

                cand_mask = (1 << n_blocks) - 1
                for ts_idx in unique_tsubs:
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b)

                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 20:
                        break

                if not candidates:
                    continue

                rng.shuffle(candidates)

                swapped = False
                for r in candidates:
                    if r in block_set:
                        continue

                    block_set.remove(b)
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1

                    block_set.add(r)
                    for ts_idx in block_tsubs[r]:
                        cover_count[ts_idx] += 1

                    improved = True
                    swapped = True
                    break

                if swapped:
                    break

        return list(block_set)

    def two_for_one_search(blocks, seed):
        """
        Try to replace two blocks with one.
        """
        rng = random.Random(seed)
        if len(blocks) < 2:
            return blocks

        block_set = set(blocks)
        cover_count = [0] * n_tsubsets
        for b in block_set:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_blocks = list(block_set)
            rng.shuffle(current_blocks)

            n_checks = 100
            for _ in range(n_checks):
                if time.time() > deadline - 2.0:
                    break
                if len(block_set) < 2:
                    break

                i, j = rng.sample(range(len(current_blocks)), 2)
                b1 = current_blocks[i]
                b2 = current_blocks[j]

                if b1 not in block_set or b2 not in block_set:
                    continue

                required_mask = 0
                for ts_idx in block_tsubs[b1]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)
                for ts_idx in block_tsubs[b2]:
                    if cover_count[ts_idx] <= 1:
                        required_mask |= (1 << ts_idx)

                if required_mask == 0:
                    continue

                cand_mask = (1 << n_blocks) - 1
                m = required_mask
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    cand_mask &= tsub_blocks_mask[ts_idx]
                    if cand_mask == 0:
                        break
                    m ^= lsb

                if cand_mask == 0:
                    continue

                cand_mask &= ~(1 << b1)
                cand_mask &= ~(1 << b2)

                candidates = []
                m = cand_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    candidates.append(idx)
                    m ^= lsb
                    if len(candidates) > 5:
                        break

                if not candidates:
                    continue

                rng.shuffle(candidates)
                r = candidates[0]

                block_set.remove(b1)
                for ts_idx in block_tsubs[b1]:
                    cover_count[ts_idx] -= 1

                block_set.remove(b2)
                for ts_idx in block_tsubs[b2]:
                    cover_count[ts_idx] -= 1

                block_set.add(r)
                for ts_idx in block_tsubs[r]:
                    cover_count[ts_idx] += 1

                improved = True
                break

        return list(block_set)

    def branch_and_bound(uncovered_mask, current_blocks, current_size, best_size, seed):
        """
        Branch-and-bound with bitset-based lower bounds.
        
        Lower bound: For each uncovered t-subset, at least one block must cover it.
        We use a greedy independent set heuristic on the uncovered subsets to estimate
        how many more blocks are needed.
        
        Returns: (found, best_solution)
        """
        if time.time() > deadline - 3.0:
            return False, None

        if uncovered_mask == 0:
            return True, list(current_blocks)

        # Lower bound: greedy independent set on uncovered subsets
        # Two t-subsets are "conflicting" if no single block covers both
        # We want a lower bound on the number of blocks needed
        # Simple LB: count uncovered subsets, divide by max coverage per block
        n_uncovered = uncovered_mask.bit_count()
        if n_uncovered == 0:
            return True, list(current_blocks)
        
        # Max possible coverage of any single block on remaining uncovered
        max_single_cover = 0
        for b_idx in range(n_blocks):
            cover = (tsub_mask[b_idx] & uncovered_mask).bit_count()
            if cover > max_single_cover:
                max_single_cover = cover
        
        if max_single_cover == 0:
            return False, None
            
        lb = (n_uncovered + max_single_cover - 1) // max_single_cover  # ceil division
        
        if current_size + lb >= best_size:
            return False, None

        # Select the most constrained uncovered t-subset (fewest candidate blocks)
        best_ts = -1
        best_candidates = None
        best_cand_count = float('inf')
        
        m = uncovered_mask
        while m:
            lsb = m & (-m)
            ts_idx = lsb.bit_length() - 1
            candidates = tsub_blocks_mask[ts_idx]
            cand_count = candidates.bit_count()
            if cand_count < best_cand_count:
                best_cand_count = cand_count
                best_ts = ts_idx
                best_candidates = candidates
            if cand_count == 0:
                return False, None
            m ^= lsb

        # Branch on which block to use for best_ts
        # Order candidates by how many other uncovered subsets they cover (greedy heuristic)
        candidate_list = []
        m = best_candidates
        while m:
            lsb = m & (-m)
            b_idx = lsb.bit_length() - 1
            if b_idx not in current_blocks:
                cover = (tsub_mask[b_idx] & uncovered_mask).bit_count()
                candidate_list.append((cover, b_idx))
            m ^= lsb
        
        # Sort by coverage descending
        candidate_list.sort(reverse=True)
        
        # Limit branching factor for efficiency
        max_branch = min(8, len(candidate_list))
        
        for cover, b_idx in candidate_list[:max_branch]:
            new_blocks = current_blocks + [b_idx]
            new_uncovered = uncovered_mask & ~tsub_mask[b_idx]
            
            found, sol = branch_and_bound(
                new_uncovered, new_blocks, current_size + 1, best_size, seed
            )
            
            if found and sol is not None:
                best_size = len(sol)
                if best_size <= current_size + 1:  # Can't do better than using this block
                    return True, sol
        
        return False, None

    def bnb_from_greedy(greedy_sol, seed):
        """
        Use branch-and-bound starting from a greedy solution.
        The greedy solution gives an upper bound.
        We try to find a smaller solution.
        """
        if time.time() > deadline - 10.0:
            return greedy_sol

        upper_bound = len(greedy_sol)
        
        # Start with empty set and try to build a covering
        # But that's expensive. Instead, use the greedy solution as a starting point
        # and try to remove blocks via B&B on the complement
        
        # Alternative: Use B&B to find if we can do better than upper_bound
        # Start with no blocks, but use the upper bound to prune
        found, sol = branch_and_bound(
            full_tsub_mask, [], 0, upper_bound, seed
        )
        
        if found and sol is not None:
            return sol
        else:
            return greedy_sol

    best_solution = None
    best_size = float('inf')

    # Run multiple restarts
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 15.0:
            break

        seed = 1000 + i
        greedy_sol = greedy_cover(seed)

        if not greedy_sol:
            continue

        # Local Search
        ls_sol = local_search(greedy_sol, seed)

        # 2-for-1 Search
        t2o_sol = two_for_one_search(ls_sol, seed)

        # Try B&B to potentially improve further
        if time.time() < deadline - 15.0 and len(t2o_sol) > 140:
            bnb_sol = bnb_from_greedy(t2o_sol, seed)
            if bnb_sol is not None:
                t2o_sol = bnb_sol

        if len(t2o_sol) < best_size:
            best_size = len(t2o_sol)
            best_solution = t2o_sol.copy()

    # Final pass: try more aggressive local search on the best solution
    if best_solution is not None and time.time() < deadline - 5.0:
        # Run local search again with different seed
        final_ls = local_search(best_solution, 9999)
        if len(final_ls) < best_size:
            best_size = len(final_ls)
            best_solution = final_ls

    # If no solution found (should not happen), fallback to basic greedy
    if best_solution is None:
        best_solution = greedy_cover(0)

    return [list(all_blocks[idx]) for idx in best_solution]
