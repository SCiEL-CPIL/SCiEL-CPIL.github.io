import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s kill

    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)

    # Precompute: for each block, list of t-subset indices it covers
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)

    # Inverse index: for each t-subset, list of blocks that cover it
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)

    # Sort t-subsets by rarity (fewer covering blocks = rarer = higher priority)
    rarity_order = sorted(range(N_t), key=lambda ti: len(t_to_blocks[ti]))

    # Greedy construction (fast, valid starting point)
    def greedy_cover():
        covered = np.zeros(N_t, dtype=bool)
        selected = []
        num_uncov = N_t
        while num_uncov > 0:
            if time.time() > deadline - 2.0:
                break
            # Find uncovered with minimum coverage count
            best_ti = -1
            best_count = float('inf')
            for ti in rarity_order:
                if not covered[ti]:
                    c = len(t_to_blocks[ti])
                    if c < best_count:
                        best_count = c
                        best_ti = ti
                        if c == 1:
                            break
            if best_ti == -1:
                break
            # Pick block covering best_ti with max new coverage
            best_bi = -1
            best_gain = -1
            for bi in t_to_blocks[best_ti]:
                gain = 0
                for ti2 in block_cover[bi]:
                    if not covered[ti2]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            if best_bi == -1:
                best_bi = t_to_blocks[best_ti][0]
            selected.append(best_bi)
            for ti in block_cover[best_bi]:
                if not covered[ti]:
                    covered[ti] = True
                    num_uncov -= 1
        return selected

    # Branch-and-bound
    def branch_and_bound(initial_blocks, best_len):
        """
        B&B on the block selection. State: (selected_blocks, covered_mask, num_uncovered).
        At each node, pick the rarest uncovered t-subset, branch on blocks covering it.
        Prune if lower_bound >= best_len.
        """
        best_sol = list(initial_blocks)
        best_len_local = len(initial_blocks)
        node_count = [0]
        max_nodes = 50000

        # Precompute coverage counts for initial solution
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in initial_blocks:
            for ti in block_cover[bi]:
                counts[ti] += 1
        covered = counts > 0
        num_uncov = int(np.sum(~covered))

        # Lower bound helpers
        def lower_bound(covered_mask, num_unc, selected_count):
            if num_unc == 0:
                return selected_count
            # LB1: each block covers at most C(k,t) t-subsets, so need at least ceil(num_unc / C(k,t))
            max_per_block = len(block_cover[0])  # C(k,t)
            lb1 = (num_unc + max_per_block - 1) // max_per_block
            # LB2: find the rarest uncovered subset; its covering blocks not yet selected give a bound
            # Actually simpler: count uncovered subsets, each needs at least 1 block, but blocks overlap
            # Use: for each uncovered ti, it must be in some block. The minimum number of blocks
            # needed is at least the size of a minimum set cover of the uncovered subsets.
            # Cheap proxy: count distinct "requirements" -- use the fact that each block can cover
            # at most max_per_block uncovered, so lb1 is valid.
            # Better LB: count uncovered, divide by max coverage per block of UNCOVERED only.
            # But that requires per-block computation. Use lb1 as primary.
            # Additional: if there's an uncovered subset with only 1 covering block total,
            # and that block isn't selected, we MUST add it -> +1
            # For now, lb1 is the main bound.
            return selected_count + lb1

        # Recursive B&B with iterative deepening on solution length
        # Instead of pure recursion, use a priority queue (best-first) to find good solutions fast
        # State: (selected_blocks_tuple, covered_mask_bytes, num_uncov, depth)
        # To keep memory bounded, limit the search.

        # Actually, let's do a simpler iterative approach:
        # Start from initial, repeatedly try to remove/swap blocks via B&B on the "must-cover" constraints.

        # Simpler structural approach: local search with B&B-inspired moves
        # For each block in the solution, try to replace it with a block that covers all its
        # unique t-subsets AND allows removal of another block.

        # Let's do: for each block, compute its "essential" t-subsets (only it covers them).
        # If none, remove it. If some, try to find a replacement that covers them and
        # then check if any other block becomes redundant.

        current_blocks = list(initial_blocks)
        cur_counts = counts.copy()
        cur_covered = covered.copy()
        cur_num_uncov = num_uncov

        for step in range(20000):
            if time.time() > deadline - 2.0:
                break
            node_count[0] += 1

            if len(current_blocks) == 0:
                break

            # Pick a random block to try to eliminate
            bi_idx = random.randrange(len(current_blocks))
            bi = current_blocks[bi_idx]

            # Find t-subsets that ONLY bi covers
            essential = []
            for ti in block_cover[bi]:
                if cur_counts[ti] == 1:
                    essential.append(ti)

            if not essential:
                # bi is redundant, remove it
                current_blocks.pop(bi_idx)
                for ti in block_cover[bi]:
                    cur_counts[ti] -= 1
                    if cur_counts[ti] == 0:
                        cur_covered[ti] = False
                        cur_num_uncov += 1
                if len(current_blocks) < best_len_local:
                    best_len_local = len(current_blocks)
                    best_sol = list(current_blocks)
                continue

            # Try to find a replacement block that covers all essential t-subsets
            if len(essential) == 1:
                candidates = t_to_blocks[essential[0]]
            else:
                # Intersection of covering blocks for all essential
                min_ti = min(essential, key=lambda ti: len(t_to_blocks[ti]))
                candidates = set(t_to_blocks[min_ti])
                for ti in essential[1:]:
                    candidates &= set(t_to_blocks[ti])
                    if not candidates:
                        break

            candidates.discard(bi)
            if not candidates:
                continue

            # Pick a candidate that maximizes the number of currently-uncovered or low-count subsets
            best_cand = -1
            best_score = -1
            for cand in candidates:
                score = 0
                for ti in block_cover[cand]:
                    if cur_counts[ti] <= 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = cand

            if best_cand == -1:
                continue

            # Try the swap: remove bi, add best_cand
            # Check validity: after swap, all essential must still be covered
            # (guaranteed by construction since best_cand covers all essential)
            # But we need to make sure no OTHER t-subset becomes uncovered
            # (i.e., for every ti in block_cover[bi], either ti in block_cover[best_cand] or cur_counts[ti] > 1)
            valid = True
            for ti in block_cover[bi]:
                if cur_counts[ti] == 1 and ti not in block_cover[best_cand]:
                    # This ti would become uncovered
                    valid = False
                    break

            if not valid:
                continue

            # Perform swap
            for ti in block_cover[bi]:
                cur_counts[ti] -= 1
            for ti in block_cover[best_cand]:
                cur_counts[ti] += 1
            current_blocks[bi_idx] = best_cand

            # After swap, try to remove any now-redundant blocks
            removed_any = True
            while removed_any and len(current_blocks) > 0:
                removed_any = False
                for i in range(len(current_blocks) - 1, -1, -1):
                    b = current_blocks[i]
                    is_red = True
                    for ti in block_cover[b]:
                        if cur_counts[ti] == 1:
                            is_red = False
                            break
                    if is_red:
                        current_blocks.pop(i)
                        for ti in block_cover[b]:
                            cur_counts[ti] -= 1
                        removed_any = True
                        if len(current_blocks) < best_len_local:
                            best_len_local = len(current_blocks)
                            best_sol = list(current_blocks)

        return best_sol, best_len_local

    # Main: greedy start, then B&B-inspired local search
    best_solution = greedy_cover()
    best_len = len(best_solution)

    # Multiple restarts with different greedy orderings
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
        # Perturb the greedy by using a different rarity tie-breaking
        rng = random.Random(42 + restart)
        # Shuffle blocks within same rarity for diversity
        # Just re-run greedy with slight randomization in block selection
        covered = np.zeros(N_t, dtype=bool)
        selected = []
        num_uncov = N_t
        while num_uncov > 0:
            if time.time() > deadline - 3.0:
                break
            # Pick random uncovered subset weighted by rarity
            uncov_list = [ti for ti in rarity_order if not covered[ti]]
            if not uncov_list:
                break
            # Weighted random pick: rarer subsets more likely
            weights = [1.0 / len(t_to_blocks[ti]) for ti in uncov_list]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_ti = uncov_list[0]
            for ti, w in zip(uncov_list, weights):
                cum += w
                if r <= cum:
                    chosen_ti = ti
                    break
            # Pick block with max new coverage
            best_bi = -1
            best_gain = -1
            for bi in t_to_blocks[chosen_ti]:
                gain = 0
                for ti2 in block_cover[bi]:
                    if not covered[ti2]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            if best_bi == -1:
                best_bi = t_to_blocks[chosen_ti][0]
            selected.append(best_bi)
            for ti in block_cover[best_bi]:
                if not covered[ti]:
                    covered[ti] = True
                    num_uncov -= 1

        if len(selected) < best_len:
            best_len = len(selected)
            best_solution = selected

        # B&B-inspired local search on this solution
        if time.time() < deadline - 5.0:
            improved, improved_len = branch_and_bound(best_solution, best_len)
            if improved_len < best_len:
                best_len = improved_len
                best_solution = improved

    # Final local search on best solution
    if time.time() < deadline - 3.0 and best_solution:
        improved, improved_len = branch_and_bound(best_solution, best_len)
        if improved_len < best_len:
            best_len = improved_len
            best_solution = improved

    if best_solution is None:
        best_solution = greedy_cover()

    return [list(all_blocks[bi]) for bi in best_solution]
