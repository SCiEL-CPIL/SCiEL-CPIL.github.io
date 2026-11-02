import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Build block_to_ts (list of t-subset indices for each block)
    # and ts_to_blocks (list of block indices for each t-subset)
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]

    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Convert to numpy arrays for fast operations
    block_to_ts_np = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def run_greedy(seed, explore_prob=0.0):
        """Randomized greedy set cover with optional exploration."""
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            if explore_prob > 0 and random.random() < explore_prob:
                # Explore: pick from blocks with gain >= max_gain - 3
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            # Identify newly covered t-subsets
            bts = block_to_ts_np[b_idx]
            newly_covered_mask = is_uncovered[bts]
            newly_covered = bts[newly_covered_mask]

            if len(newly_covered) == 0:
                continue

            is_uncovered[newly_covered] = False

            # Vectorized gain update using bincount
            blocks_to_decrement = []
            for ts_idx in newly_covered:
                blocks_to_decrement.extend(ts_to_blocks[ts_idx])

            if blocks_to_decrement:
                counts = np.bincount(blocks_to_decrement, minlength=num_k_blocks)
                gain -= counts
                gain[is_selected] = -1000

        return sol

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # Initial deterministic greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Run multiple randomized greedy with varying exploration
    num_runs = 200
    for i in range(1, num_runs):
        if time.time() > deadline - 20.0:
            break

        # Vary exploration probability
        if i < 60:
            explore = 0.0
        elif i < 120:
            explore = 0.1
        else:
            explore = 0.2

        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    def optimize_redundancy(blocks):
        """Remove redundant blocks (those with no unique t-subsets)."""
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)

        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)

        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks

                redundant = True
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break

                if redundant:
                    blocks.pop(i)
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    block_ts_indices.pop(i)
                    changed = True
                    break

        return blocks

    def local_search_swap(blocks, max_iterations=50):
        """Swap-based local search: replace blocks with alternatives covering same unique t-subsets."""
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)

        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_to_ts[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index_map[ts])
                block_ts_indices.append(ts_list)

        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1

        sol_block_indices = set()
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                sol_block_indices.add(idx)

        improved = False
        for iteration in range(max_iterations):
            if time.time() > deadline - 5.0:
                break

            any_swap = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks, improved

                # Find t-subsets only covered by this block
                unique_ts = []
                for ts_idx in block_ts_indices[i]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)

                if not unique_ts:
                    # Block is redundant, remove it
                    for ts_idx in block_ts_indices[i]:
                        cover_count[ts_idx] -= 1
                    blocks.pop(i)
                    block_ts_indices.pop(i)
                    old_idx = get_block_idx(blocks[i-1]) if i > 0 else -1
                    # Actually we popped, so adjust
                    any_swap = True
                    improved = True
                    break

                # Find candidate blocks that cover ALL unique_ts
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break

                if not candidates:
                    continue

                valid_candidates = [c for c in candidates if c not in sol_block_indices]
                if not valid_candidates:
                    continue

                # Pick a random candidate
                new_block_idx = random.choice(valid_candidates)
                new_block = list(all_k_blocks[new_block_idx])

                old_block = blocks[i]
                old_block_idx = get_block_idx(old_block)

                # Update cover counts: remove old block's contribution, add new block's
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] -= 1

                new_ts_list = block_to_ts[new_block_idx]
                for ts_idx in new_ts_list:
                    cover_count[ts_idx] += 1

                blocks[i] = new_block
                block_ts_indices[i] = new_ts_list
                sol_block_indices.discard(old_block_idx)
                sol_block_indices.add(new_block_idx)

                any_swap = True
                improved = True

            if not any_swap:
                break

        return blocks, improved

    # Apply redundancy removal and local search to best solution
    best_solution = optimize_redundancy(best_solution)
    best_solution, _ = local_search_swap(best_solution)
    best_len = len(best_solution)

    # Additional refinement: try more local search passes
    for _ in range(3):
        if time.time() > deadline - 10.0:
            break
        best_solution = optimize_redundancy(best_solution)
        best_solution, _ = local_search_swap(best_solution, max_iterations=30)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Return the final list of blocks
    return best_solution
