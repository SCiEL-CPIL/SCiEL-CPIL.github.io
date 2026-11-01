import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute all t-subsets and k-blocks
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # Build incidence: for each block, list of t-subset indices it covers
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Precompute bitmask for each block over t-subsets (for fast coverage check)
    # Use numpy uint64 bitmasks; num_t_subsets = C(19,5) = 11628, so need multiple words
    num_words = (num_t_subsets + 63) // 64
    block_bitmasks = np.zeros((num_k_blocks, num_words), dtype=np.uint64)
    for j in range(num_k_blocks):
        for ts_idx in block_to_ts[j]:
            word = ts_idx >> 6
            bit = ts_idx & 63
            block_bitmasks[j, word] |= (1 << bit)

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

    def randomized_greedy(seed, top_k=5):
        """Greedy with random tie-breaking among top-k candidates."""
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 3.0:
                return None

            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break

            # Find all blocks with max gain
            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            # Pick randomly from top-k candidates (or all if fewer)
            pick_from = min(top_k, len(candidates))
            if pick_from < len(candidates):
                # Randomly choose which k to consider, then pick from those
                chosen = random.sample(list(candidates), pick_from)
                b_idx = chosen[random.randint(0, pick_from - 1)]
            else:
                b_idx = candidates[random.randint(0, len(candidates) - 1)]

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1

            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

        return sol

    def iterative_removal(blocks):
        """Iteratively remove blocks that are fully redundant."""
        if not blocks:
            return blocks

        # Build cover count
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1

        # Also track which block covers each t-subset (for quick redundancy check)
        # We'll use a list of block indices per t-subset
        ts_covers = [[] for _ in range(num_t_subsets)]
        for bi, b in enumerate(blocks):
            for ts in itertools.combinations(b, t):
                ts_covers[ts_index_map[ts]].append(bi)

        changed = True
        while changed:
            if time.time() > deadline - 2.0:
                break
            changed = False
            # Scan in reverse to avoid index issues
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                # Check if block i is redundant: every t-subset it covers
                # is covered by at least one other block
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    ts_idx = ts_index_map[ts]
                    # Count how many blocks in current solution cover this ts
                    if cover_count[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for ts in itertools.combinations(blocks[i], t):
                        cover_count[ts_index_map[ts]] -= 1
                    blocks.pop(i)
                    changed = True
                    break  # Restart scan after removal

        return blocks

    def local_search_swap(blocks, max_iters=500):
        """Try to swap a block for another that covers its unique subsets."""
        if not blocks:
            return blocks, False

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1

        sol_set = set(id(b) for b in blocks)  # Use identity for speed
        # Actually use tuple for set membership
        sol_tuples = set(tuple(b) for b in blocks)

        improved = False
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break

            i = random.randint(0, len(blocks) - 1)
            # Find unique t-subsets of block i
            unique_ts = []
            for ts in itertools.combinations(blocks[i], t):
                ts_idx = ts_index_map[ts]
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)

            if not unique_ts:
                # Block is fully redundant, just remove it
                for ts in itertools.combinations(blocks[i], t):
                    cover_count[ts_index_map[ts]] -= 1
                blocks.pop(i)
                improved = True
                break

            # Find a block that covers all unique_ts
            # Use intersection of ts_to_blocks
            candidates = set(ts_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(ts_to_blocks[ts_idx])
                if not candidates:
                    break

            if not candidates:
                continue

            # Filter out current block and already-in-solution blocks
            current_idx = all_k_blocks.index(blocks[i]) if blocks[i] in all_k_blocks else -1
            valid = [c for c in candidates
                     if c != current_idx and tuple(all_k_blocks[c]) not in sol_tuples]
            if not valid:
                continue

            new_idx = random.choice(valid)
            new_block = list(all_k_blocks[new_idx])

            # Perform swap
            for ts in itertools.combinations(blocks[i], t):
                cover_count[ts_index_map[ts]] -= 1
            for ts in itertools.combinations(new_block, t):
                cover_count[ts_index_map[ts]] += 1

            blocks[i] = new_block
            sol_tuples.discard(tuple(blocks[i]))  # old is gone
            sol_tuples.add(tuple(new_block))
            improved = True
            break

        return blocks, improved

    # Main search
    best_solution = None
    best_len = float('inf')

    # Phase 1: Try multiple randomized greedy constructions
    num_constructions = 30
    for i in range(num_constructions):
        if time.time() > deadline - 10.0:
            break
        sol = randomized_greedy(i, top_k=random.choice([3, 5, 10]))
        if sol is not None:
            # Apply iterative removal
            sol = iterative_removal(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol

    if best_solution is None:
        # Fallback: deterministic greedy
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        while True:
            max_gain = int(np.max(gain))
            if max_gain <= 0:
                break
            b_idx = int(np.argmax(gain))
            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True
            gain[b_idx] = -1
            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1
        best_solution = sol
        best_len = len(sol)

    # Phase 2: Local search on best solution
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        best_solution, improved = local_search_swap(best_solution, max_iters=200)
        if not improved:
            break
        best_solution = iterative_removal(best_solution)
        if len(best_solution) < best_len:
            best_len = len(best_solution)

    # Phase 3: More aggressive iterative removal passes
    for _ in range(20):
        if time.time() > deadline - 2.0:
            break
        new_sol = iterative_removal(list(best_solution))
        if len(new_sol) < len(best_solution):
            best_solution = new_sol
            best_len = len(best_solution)
        else:
            break

    return best_solution
