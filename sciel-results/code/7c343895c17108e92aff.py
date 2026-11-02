import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 143.0

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

    all_t_subsets = list(itertools.combinations(range(v), t))
    num_ts = len(all_t_subsets)
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_blocks)

    ts_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Build per-block t-subset index lists
    block_ts_lists = []
    for block in all_k_blocks:
        lst = []
        for ts in itertools.combinations(block, t):
            lst.append(ts_index[ts])
        block_ts_lists.append(lst)

    # Build per-t-subset block index lists
    ts_to_blocks = [[] for _ in range(num_ts)]
    for j, lst in enumerate(block_ts_lists):
        for ts_idx in lst:
            ts_to_blocks[ts_idx].append(j)

    # Bitmask: 185 uint64 words covering 11628 bits
    num_words = (num_ts + 63) // 64
    block_bitmasks = np.zeros((num_blocks, num_words), dtype=np.uint64)
    for j, lst in enumerate(block_ts_lists):
        for ts_idx in lst:
            w = ts_idx >> 6
            b = ts_idx & 63
            block_bitmasks[j, w] |= (1 << b)

    # Precompute: for each t-subset, the list of block indices (as numpy array)
    ts_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    def count_bits(mask):
        return int(np.sum(mask))  # mask is uint64 array; sum of bit counts
        # Actually need bit_count per word then sum
        # np.bit_count works on uint64 arrays in numpy >= 1.24
        # Fallback: use lookup or manual

    def bit_count_arr(arr):
        # arr: uint64 array of shape (num_words,)
        # Count bits in each word, sum
        if hasattr(np, 'bit_count'):
            return int(np.sum(np.bit_count(arr)))
        # Fallback: manual
        total = 0
        for x in arr:
            total += bin(x).count('1')
        return total

    def run_greedy(seed, explore_prob=0.0):
        random.seed(seed)
        gain = np.full(num_blocks, C(k, t), dtype=np.int32)
        is_selected = np.zeros(num_blocks, dtype=bool)
        covered_mask = np.zeros(num_words, dtype=np.uint64)
        sol = []

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = int(np.max(gain[~is_selected])) if np.any(~is_selected) else 0
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            if explore_prob > 0 and random.random() < explore_prob:
                explore_cands = np.where((gain >= max_gain - 3) & (~is_selected))[0]
                if len(explore_cands) > 0:
                    b_idx = int(explore_cands[random.randint(0, len(explore_cands) - 1)])
                else:
                    b_idx = int(candidates[random.randint(0, len(candidates) - 1)])
            else:
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])

            sol.append(list(all_k_blocks[b_idx]))
            is_selected[b_idx] = True

            # Compute newly covered t-subsets: bits in block that are not yet covered
            bm = block_bitmasks[b_idx]
            newly = bm & (~covered_mask)
            new_count = bit_count_arr(newly)

            if new_count == 0:
                gain[b_idx] = -1
                continue

            covered_mask |= bm

            # Update gains: for each block, gain decreases by number of newly covered ts it contains
            # Vectorized: for each block j, decrement = bit_count(block_bitmasks[j] & newly)
            # Do this in chunks to avoid huge memory
            if new_count > 0:
                # Only update unselected blocks
                unselected = np.where(~is_selected)[0]
                if len(unselected) > 0:
                    # Process in chunks of 1000 blocks
                    chunk_size = 1000
                    for start in range(0, len(unselected), chunk_size):
                        chunk = unselected[start:start + chunk_size]
                        if len(chunk) == 0:
                            continue
                        # For each block in chunk, count bits in (block_bitmasks[chunk] & newly)
                        # block_bitmasks[chunk] has shape (len(chunk), num_words)
                        # newly has shape (num_words,)
                        # Broadcasting: (len(chunk), num_words) & (num_words,)
                        masked = block_bitmasks[chunk] & newly
                        # Count bits per row
                        if hasattr(np, 'bit_count'):
                            decrements = np.sum(np.bit_count(masked), axis=1)
                        else:
                            decrements = np.array([bit_count_arr(masked[i]) for i in range(len(chunk))], dtype=np.int32)
                        gain[chunk] -= decrements
                        gain[chunk] = np.maximum(gain[chunk], -1)

            gain[b_idx] = -1

        return sol

    # Build a maximal t-packing for scaffold
    def build_packing(seed):
        random.seed(seed)
        random.shuffle(all_k_blocks)
        packing = []
        used_ts = set()
        for block in all_k_blocks:
            if time.time() > deadline - 10.0:
                break
            bts = set()
            for ts in itertools.combinations(block, t):
                if ts in used_ts:
                    break
                bts.add(ts)
            else:
                # All t-subsets are unused
                packing.append(block)
                used_ts.update(bts)
                if len(packing) >= 300:
                    break
        return packing

    # Initial greedy
    best_solution = run_greedy(0)
    if best_solution is None:
        best_solution = [list(all_k_blocks[0])]
    best_len = len(best_solution)

    # Multiple randomized greedy runs
    num_runs = 120
    for i in range(1, num_runs):
        if time.time() > deadline - 20.0:
            break
        explore = 0.0 if i < 40 else (0.05 if i < 80 else 0.1)
        sol = run_greedy(i, explore_prob=explore)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol

    def optimize_redundancy(blocks):
        if not blocks:
            return blocks
        cover_count = np.zeros(num_ts, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_ts_lists[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index[ts])
                block_ts_indices.append(ts_list)

        for ts_list in block_ts_indices:
            for ts_idx in ts_list:
                cover_count[ts_idx] += 1

        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 3.0:
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

    def local_search_swap(blocks, seed=0):
        random.seed(seed)
        if not blocks:
            return blocks, False

        cover_count = np.zeros(num_ts, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_ts_lists[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index[ts])
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
        n = len(blocks)
        order = list(range(n))
        random.shuffle(order)

        for i in order:
            if time.time() > deadline - 3.0:
                return blocks, improved

            unique_ts = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)

            if not unique_ts:
                continue

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

            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])
            old_block_idx = get_block_idx(blocks[i])

            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1
            for ts_idx in block_to_ts[new_block_idx] if hasattr(block_to_ts, '__getitem__') else block_ts_lists[new_block_idx]:
                cover_count[ts_idx] += 1

            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)
            block_ts_indices[i] = block_ts_lists[new_block_idx]
            improved = True

        return blocks, improved

    # Iterative compression: remove one block, re-cover its unique t-subsets
    def iterative_compression(blocks, seed=0):
        random.seed(seed)
        if len(blocks) < 2:
            return blocks, False

        cover_count = np.zeros(num_ts, dtype=np.int32)
        block_ts_indices = []
        for b in blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                block_ts_indices.append(block_ts_lists[idx])
            else:
                ts_list = []
                for ts in itertools.combinations(b, t):
                    ts_list.append(ts_index[ts])
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
        n = len(blocks)
        order = list(range(n))
        random.shuffle(order)

        for i in order:
            if time.time() > deadline - 3.0:
                return blocks, improved

            # Find t-subsets uniquely covered by block i
            unique_ts = []
            for ts_idx in block_ts_indices[i]:
                if cover_count[ts_idx] == 1:
                    unique_ts.append(ts_idx)

            if not unique_ts:
                # Block is fully redundant; remove it
                blocks.pop(i)
                for ts_idx in block_ts_indices[i]:
                    cover_count[ts_idx] -= 1
                block_ts_indices.pop(i)
                sol_block_indices.discard(get_block_idx(blocks[i]) if i < len(blocks) else -1)
                improved = True
                # Adjust indices
                order = [x if x < i else x - 1 for x in order if x != i]
                n = len(blocks)
                continue

            # Try to find a single block that covers all unique_ts and is not in solution
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

            new_block_idx = random.choice(valid_candidates)
            new_block = list(all_k_blocks[new_block_idx])

            # Remove old block
            old_block_idx = get_block_idx(blocks[i])
            for ts_idx in block_ts_indices[i]:
                cover_count[ts_idx] -= 1

            # Add new block
            for ts_idx in block_ts_lists[new_block_idx]:
                cover_count[ts_idx] += 1

            blocks[i] = new_block
            sol_block_indices.discard(old_block_idx)
            sol_block_indices.add(new_block_idx)
            block_ts_indices[i] = block_ts_lists[new_block_idx]
            improved = True

        return blocks, improved

    # Polish loop
    for iteration in range(80):
        if time.time() > deadline - 5.0:
            break

        # Try iterative compression
        best_solution, improved = iterative_compression(best_solution, seed=iteration)
        if improved:
            best_solution = optimize_redundancy(best_solution)
            if len(best_solution) < best_len:
                best_len = len(best_solution)
        else:
            # Try swap-based local search
            best_solution, improved = local_search_swap(best_solution, seed=iteration)
            if improved:
                best_solution = optimize_redundancy(best_solution)
                if len(best_solution) < best_len:
                    best_len = len(best_solution)
            else:
                # Try a fresh greedy from a random subset
                if time.time() > deadline - 10.0:
                    break
                fresh = run_greedy(1000 + iteration, explore_prob=0.1)
                if fresh is not None:
                    fresh = optimize_redundancy(fresh)
                    if len(fresh) < best_len:
                        best_len = len(fresh)
                        best_solution = fresh
                    else:
                        break

    return best_solution
