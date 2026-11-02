import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # Incidence structures:
    # block_to_ts: list of lists, block_to_ts[b] = list of t-subset indices covered by block b
    # ts_to_blocks: list of lists, ts_to_blocks[ts] = list of block indices covering t-subset ts
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Convert to numpy arrays for faster access
    block_to_ts_np = [np.array(ts_list, dtype=np.int32) for ts_list in block_to_ts]
    ts_to_blocks_np = [np.array(blks, dtype=np.int32) for blks in ts_to_blocks]

    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # 2. Greedy construction (deterministic first pass)
    def run_greedy(seed=None, deterministic=False):
        if seed is not None:
            random.seed(seed)
        gain = np.full(num_k_blocks, len(block_to_ts[0]), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 5.0:
                return None

            max_gain = np.max(gain)
            if max_gain <= 0:
                break

            if deterministic:
                b_idx = int(np.argmax(gain))
            else:
                candidates = np.where((gain == max_gain) & (~is_selected))[0]
                if len(candidates) == 0:
                    break
                b_idx = int(candidates[random.randint(0, len(candidates) - 1)])

            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1

            for ts_idx in block_to_ts_np[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks_np[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

        return sol

    # 3. Polish: remove redundant blocks
    def polish_blocks(block_indices):
        if not block_indices:
            return block_indices

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts_np[b]:
                cover_count[ts_idx] += 1

        i = len(block_indices) - 1
        while i >= 0:
            if time.time() > deadline - 1.0:
                break
            b = block_indices[i]
            can_remove = True
            for ts_idx in block_to_ts_np[b]:
                if cover_count[ts_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts_idx in block_to_ts_np[b]:
                    cover_count[ts_idx] -= 1
                block_indices.pop(i)
            else:
                i -= 1
        return block_indices

    # 4. Improved 1-exchange local search: try replacing one block with another to reduce redundancy
    def local_search_swap(block_indices, max_iters=50):
        if not block_indices:
            return block_indices, False

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts_np[b]:
                cover_count[ts_idx] += 1

        improved = False
        iters = 0
        while iters < max_iters:
            if time.time() > deadline - 1.0:
                break
            iters += 1

            found_swap = False
            for i in range(len(block_indices)):
                if time.time() > deadline - 1.0:
                    return block_indices, improved

                b = block_indices[i]
                # Find t-subsets uniquely covered by block b
                unique_ts = []
                for ts_idx in block_to_ts_np[b]:
                    if cover_count[ts_idx] == 1:
                        unique_ts.append(ts_idx)

                if not unique_ts:
                    continue

                # Find candidate blocks that cover ALL unique_ts
                candidates = set(ts_to_blocks_np[unique_ts[0]].tolist())
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks_np[ts_idx].tolist())
                    if not candidates:
                        break

                if not candidates:
                    continue

                candidates.discard(b)

                if not candidates:
                    continue

                # Pick the best candidate: one that maximizes redundancy gain
                best_c = None
                best_slack = -1
                for c in candidates:
                    # Calculate slack: how many extra t-subsets c covers that are already covered
                    # A good replacement covers unique_ts and adds minimal new coverage
                    # Actually, we want to minimize the "cost" - but since we're swapping,
                    # the key is that c covers all unique_ts. Among those, prefer ones
                    # that overlap more with existing blocks (higher redundancy)
                    overlap = 0
                    for ts_idx in block_to_ts_np[c]:
                        if cover_count[ts_idx] > 1:
                            overlap += 1
                    if overlap > best_slack:
                        best_slack = overlap
                        best_c = c

                if best_c is None:
                    continue

                # Apply the swap
                for ts_idx in block_to_ts_np[b]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts_np[best_c]:
                    cover_count[ts_idx] += 1
                block_indices[i] = best_c
                improved = True
                found_swap = True
                break

            if not found_swap:
                break

        return block_indices, improved

    # 5. 2-exchange local search: try removing two blocks and adding two different blocks
    def local_search_2swap(block_indices, max_iters=20):
        if len(block_indices) < 4:
            return block_indices, False

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts_np[b]:
                cover_count[ts_idx] += 1

        improved = False
        iters = 0
        while iters < max_iters:
            if time.time() > deadline - 1.0:
                break
            iters += 1

            found_swap = False
            n = len(block_indices)
            for i in range(n):
                if time.time() > deadline - 1.0:
                    return block_indices, improved
                for j in range(i + 1, n):
                    if time.time() > deadline - 1.0:
                        return block_indices, improved

                    b1 = block_indices[i]
                    b2 = block_indices[j]

                    # Find t-subsets uniquely covered by b1 or b2
                    unique_ts = set()
                    for ts_idx in block_to_ts_np[b1]:
                        if cover_count[ts_idx] == 1:
                            unique_ts.add(ts_idx)
                    for ts_idx in block_to_ts_np[b2]:
                        if cover_count[ts_idx] == 1:
                            unique_ts.add(ts_idx)

                    if not unique_ts:
                        continue

                    # Find pairs of candidate blocks that together cover all unique_ts
                    # This is expensive, so limit to a few candidates
                    unique_ts_list = list(unique_ts)
                    if len(unique_ts_list) > 10:
                        continue

                    # Get candidate blocks for each unique ts
                    cand_sets = []
                    for ts_idx in unique_ts_list:
                        cand_sets.append(set(ts_to_blocks_np[ts_idx].tolist()))

                    # Intersect to get blocks that cover at least one unique ts
                    all_cands = set()
                    for cs in cand_sets:
                        all_cands.update(cs)
                    all_cands.discard(b1)
                    all_cands.discard(b2)

                    if len(all_cands) > 50:
                        all_cands = set(random.sample(list(all_cands), 50))

                    # Try pairs
                    cand_list = list(all_cands)
                    for a in range(len(cand_list)):
                        if time.time() > deadline - 1.0:
                            return block_indices, improved
                        for c in cand_list[a+1:]:
                            # Check if {cand_list[a], c} covers all unique_ts
                            covers_all = True
                            for ts_idx in unique_ts_list:
                                if cand_list[a] not in ts_to_blocks_np[ts_idx] and c not in ts_to_blocks_np[ts_idx]:
                                    covers_all = False
                                    break
                            if not covers_all:
                                continue

                            # Apply the 2-swap
                            for ts_idx in block_to_ts_np[b1]:
                                cover_count[ts_idx] -= 1
                            for ts_idx in block_to_ts_np[b2]:
                                cover_count[ts_idx] -= 1
                            for ts_idx in block_to_ts_np[cand_list[a]]:
                                cover_count[ts_idx] += 1
                            for ts_idx in block_to_ts_np[c]:
                                cover_count[ts_idx] += 1
                            block_indices[i] = cand_list[a]
                            block_indices[j] = c
                            improved = True
                            found_swap = True
                            break
                        if found_swap:
                            break
                    if found_swap:
                        break
                if found_swap:
                    break

            if not found_swap:
                break

        return block_indices, improved

    # 6. Main search loop
    best_blocks = None
    best_len = float('inf')

    # Initial deterministic greedy
    sol = run_greedy(deterministic=True)
    if sol is not None:
        sol = polish_blocks(sol)
        sol, _ = local_search_swap(sol, max_iters=30)
        sol = polish_blocks(sol)
        sol, _ = local_search_2swap(sol, max_iters=10)
        sol = polish_blocks(sol)
        best_blocks = sol[:]
        best_len = len(sol)

    # Randomized greedy restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        sol = run_greedy(seed=i)
        if sol is None:
            break

        sol = polish_blocks(sol)
        if len(sol) < best_len:
            sol, _ = local_search_swap(sol, max_iters=30)
            sol = polish_blocks(sol)
            sol, _ = local_search_2swap(sol, max_iters=10)
            sol = polish_blocks(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]
        elif len(sol) <= best_len + 5:
            # Close enough, try local search
            sol, _ = local_search_swap(sol, max_iters=30)
            sol = polish_blocks(sol)
            sol, _ = local_search_2swap(sol, max_iters=10)
            sol = polish_blocks(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]

    # Final polish on best solution
    if best_blocks is not None:
        best_blocks = polish_blocks(best_blocks)
        best_blocks, _ = local_search_swap(best_blocks, max_iters=50)
        best_blocks = polish_blocks(best_blocks)
        best_blocks, _ = local_search_2swap(best_blocks, max_iters=20)
        best_blocks = polish_blocks(best_blocks)
        # One more pass
        best_blocks = polish_blocks(best_blocks)
        best_blocks, _ = local_search_swap(best_blocks, max_iters=30)
        best_blocks = polish_blocks(best_blocks)

    if best_blocks is None:
        return []

    result = [list(all_k_blocks[b]) for b in best_blocks]
    return result
