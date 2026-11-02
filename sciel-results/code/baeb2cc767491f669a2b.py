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

            for ts_idx in block_to_ts[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

        return sol

    # 3. Polish: remove redundant blocks
    def polish_blocks(block_indices):
        if not block_indices:
            return block_indices

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts[b]:
                cover_count[ts_idx] += 1

        i = len(block_indices) - 1
        while i >= 0:
            if time.time() > deadline - 1.0:
                break
            b = block_indices[i]
            can_remove = True
            for ts_idx in block_to_ts[b]:
                if cover_count[ts_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                for ts_idx in block_to_ts[b]:
                    cover_count[ts_idx] -= 1
                block_indices.pop(i)
            else:
                i -= 1
        return block_indices

    # 4. 1-exchange local search: try replacing one block with another to reduce redundancy
    def local_search_swap(block_indices):
        if not block_indices:
            return block_indices, False

        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in block_indices:
            for ts_idx in block_to_ts[b]:
                cover_count[ts_idx] += 1

        improved = False
        for i in range(len(block_indices)):
            if time.time() > deadline - 1.0:
                break

            b = block_indices[i]
            # Find t-subsets uniquely covered by block b
            unique_ts = []
            for ts_idx in block_to_ts[b]:
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

            candidates.discard(b)

            # Filter: candidate must not introduce new uncovered t-sets
            # Actually, we just need a valid replacement that maintains coverage
            # Let's check if replacing b with c maintains full coverage
            for c in list(candidates):
                # Check coverage: after removing b and adding c, all t-sets still covered
                # t-sets only covered by b must also be covered by c (already ensured by unique_ts check)
                # t-sets covered by b and others: fine
                # t-sets only covered by c (if any) and not by b: need to check
                # Actually, we need to ensure that removing b doesn't leave any t-set uncovered
                # which is already handled by unique_ts. Adding c might cover some, but we don't
                # lose coverage by adding c. So the only risk is if c doesn't cover all unique_ts,
                # but we already filtered for that.
                # So any c in candidates works. Pick the first one.
                # Remove b, add c
                for ts_idx in block_to_ts[b]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_to_ts[c]:
                    cover_count[ts_idx] += 1
                block_indices[i] = c
                improved = True
                break

            if improved:
                break  # Restart the loop after a change

        return block_indices, improved

    # 5. Main search loop
    best_blocks = None
    best_len = float('inf')

    # Initial deterministic greedy
    sol = run_greedy(deterministic=True)
    if sol is not None:
        sol = polish_blocks(sol)
        sol, _ = local_search_swap(sol)
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
            sol, _ = local_search_swap(sol)
            sol = polish_blocks(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]
        elif len(sol) <= best_len + 5:
            # Close enough, try local search
            sol, _ = local_search_swap(sol)
            sol = polish_blocks(sol)
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]

    # Final polish on best solution
    if best_blocks is not None:
        best_blocks = polish_blocks(best_blocks)
        best_blocks, _ = local_search_swap(best_blocks)
        best_blocks = polish_blocks(best_blocks)

    if best_blocks is None:
        return []

    result = [list(all_k_blocks[b]) for b in best_blocks]
    return result
