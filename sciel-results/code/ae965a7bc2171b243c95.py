import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin

    # Helper
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

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # block_to_ts: for each block, list of t-subset indices it covers
    block_to_ts = []
    for block in all_k_blocks:
        ts_list = []
        for ts in itertools.combinations(block, t):
            ts_list.append(ts_index_map[ts])
        block_to_ts.append(ts_list)

    # ts_to_blocks: for each t-subset, list of block indices covering it
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, ts_list in enumerate(block_to_ts):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(j)

    # Precompute as numpy arrays for speed
    # block_to_ts_arr: 2D array with padding, or use list of arrays
    block_to_ts_arr = [np.array(ts_list, dtype=np.int32) for ts_list in block_to_ts]

    def get_block_idx(block):
        # Use a dictionary for fast lookup
        pass

    # Build block tuple to index map
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # Randomized greedy
    def randomized_greedy(seed):
        random.seed(seed)
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []

        while True:
            if time.time() > deadline - 3.0:
                return None

            max_gain = np.max(gain)
            if max_gain <= 0:
                break

            candidates = np.where((gain == max_gain) & (~is_selected))[0]
            if len(candidates) == 0:
                break

            b_idx = candidates[random.randint(0, len(candidates) - 1)]

            sol.append(b_idx)
            is_selected[b_idx] = True
            gain[b_idx] = -1

            for ts_idx in block_to_ts_arr[b_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for nb_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[nb_idx]:
                            gain[nb_idx] -= 1

        return sol

    # Simulated annealing
    def simulated_annealing(initial_blocks, time_budget):
        """
        initial_blocks: list of block indices
        Returns list of block indices (may be shorter)
        """
        # State: set of selected block indices
        selected = set(initial_blocks)
        n_blocks = len(selected)

        # Cover count array
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            for ts_idx in block_to_ts_arr[b_idx]:
                cover_count[ts_idx] += 1

        # Current cost: number of uncovered t-subsets
        current_cost = np.sum(cover_count == 0)

        best_cost = current_cost
        best_selected = set(selected)

        # Temperature schedule
        T0 = 5.0
        T_end = 0.01
        # We'll use geometric cooling
        # Determine number of steps we can do
        # Each step is fast, but we need to check time

        # Precompute for each block, the t-subsets it covers (already have block_to_ts_arr)

        # For efficiency, track which t-subsets are "critical" (covered exactly once)
        # We'll just recompute cost incrementally

        # Moves:
        # 1. Remove a random block from selected
        # 2. Add a random block not in selected
        # 3. Swap: remove one, add another

        # To speed up, precompute for each t-subset, the list of blocks covering it (ts_to_blocks)

        # We'll do a fixed number of iterations or until time runs out
        T = T0
        cooling_rate = 0.9995  # Will adjust based on available time

        # Estimate steps: each step ~ 10-50us, so 112s / 30us ~ 3.7M steps
        # But we have multiple restarts, so per restart maybe 200k steps

        while time.time() < time_budget:
            # Choose move type
            move_type = random.random()
            if move_type < 0.3 and len(selected) > 1:
                # Remove
                b_idx = random.choice(list(selected))
                # Compute delta cost
                # Remove block: for each ts it covers, decrement cover_count
                # If any ts goes from 1 to 0, cost increases by 1
                # If any ts goes from 2 to 1, cost unchanged
                # If any ts goes from >2 to >1, cost unchanged

                # Actually, let's just compute the change
                delta_cost = 0
                for ts_idx in block_to_ts_arr[b_idx]:
                    if cover_count[ts_idx] == 1:
                        delta_cost += 1
                    # else: cover_count[ts_idx] >= 2, so removing doesn't make it uncovered

                # Apply
                for ts_idx in block_to_ts_arr[b_idx]:
                    cover_count[ts_idx] -= 1
                selected.discard(b_idx)
                new_cost = current_cost + delta_cost

                # Accept?
                if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / max(T, 1e-10)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = set(selected)
                else:
                    # Revert
                    for ts_idx in block_to_ts_arr[b_idx]:
                        cover_count[ts_idx] += 1
                    selected.add(b_idx)

            elif move_type < 0.6:
                # Add
                # Pick a random block not in selected
                # To make it more likely to be useful, pick from blocks that cover some uncovered ts
                # For simplicity, just pick random
                b_idx = random.randint(0, num_k_blocks - 1)
                if b_idx in selected:
                    continue

                # Compute delta cost
                delta_cost = 0
                for ts_idx in block_to_ts_arr[b_idx]:
                    if cover_count[ts_idx] == 0:
                        delta_cost -= 1

                # Apply
                for ts_idx in block_to_ts_arr[b_idx]:
                    cover_count[ts_idx] += 1
                selected.add(b_idx)
                new_cost = current_cost + delta_cost

                if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / max(T, 1e-10)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = set(selected)
                else:
                    # Revert
                    for ts_idx in block_to_ts_arr[b_idx]:
                        cover_count[ts_idx] -= 1
                    selected.discard(b_idx)

            else:
                # Swap
                if len(selected) == 0:
                    continue
                b_remove = random.choice(list(selected))
                b_add = random.randint(0, num_k_blocks - 1)
                if b_add in selected:
                    continue
                if b_add == b_remove:
                    continue

                # Compute delta cost
                delta_cost = 0
                # Remove b_remove
                for ts_idx in block_to_ts_arr[b_remove]:
                    if cover_count[ts_idx] == 1:
                        delta_cost += 1
                # Add b_add
                # But we need to account for the fact that b_remove is being removed
                # So for ts in b_add, the effective cover count is cover_count[ts] - (1 if ts in b_remove else 0)
                for ts_idx in block_to_ts_arr[b_add]:
                    eff_count = cover_count[ts_idx]
                    # Check if b_remove also covers this ts
                    # This is expensive. Let's just compute net effect differently.
                    # Actually, let's just apply both and compute cost change.
                    pass

                # Simpler: just apply both moves and compute cost
                # Remove b_remove
                for ts_idx in block_to_ts_arr[b_remove]:
                    cover_count[ts_idx] -= 1
                selected.discard(b_remove)

                # Add b_add
                for ts_idx in block_to_ts_arr[b_add]:
                    cover_count[ts_idx] += 1
                selected.add(b_add)

                new_cost = np.sum(cover_count == 0)

                if new_cost < current_cost or random.random() < np.exp(-(new_cost - current_cost) / max(T, 1e-10)):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = set(selected)
                else:
                    # Revert
                    for ts_idx in block_to_ts_arr[b_add]:
                        cover_count[ts_idx] -= 1
                    selected.discard(b_add)
                    for ts_idx in block_to_ts_arr[b_remove]:
                        cover_count[ts_idx] += 1
                    selected.add(b_remove)

            # Cool
            T *= cooling_rate
            if T < T_end:
                T = T_end

        return list(best_selected)

    # Find best initial solution via randomized greedy
    best_initial = None
    best_initial_len = float('inf')

    # Run several randomized greedy to find good starting points
    num_greedy_runs = 20
    for i in range(num_greedy_runs):
        if time.time() > deadline - 30.0:
            break
        sol = randomized_greedy(i)
        if sol is not None and len(sol) < best_initial_len:
            best_initial_len = len(sol)
            best_initial = sol

    if best_initial is None:
        # Fallback: deterministic greedy
        gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
        is_uncovered = np.full(num_t_subsets, True, dtype=bool)
        is_selected = np.full(num_k_blocks, False, dtype=bool)
        sol = []
        while True:
            if time.time() > deadline - 2.0:
                break
            max_idx = np.argmax(gain)
            if gain[max_idx] <= 0:
                break
            sol.append(max_idx)
            is_selected[max_idx] = True
            gain[max_idx] = -1
            for ts_idx in block_to_ts_arr[max_idx]:
                if is_uncovered[ts_idx]:
                    is_uncovered[ts_idx] = False
                    for b_idx in ts_to_blocks[ts_idx]:
                        if not is_selected[b_idx]:
                            gain[b_idx] -= 1
        best_initial = sol
        best_initial_len = len(sol)

    # Run simulated annealing from best_initial
    # Give it most of the remaining time
    sa_time_budget = deadline - 5.0
    sa_result = simulated_annealing(best_initial, sa_time_budget)

    # Convert to list of blocks
    result = [list(all_k_blocks[b_idx]) for b_idx in sa_result]

    # Ensure distinctness (should be, but just in case)
    # Remove duplicates
    seen = set()
    unique_result = []
    for block in result:
        tblock = tuple(block)
        if tblock not in seen:
            seen.add(tblock)
            unique_result.append(block)

    return unique_result
