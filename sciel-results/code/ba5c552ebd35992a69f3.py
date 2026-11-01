import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

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
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    block_tuple_to_idx = {tuple(b): i for i, b in enumerate(all_k_blocks)}

    # Build incidence
    block_to_ts = []
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    for j, block in enumerate(all_k_blocks):
        ts_list = []
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_list.append(idx)
            ts_to_blocks[idx].append(j)
        block_to_ts.append(ts_list)

    # Greedy Phase (fast initial solution)
    gain = np.full(num_k_blocks, C(k, t), dtype=np.int32)
    is_uncovered = np.full(num_t_subsets, True, dtype=bool)
    is_selected = np.full(num_k_blocks, False, dtype=bool)
    solution_blocks = []

    while True:
        if time.time() > deadline - 20.0:
            break
        max_idx = np.argmax(gain)
        if gain[max_idx] <= 0:
            break
        solution_blocks.append(list(all_k_blocks[max_idx]))
        is_selected[max_idx] = True
        gain[max_idx] = -1
        for ts_idx in block_to_ts[max_idx]:
            if is_uncovered[ts_idx]:
                is_uncovered[ts_idx] = False
                for b_idx in ts_to_blocks[ts_idx]:
                    if not is_selected[b_idx]:
                        gain[b_idx] -= 1

    best_solution = solution_blocks
    best_len = len(solution_blocks)

    # Redundancy removal
    def optimize_redundancy(blocks):
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5.0:
                    return blocks
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[ts_index_map[ts]] == 1:
                        redundant = False
                        break
                if redundant:
                    block_to_remove = blocks[i]
                    blocks.pop(i)
                    for ts in itertools.combinations(block_to_remove, t):
                        cover_count[ts_index_map[ts]] -= 1
                    changed = True
                    break
        return blocks

    best_solution = optimize_redundancy(best_solution)
    best_len = len(best_solution)

    def get_block_idx(block):
        return block_tuple_to_idx.get(tuple(block), -1)

    # Simulated Annealing on binary selection vector
    # Objective: minimize num_selected + penalty * num_uncovered
    # State: selected (bool array), cover_count (int array)
    # Move: flip one bit (turn on or off a block)
    
    def simulated_annealing(initial_blocks, max_time=80.0):
        # Initialize state
        selected = np.zeros(num_k_blocks, dtype=bool)
        for b in initial_blocks:
            idx = get_block_idx(b)
            if idx >= 0:
                selected[idx] = True
        
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in initial_blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        
        num_uncovered = int(np.sum(cover_count == 0))
        num_selected = int(np.sum(selected))
        penalty = 10000  # Large penalty to ensure coverage is prioritized
        
        def current_cost():
            return num_selected + penalty * num_uncovered
        
        best_cost = current_cost()
        best_selected = selected.copy()
        best_cover = cover_count.copy()
        
        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        total_time = max_time
        sa_start = time.time()
        
        # Precompute which blocks are selected for quick access
        selected_indices = list(np.where(selected)[0])
        
        iteration = 0
        while time.time() < sa_start + total_time:
            # Adaptive temperature
            elapsed = time.time() - sa_start
            progress = min(1.0, elapsed / total_time)
            T = T_start * (T_end / T_start) ** progress
            
            # Choose move: with probability p_off try to turn off, else turn on
            p_off = 0.6  # Bias towards removing blocks (minimization)
            
            if random.random() < p_off and len(selected_indices) > 0:
                # Turn off a random selected block
                pos = random.randint(0, len(selected_indices) - 1)
                b_idx = selected_indices[pos]
                
                # Compute delta
                delta_cost = 1  # +1 to num_selected
                would_uncover = 0
                for ts_idx in block_to_ts[b_idx]:
                    if cover_count[ts_idx] == 1:
                        would_uncover += 1
                delta_cost += penalty * would_uncover
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    selected[b_idx] = False
                    selected_indices.pop(pos)
                    for ts_idx in block_to_ts[b_idx]:
                        cover_count[ts_idx] -= 1
                    num_selected -= 1
                    num_uncovered += would_uncover
                    
                    cost = current_cost()
                    if cost < best_cost:
                        best_cost = cost
                        best_selected = selected.copy()
                        best_cover = cover_count.copy()
                else:
                    # Reject (no change)
                    pass
            else:
                # Turn on a random unselected block
                # Sample a few candidates to find one with good gain
                num_unselected = num_k_blocks - len(selected_indices)
                if num_unselected <= 0:
                    continue
                
                # Sample up to 10 random unselected blocks
                sample_size = min(10, num_unselected)
                best_candidate = -1
                best_gain = -1
                
                for _ in range(sample_size):
                    # Pick random index not in selected
                    while True:
                        rand_idx = random.randint(0, num_k_blocks - 1)
                        if not selected[rand_idx]:
                            break
                    
                    gain = 0
                    for ts_idx in block_to_ts[rand_idx]:
                        if cover_count[ts_idx] == 0:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = rand_idx
                
                if best_candidate == -1 or best_gain == 0:
                    continue
                
                b_idx = best_candidate
                delta_cost = 1 - penalty * best_gain  # +1 selected, -penalty*uncovered_covered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    # Accept
                    selected[b_idx] = True
                    selected_indices.append(b_idx)
                    for ts_idx in block_to_ts[b_idx]:
                        if cover_count[ts_idx] == 0:
                            num_uncovered -= 1
                        cover_count[ts_idx] += 1
                    num_selected += 1
                    
                    cost = current_cost()
                    if cost < best_cost:
                        best_cost = cost
                        best_selected = selected.copy()
                        best_cover = cover_count.copy()
            
            iteration += 1
        
        # Extract best solution
        best_blocks = []
        for i in range(num_k_blocks):
            if best_selected[i]:
                best_blocks.append(list(all_k_blocks[i]))
        
        # Ensure validity: add back any uncovered subsets
        final_cover = np.zeros(num_t_subsets, dtype=np.int32)
        for b in best_blocks:
            for ts in itertools.combinations(b, t):
                final_cover[ts_index_map[ts]] += 1
        uncovered = np.where(final_cover == 0)[0]
        if len(uncovered) > 0:
            for ts_idx in uncovered:
                if time.time() > deadline - 3.0:
                    break
                for b_idx in ts_to_blocks[ts_idx]:
                    if not best_selected[b_idx]:
                        best_selected[b_idx] = True
                        best_blocks.append(list(all_k_blocks[b_idx]))
                        for ts2 in block_to_ts[b_idx]:
                            final_cover[ts2] += 1
                        break
        
        return best_blocks

    # Run multiple SA restarts with different initial solutions
    # First, use the greedy solution
    sa_result = simulated_annealing(best_solution, max_time=60.0)
    sa_result = optimize_redundancy(sa_result)
    if len(sa_result) < best_len:
        best_len = len(sa_result)
        best_solution = sa_result

    # Second, try a smaller random initial solution
    if time.time() < deadline - 30.0:
        random.seed(42)
        # Start with a smaller random set and let SA grow it
        initial_size = max(10, best_len // 2)
        random_blocks = random.sample(range(num_k_blocks), min(initial_size, num_k_blocks))
        initial_solution = [list(all_k_blocks[i]) for i in random_blocks]
        
        # Ensure coverage by adding blocks greedily
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b in initial_solution:
            for ts in itertools.combinations(b, t):
                cover_count[ts_index_map[ts]] += 1
        
        uncovered = np.where(cover_count == 0)[0]
        for ts_idx in uncovered:
            if time.time() > deadline - 10.0:
                break
            # Find best block covering this t-subset
            best_b = -1
            best_gain = -1
            for b_idx in ts_to_blocks[ts_idx]:
                gain = 0
                for ts2 in block_to_ts[b_idx]:
                    if cover_count[ts2] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
            if best_b >= 0:
                initial_solution.append(list(all_k_blocks[best_b]))
                for ts2 in block_to_ts[best_b]:
                    cover_count[ts2] += 1
        
        sa_result2 = simulated_annealing(initial_solution, max_time=25.0)
        sa_result2 = optimize_redundancy(sa_result2)
        if len(sa_result2) < best_len:
            best_len = len(sa_result2)
            best_solution = sa_result2

    # Final redundancy cleanup
    best_solution = optimize_redundancy(best_solution)

    return best_solution
