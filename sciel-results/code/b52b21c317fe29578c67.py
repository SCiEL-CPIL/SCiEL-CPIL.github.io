import itertools
import random
import numpy as np
from collections import defaultdict
from math import comb
import time

def construct(v, k, t):
    # Generate all k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map t-subsets to indices for coverage tracking
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute: for each block, which t-subsets it covers
    block_to_tsubs = []
    tsub_to_blocks = defaultdict(list)
    
    for b_idx, block in enumerate(all_blocks):
        tsubs_in_block = []
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            tsub_to_blocks[idx].append(b_idx)
            tsubs_in_block.append(idx)
        block_to_tsubs.append(tsubs_in_block)
    
    # Precompute tsub_to_blocks as numpy arrays for faster operations
    tsub_to_blocks_arr = [np.array(tsub_to_blocks[i], dtype=np.int32) for i in range(n_tsubs)]
    
    def greedy_init(rng, time_budget):
        """Run randomized greedy to get a starting solution with time awareness."""
        covered = np.zeros(n_tsubs, dtype=bool)
        gain = np.full(n_blocks, comb(k, t), dtype=np.int32)
        selected = []
        in_solution = np.zeros(n_blocks, dtype=bool)
        
        start = time.time()
        while True:
            if time.time() - start > time_budget:
                break
            max_gain = np.max(gain)
            if max_gain == 0:
                break
            candidates = np.where(gain == max_gain)[0]
            b_idx = candidates[rng.randint(0, len(candidates) - 1)]
            selected.append(b_idx)
            in_solution[b_idx] = True
            
            for ts_idx in block_to_tsubs[b_idx]:
                if not covered[ts_idx]:
                    covered[ts_idx] = True
                    for other_b in tsub_to_blocks_arr[ts_idx]:
                        gain[other_b] -= 1
        
        # If we ran out of time, ensure we have a valid cover by completing greedily
        if time.time() - start >= time_budget and not np.all(covered):
            # Complete the cover without time limit
            while True:
                max_gain = np.max(gain)
                if max_gain == 0:
                    break
                candidates = np.where(gain == max_gain)[0]
                b_idx = candidates[rng.randint(0, len(candidates) - 1)]
                if not in_solution[b_idx]:
                    selected.append(b_idx)
                    in_solution[b_idx] = True
                    for ts_idx in block_to_tsubs[b_idx]:
                        if not covered[ts_idx]:
                            covered[ts_idx] = True
                            for other_b in tsub_to_blocks_arr[ts_idx]:
                                gain[other_b] -= 1
        
        return selected, in_solution

    def remove_redundant(selected_list, in_solution, cover_count, rng):
        """Remove all redundant blocks. Returns new length."""
        improved = True
        while improved:
            improved = False
            order = list(range(len(selected_list)))
            rng.shuffle(order)
            for i in order:
                if i >= len(selected_list):
                    continue
                b_idx = selected_list[i]
                is_redundant = True
                for ts_idx in block_to_tsubs[b_idx]:
                    if cover_count[ts_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    in_solution[b_idx] = False
                    for ts_idx in block_to_tsubs[b_idx]:
                        cover_count[ts_idx] -= 1
                    selected_list.pop(i)
                    improved = True
        return len(selected_list)

    def add_remove_cycle(selected_list, in_solution, cover_count, rng, time_budget):
        """
        Try to add a block that covers many under-covered t-subsets,
        then remove redundancies. Accept if it reduces size.
        """
        start = time.time()
        best_len = len(selected_list)
        best_solution = selected_list.copy()
        best_in_solution = in_solution.copy()
        best_cover_count = cover_count.copy()
        
        while time.time() - start < time_budget:
            # Find t-subsets with lowest coverage
            min_count = np.min(cover_count)
            
            # Try adding a random block that covers at least one minimally-covered t-subset
            if min_count <= 1:
                # Pick a random t-subset with minimum coverage
                min_tsubs = np.where(cover_count == min_count)[0]
                ts_idx = min_tsubs[rng.randint(0, len(min_tsubs) - 1)]
                
                # Pick a random block covering this t-subset
                candidate_blocks = tsub_to_blocks_arr[ts_idx]
                if len(candidate_blocks) == 0:
                    break
                b_in = candidate_blocks[rng.randint(0, len(candidate_blocks) - 1)]
                
                if in_solution[b_in]:
                    continue
                
                # Snapshot state
                snap_selected = selected_list.copy()
                snap_in_solution = in_solution.copy()
                snap_cover_count = cover_count.copy()
                
                # Add b_in
                in_solution[b_in] = True
                selected_list.append(b_in)
                for ts_idx2 in block_to_tsubs[b_in]:
                    cover_count[ts_idx2] += 1
                
                # Remove redundancies
                new_len = remove_redundant(selected_list, in_solution, cover_count, rng)
                
                if new_len < best_len:
                    best_len = new_len
                    best_solution = selected_list.copy()
                    best_in_solution = in_solution.copy()
                    best_cover_count = cover_count.copy()
                else:
                    # Revert
                    selected_list[:] = snap_selected
                    in_solution[:] = snap_in_solution
                    cover_count[:] = snap_cover_count
        
        # Apply best
        selected_list[:] = best_solution
        in_solution[:] = best_in_solution
        cover_count[:] = best_cover_count
        return best_len

    # Main ILS Loop
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 110
    
    global_best_len = float('inf')
    global_best_solution = []
    global_best_in_solution = None
    global_best_cover_count = None
    
    num_restarts = 0
    while time.time() - start_time < time_limit:
        num_restarts += 1
        remaining = time_limit - (time.time() - start_time)
        
        # Allocate time: 40% for init, 60% for local search
        init_time = min(remaining * 0.4, 30)
        ls_time = remaining - init_time
        
        # 1. Randomized Greedy Init
        init_sel, init_in_sol = greedy_init(rng, init_time)
        
        # 2. Build cover_count
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for b in init_sel:
            for ts_idx in block_to_tsubs[b]:
                cover_count[ts_idx] += 1
        
        selected_list = init_sel
        in_solution = init_in_sol
        
        # 3. Local Search: First remove redundancies
        remove_redundant(selected_list, in_solution, cover_count, rng)
        
        # 4. Add-Remove cycles
        if time.time() - start_time < time_limit - 5:
            ls_remaining = time_limit - 5 - (time.time() - start_time)
            if ls_remaining > 1:
                add_remove_cycle(selected_list, in_solution, cover_count, rng, ls_remaining)
        
        current_len = len(selected_list)
        if current_len < global_best_len:
            global_best_len = current_len
            global_best_solution = selected_list.copy()
            global_best_in_solution = in_solution.copy()
            global_best_cover_count = cover_count.copy()
    
    # Final polish: one more removal pass on the best solution
    if global_best_solution:
        final_selected = global_best_solution.copy()
        final_in_sol = global_best_in_solution.copy()
        final_cover = global_best_cover_count.copy()
        remove_redundant(final_selected, final_in_sol, final_cover, rng)
        if len(final_selected) < global_best_len:
            global_best_solution = final_selected
    
    # Convert best_solution indices to blocks
    return [list(all_blocks[i]) for i in global_best_solution]
