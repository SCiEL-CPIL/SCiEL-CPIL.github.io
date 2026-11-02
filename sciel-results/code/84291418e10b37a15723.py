import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 118.0  # Safety margin for 150s limit
    
    # 1. Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    t_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    N_b = len(all_blocks)
    
    # For each block, which t-subsets does it cover?
    # block_cover[b] = list of indices of t-subsets contained in block b
    block_cover = []
    for b in all_blocks:
        covered = [t_to_idx[s] for s in itertools.combinations(b, t)]
        block_cover.append(covered)
        
    # Inverse index: for each t-subset, which blocks cover it?
    t_to_blocks = [[] for _ in range(N_t)]
    for bi, covered in enumerate(block_cover):
        for ti in covered:
            t_to_blocks[ti].append(bi)
            
    # Precompute numpy arrays for block covers for faster updates?
    # Lists are fine for now, but we might use them in SA.
    
    # 2. Initial Solution Construction (Greedy)
    # We start with a greedy solution to have a valid baseline.
    
    def greedy_init(rng):
        """
        Fast greedy construction.
        """
        uncovered_mask = np.ones(N_t, dtype=bool)
        num_uncovered = N_t
        selected = []
        
        while num_uncovered > 0:
            best_block = -1
            best_gain = -1
            
            # Sample a few uncovered t-subsets to guide the search
            # This is much faster than scanning all blocks
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
                
            sample_size = min(5, len(uncovered_indices))
            sampled_ts = rng.choice(uncovered_indices, size=sample_size, replace=False)
            
            candidates = set()
            for ti in sampled_ts:
                candidates.update(t_to_blocks[ti])
                
            for bi in candidates:
                gain = 0
                for ti in block_cover[bi]:
                    if uncovered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
            
            if best_block == -1:
                # Fallback
                ti = rng.choice(uncovered_indices)
                best_block = rng.choice(t_to_blocks[ti])
                
            selected.append(best_block)
            
            for ti in block_cover[best_block]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    num_uncovered -= 1
                    
        return selected

    # 3. Simulated Annealing
    # State: set of blocks (represented as a set of indices)
    # Energy: number of blocks + Lambda * (number of uncovered t-subsets)
    # We want to minimize this.
    
    # Initial state
    init_rng = random.Random(42)
    current_blocks = set(greedy_init(init_rng))
    
    # Calculate initial coverage counts
    # counts[ti] = number of blocks covering ti
    counts = np.zeros(N_t, dtype=np.int32)
    for bi in current_blocks:
        for ti in block_cover[bi]:
            counts[ti] += 1
            
    # Number of uncovered
    num_uncovered = np.sum(counts == 0)
    
    # Parameters
    T_start = 50.0
    T_end = 1.0
    cooling_rate = 0.9995 # Adjust based on expected steps
    # We have ~118 seconds.
    # Each step is fast (update counts for 1 block: 56 ops).
    # We can do ~1M steps? 1M * 56 = 56M ops. Python might do 10M-50M simple ops/sec.
    # So ~1-5 seconds for 1M steps. We have plenty of time.
    # Let's target ~500k-1M steps.
    
    best_blocks = set(current_blocks)
    best_energy = len(current_blocks) + 1000 * num_uncovered # High penalty for infeasibility initially
    
    rng = random.Random(123)
    
    # Precompute list of block indices for random selection
    block_indices_list = list(range(N_b))
    
    # Main SA Loop
    T = T_start
    steps = 0
    max_steps = 1000000
    
    # To speed up, we don't recalculate num_uncovered from scratch.
    # We maintain it.
    
    while steps < max_steps and time.time() < deadline - 2.0:
        steps += 1
        
        # 1. Propose a move
        # Types of moves:
        # a) Add a random block
        # b) Remove a random block
        # c) Swap a random block with another random block
        
        move_type = rng.random()
        
        if move_type < 0.3:
            # Add
            if len(current_blocks) > 10: # Avoid growing too much
                new_bi = rng.choice(block_indices_list)
                if new_bi in current_blocks:
                    continue
                # Calculate delta
                # Delta blocks: +1
                # Delta uncovered:
                # For each ti in new_bi:
                #   if counts[ti] == 0: uncovered -= 1
                #   else: unchanged
                delta_uncovered = 0
                for ti in block_cover[new_bi]:
                    if counts[ti] == 0:
                        delta_uncovered -= 1
                
                delta_energy = 1 + 1000 * delta_uncovered
                
                # Acceptance
                if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                    # Apply
                    current_blocks.add(new_bi)
                    for ti in block_cover[new_bi]:
                        counts[ti] += 1
                    num_uncovered += delta_uncovered
                    
                    # Check best
                    energy = len(current_blocks) + 1000 * num_uncovered
                    if energy < best_energy:
                        best_energy = energy
                        best_blocks = set(current_blocks)
                        
        elif move_type < 0.6:
            # Remove
            if not current_blocks:
                continue
            old_bi = rng.choice(list(current_blocks))
            
            # Calculate delta
            # Delta blocks: -1
            # Delta uncovered:
            # For each ti in old_bi:
            #   if counts[ti] == 1: uncovered += 1
            delta_uncovered = 0
            for ti in block_cover[old_bi]:
                if counts[ti] == 1:
                    delta_uncovered += 1
                    
            delta_energy = -1 + 1000 * delta_uncovered
            
            if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                # Apply
                current_blocks.remove(old_bi)
                for ti in block_cover[old_bi]:
                    counts[ti] -= 1
                num_uncovered += delta_uncovered
                
                energy = len(current_blocks) + 1000 * num_uncovered
                if energy < best_energy:
                    best_energy = energy
                    best_blocks = set(current_blocks)
                    
        else:
            # Swap
            if not current_blocks:
                continue
            old_bi = rng.choice(list(current_blocks))
            new_bi = rng.choice(block_indices_list)
            if new_bi == old_bi:
                continue
            if new_bi in current_blocks:
                continue
            
            # Calculate delta
            # Remove old_bi, Add new_bi. Net block change: 0.
            # Delta uncovered:
            # 1. Effect of removing old_bi:
            #    ti in old_bi: if counts[ti] == 1, it becomes 0 (uncovered +1)
            # 2. Effect of adding new_bi:
            #    ti in new_bi:
            #       if it was 0 (and not in old_bi? or in old_bi?):
            #       We need to be careful with the order.
            
            # Let's calculate the final state of counts for affected t-subsets.
            # Affected set = union(old_bi, new_bi)
            
            # Simpler:
            # delta_uncovered = 0
            # For ti in old_bi:
            #    if counts[ti] == 1:
            #       # This ti becomes 0 unless new_bi covers it
            #       if new_bi covers ti:
            #          # counts[ti] becomes 0 - 1 + 1 = 0? No.
            #          # counts[ti] is 1. Remove old -> 0. Add new -> 1.
            #          # So it stays covered. No change in uncovered count?
            #          # Wait. If counts[ti] was 1, and we remove one block covering it, it goes to 0.
            #          # If we add a new block covering it, it goes to 1.
            #          # So net change is 0.
            #       else:
            #          # counts[ti] goes to 0. Uncovered +1.
            #          delta_uncovered += 1
            # For ti in new_bi:
            #    if counts[ti] == 0:
            #       # It was uncovered.
            #       # If old_bi covered it, it would have been handled above?
            #       # No, old_bi covers it implies counts[ti] >= 1.
            #       # So if counts[ti] == 0, old_bi did NOT cover it.
            #       # So adding new_bi makes it covered. Uncovered -1.
            #       delta_uncovered -= 1
            
            # This logic seems correct.
            delta_uncovered = 0
            for ti in block_cover[old_bi]:
                if counts[ti] == 1:
                    # Check if new_bi covers ti
                    # We need a fast way to check if ti is in new_bi.
                    # block_cover[new_bi] is a list.
                    # We can precompute a set for each block? Or just check intersection?
                    # Intersection of two lists of size 56 is fast.
                    # But we are doing this for every ti in old_bi (56 times).
                    # 56 * 56 = 3136 ops. Acceptable.
                    # Better: Precompute a set of t-indices for each block?
                    # That uses memory but speeds up lookup.
                    # Let's use a set for the current block's covers for speed in this loop?
                    # No, let's just iterate.
                    
                    # Optimization: Precompute set of t-indices for each block?
                    # N_b = 75k. 75k * 56 ints. ~4M ints. ~32MB. Acceptable.
                    # Let's do that outside the loop.
                    
                    pass # Placeholder for optimization idea
                    
            # Let's implement the check efficiently.
            # Precompute block_cover_sets?
            # Actually, we can just check if ti is in the list of new_bi's covers.
            # Since lists are small, `ti in list` is O(56).
            # Total 56 * 56 = 3136.
            
            for ti in block_cover[old_bi]:
                if counts[ti] == 1:
                    if ti not in block_cover[new_bi]:
                        delta_uncovered += 1
                        
            for ti in block_cover[new_bi]:
                if counts[ti] == 0:
                    # Note: if ti was in old_bi, counts[ti] would be >= 1.
                    # So if counts[ti] == 0, it's definitely not in old_bi.
                    delta_uncovered -= 1
            
            delta_energy = 1000 * delta_uncovered # Block count unchanged
            
            if delta_energy < 0 or rng.random() < np.exp(-delta_energy / T):
                # Apply
                # Remove old
                for ti in block_cover[old_bi]:
                    counts[ti] -= 1
                # Add new
                for ti in block_cover[new_bi]:
                    counts[ti] += 1
                # Update set
                current_blocks.remove(old_bi)
                current_blocks.add(new_bi)
                num_uncovered += delta_uncovered
                
                energy = len(current_blocks) + 1000 * num_uncovered
                if energy < best_energy:
                    best_energy = energy
                    best_blocks = set(current_blocks)
                    
        # Cool down
        T *= cooling_rate
        
    # 4. Final Repair and Pruning
    # The best solution found by SA might be infeasible (num_uncovered > 0).
    # We need to repair it or fall back to the best feasible solution.
    
    # Let's keep track of the best *feasible* solution separately during SA?
    # Or just repair the best_energy solution?
    # Repairing an infeasible solution is basically greedy completion.
    
    # Let's ensure we have a valid solution.
    # If best_blocks is infeasible, we complete it.
    
    # Re-calculate coverage for best_blocks
    final_counts = np.zeros(N_t, dtype=np.int32)
    for bi in best_blocks:
        for ti in block_cover[bi]:
            final_counts[ti] += 1
            
    # Find uncovered
    uncovered_mask = final_counts == 0
    num_uncovered_final = np.sum(uncovered_mask)
    
    # Complete the solution if needed
    if num_uncovered_final > 0:
        # Greedy completion
        while num_uncovered_final > 0:
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            # Sample
            sample_size = min(5, len(uncovered_indices))
            sampled_ts = rng.choice(uncovered_indices, size=sample_size, replace=False)
            candidates = set()
            for ti in sampled_ts:
                candidates.update(t_to_blocks[ti])
            
            best_block = -1
            best_gain = -1
            for bi in candidates:
                if bi in best_blocks:
                    continue
                gain = 0
                for ti in block_cover[bi]:
                    if uncovered_mask[ti]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = bi
                    
            if best_block == -1:
                ti = rng.choice(uncovered_indices)
                best_block = rng.choice(t_to_blocks[ti])
                
            best_blocks.add(best_block)
            for ti in block_cover[best_block]:
                if uncovered_mask[ti]:
                    uncovered_mask[ti] = False
                    num_uncovered_final -= 1
                final_counts[ti] += 1
                
    # Prune redundant blocks
    # This is crucial for size reduction.
    current_list = list(best_blocks)
    changed = True
    while changed:
        changed = False
        # Recalculate counts
        counts = np.zeros(N_t, dtype=np.int32)
        for bi in current_list:
            for ti in block_cover[bi]:
                counts[ti] += 1
                
        # Find redundant
        removable = []
        for bi in current_list:
            is_redundant = True
            for ti in block_cover[bi]:
                if counts[ti] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                removable.append(bi)
                
        if removable:
            for bi in removable:
                current_list.remove(bi)
                for ti in block_cover[bi]:
                    counts[ti] -= 1
            changed = True
            
    # Return the pruned solution
    if not current_list:
        # Fallback if something went wrong
        current_list = list(greedy_init(rng))
        
    return [list(all_blocks[bi]) for bi in current_list]
