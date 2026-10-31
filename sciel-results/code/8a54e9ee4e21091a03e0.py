import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115  # Leave margin
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(all_points, t))
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_tsubsets = len(all_tsubsets)
    
    # Precompute all k-blocks and their covered t-subsets
    all_blocks = list(itertools.combinations(all_points, k))
    block_to_tsubs = {}
    tsub_to_blocks = {i: [] for i in range(n_tsubsets)}
    
    for b in all_blocks:
        covered_tsubs = []
        for ts in itertools.combinations(b, t):
            idx = tsubset_index[ts]
            covered_tsubs.append(idx)
            tsub_to_blocks[idx].append(b)
        block_to_tsubs[b] = covered_tsubs

    def greedy_init():
        # Standard Greedy Set Cover with optimization
        uncovered = [1] * n_tsubsets
        n_uncovered = n_tsubsets
        blocks = []
        block_set = set()
        
        # Track for each block how many uncovered tsubs it covers
        # We'll recompute gain each iteration but use early termination
        
        while n_uncovered > 0:
            best_block = None
            best_gain = -1
            
            for b in all_blocks:
                if b in block_set:
                    continue
                gain = 0
                for ts_idx in block_to_tsubs[b]:
                    if uncovered[ts_idx]:
                        gain += 1
                        if gain <= best_gain:
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is None or best_gain == 0:
                break
                
            blocks.append(best_block)
            block_set.add(best_block)
            
            for ts_idx in block_to_tsubs[best_block]:
                if uncovered[ts_idx]:
                    uncovered[ts_idx] = 0
                    n_uncovered -= 1
                    
        return blocks

    def local_search(blocks, time_budget=10.0):
        """Remove redundant blocks iteratively"""
        blocks = list(blocks)
        block_set = set(blocks)
        
        # Build coverage
        coverage = [0] * n_tsubsets
        for b in blocks:
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1
        
        removed_any = True
        while removed_any and (time.time() - start_time) < time_limit - time_budget:
            removed_any = False
            for i, b in enumerate(blocks):
                # Check if b can be removed
                can_remove = True
                for ts_idx in block_to_tsubs[b]:
                    if coverage[ts_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove b
                    for ts_idx in block_to_tsubs[b]:
                        coverage[ts_idx] -= 1
                    blocks.pop(i)
                    block_set.remove(b)
                    removed_any = True
                    break
            if not removed_any:
                # Try swap: replace b with b' that covers all uniquely covered tsubs
                for i, b in enumerate(blocks):
                    # Find tsubs uniquely covered by b
                    unique_tsubs = [ts_idx for ts_idx in block_to_tsubs[b] if coverage[ts_idx] == 1]
                    if not unique_tsubs:
                        continue
                    # Find a block that covers all unique_tsubs
                    for b_new in all_blocks:
                        if b_new in block_set:
                            continue
                        covers_all = all(ts_idx in block_to_tsubs[b_new] for ts_idx in unique_tsubs)
                        if covers_all:
                            # Check if swap is beneficial (same size, so neutral, but might enable future removals)
                            # Apply swap
                            for ts_idx in block_to_tsubs[b]:
                                coverage[ts_idx] -= 1
                            for ts_idx in block_to_tsubs[b_new]:
                                coverage[ts_idx] += 1
                            blocks[i] = b_new
                            block_set.remove(b)
                            block_set.add(b_new)
                            removed_any = True
                            break
                    if removed_any:
                        break
        return blocks

    def simulated_annealing(init_blocks, max_time=60.0):
        # Initialize state
        current_blocks = list(init_blocks)
        current_block_set = set(current_blocks)
        
        # Coverage counts
        coverage = [0] * n_tsubsets
        for b in current_blocks:
            for ts_idx in block_to_tsubs[b]:
                coverage[ts_idx] += 1
        
        # Current Energy (number of uncovered)
        current_energy = sum(1 for c in coverage if c == 0)
        
        best_blocks = current_blocks[:]
        best_energy = current_energy
        best_size = len(current_blocks)
        
        # SA Parameters - improved schedule
        T = 5.0
        T_min = 0.005
        alpha = 0.997  # Slower cooling
        
        current_time = time.time()
        
        # Helper to calculate delta energy for a move
        def get_remove_delta(b):
            delta = 0
            for ts_idx in block_to_tsubs[b]:
                if coverage[ts_idx] == 1:
                    delta += 1
            return delta

        def get_swap_delta(b_old, b_new):
            # Efficient delta calculation
            affected = set(block_to_tsubs[b_old])
            affected.update(block_to_tsubs[b_new])
            
            old_energy_part = 0
            new_energy_part = 0
            
            for ts in affected:
                old_c = coverage[ts]
                new_c = old_c
                if ts in block_to_tsubs[b_old]:
                    new_c -= 1
                if ts in block_to_tsubs[b_new]:
                    new_c += 1
                
                if old_c == 0:
                    old_energy_part += 1
                if new_c == 0:
                    new_energy_part += 1
                    
            return new_energy_part - old_energy_part

        iteration = 0
        no_improve_count = 0
        last_improve_iter = 0
        
        while (time.time() - start_time) < time_limit - 15:
            # Choose a move
            move_type = random.random()
            
            if move_type < 0.35:
                # Remove
                if len(current_blocks) == 0:
                    continue
                idx = random.randint(0, len(current_blocks) - 1)
                b_old = current_blocks[idx]
                delta = get_remove_delta(b_old)
                
                # Acceptance
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                    # Apply
                    current_blocks.pop(idx)
                    current_block_set.remove(b_old)
                    for ts_idx in block_to_tsubs[b_old]:
                        coverage[ts_idx] -= 1
                    current_energy += delta
                    
                    if current_energy == 0 and len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_energy = 0
                        best_blocks = current_blocks[:]
                        last_improve_iter = iteration
                        no_improve_count = 0
                    elif current_energy == 0:
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                        
            elif move_type < 0.85:
                # Swap
                if len(current_blocks) == 0:
                    continue
                idx = random.randint(0, len(current_blocks) - 1)
                b_old = current_blocks[idx]
                
                b_new = random.choice(all_blocks)
                if b_new in current_block_set:
                    continue
                
                delta = get_swap_delta(b_old, b_new)
                
                # Acceptance
                if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                    # Apply
                    current_blocks[idx] = b_new
                    current_block_set.remove(b_old)
                    current_block_set.add(b_new)
                    
                    for ts_idx in block_to_tsubs[b_old]:
                        coverage[ts_idx] -= 1
                    for ts_idx in block_to_tsubs[b_new]:
                        coverage[ts_idx] += 1
                        
                    current_energy += delta
                    
                    if current_energy == 0 and len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_energy = 0
                        best_blocks = current_blocks[:]
                        last_improve_iter = iteration
                        no_improve_count = 0
                    elif current_energy == 0:
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
            else:
                # 2-opt style: remove two blocks, try to add one that covers both
                if len(current_blocks) < 2:
                    continue
                idx1 = random.randint(0, len(current_blocks) - 1)
                idx2 = random.randint(0, len(current_blocks) - 1)
                if idx1 == idx2:
                    continue
                b1 = current_blocks[idx1]
                b2 = current_blocks[idx2]
                
                # Find tsubs uniquely covered by b1 or b2
                unique_to_both = []
                for ts_idx in block_to_tsubs[b1]:
                    if coverage[ts_idx] == 1:
                        unique_to_both.append(ts_idx)
                for ts_idx in block_to_tsubs[b2]:
                    if coverage[ts_idx] == 1 and ts_idx not in block_to_tsubs[b1]:
                        unique_to_both.append(ts_idx)
                
                if not unique_to_both:
                    continue
                
                # Find a block that covers all unique_to_both
                for b_new in random.sample(all_blocks, min(100, len(all_blocks))):
                    if b_new in current_block_set:
                        continue
                    covers_all = all(ts_idx in block_to_tsubs[b_new] for ts_idx in unique_to_both)
                    if covers_all:
                        # Compute delta
                        delta = 0
                        affected = set(block_to_tsubs[b1])
                        affected.update(block_to_tsubs[b2])
                        affected.update(block_to_tsubs[b_new])
                        
                        old_energy_part = 0
                        new_energy_part = 0
                        for ts in affected:
                            old_c = coverage[ts]
                            new_c = old_c
                            if ts in block_to_tsubs[b1]:
                                new_c -= 1
                            if ts in block_to_tsubs[b2]:
                                new_c -= 1
                            if ts in block_to_tsubs[b_new]:
                                new_c += 1
                            if old_c == 0:
                                old_energy_part += 1
                            if new_c == 0:
                                new_energy_part += 1
                        delta = new_energy_part - old_energy_part
                        # Net size change: -2 + 1 = -1, so we always improve size if valid
                        if delta <= 0:
                            # Apply
                            for ts_idx in block_to_tsubs[b1]:
                                coverage[ts_idx] -= 1
                            for ts_idx in block_to_tsubs[b2]:
                                coverage[ts_idx] -= 1
                            for ts_idx in block_to_tsubs[b_new]:
                                coverage[ts_idx] += 1
                            
                            # Remove b1 and b2, add b_new
                            if idx1 > idx2:
                                idx1, idx2 = idx2, idx1
                            current_blocks.pop(idx2)
                            current_blocks.pop(idx1)
                            current_block_set.remove(b1)
                            current_block_set.remove(b2)
                            current_blocks.append(b_new)
                            current_block_set.add(b_new)
                            current_energy += delta
                            
                            if current_energy == 0 and len(current_blocks) < best_size:
                                best_size = len(current_blocks)
                                best_energy = 0
                                best_blocks = current_blocks[:]
                                last_improve_iter = iteration
                                no_improve_count = 0
                            break
                        break
            
            iteration += 1
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_blocks

    # Run
    init_blocks = greedy_init()
    result_blocks = simulated_annealing(init_blocks)
    
    # Local search refinement
    result_blocks = local_search(result_blocks, time_budget=5.0)
    
    # Ensure validity
    coverage = [0] * n_tsubsets
    for b in result_blocks:
        for ts_idx in block_to_tsubs[b]:
            coverage[ts_idx] += 1
    
    if any(c == 0 for c in coverage):
        # Invalid, fall back to greedy
        result_blocks = init_blocks
    
    return [list(b) for b in result_blocks]
