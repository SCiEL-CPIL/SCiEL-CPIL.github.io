import itertools
import random
import time
import math

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Simulated Annealing 
    on top of a greedy initialization and redundancy removal.
    """
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    
    # Map t-subset to index for O(1) lookup
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all possible blocks (k-subsets) and their covered t-subsets
    # For v=19, k=7, t=4: C(19,7)=50388 blocks. Each covers C(7,4)=35 t-subsets.
    # This is feasible to precompute.
    all_blocks = list(itertools.combinations(points, k))
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # covered_t_subsets[i] is a list of t-subset indices covered by block i
    covered_t_subsets = []
    for b in all_blocks:
        covered = [t_sub_to_idx[ts] for ts in itertools.combinations(b, t)]
        covered_t_subsets.append(covered)
        
    # Helper to get t-subset indices for a given block tuple
    def get_block_covered_indices(block_tuple):
        return [t_sub_to_idx[ts] for ts in itertools.combinations(block_tuple, t)]

    # Greedy initialization
    def greedy_init():
        uncovered_counts = [1] * num_t_subsets
        blocks = []
        remaining_blocks = set(range(len(all_blocks)))
        
        while any(c > 0 for c in uncovered_counts):
            best_block_idx = -1
            best_gain = 0
            # Randomly sample a subset of remaining blocks if there are too many?
            # For C(19,7), 50k blocks is manageable to scan if we do it a few times.
            # But scanning 50k * 35 operations per step * ~150 steps = 262M ops. Might be slow in Python.
            # Let's optimize: pick a random uncovered t-subset, and only consider blocks covering it.
            
            # Find an uncovered t-subset
            uncovered_indices = [i for i, c in enumerate(uncovered_counts) if c > 0]
            if not uncovered_indices:
                break
            
            target_t_idx = random.choice(uncovered_indices)
            target_t_sub = all_t_subsets[target_t_idx]
            
            # Find blocks that cover this target t-subset
            # We can precompute a map from t-subset index to list of block indices covering it
            # But building that map takes memory. Let's just filter.
            # Actually, let's just iterate through a random sample of remaining blocks or all.
            # Given the time limit, let's try to be smart.
            
            # Optimization: Precompute block coverage for t-subsets?
            # Let's just scan all remaining blocks but break early if we find a very good one?
            # No, we need the best.
            
            # Alternative: Just pick a random block covering the target t-subset?
            # That's weaker.
            
            # Let's stick to scanning all remaining blocks for the greedy step, 
            # but limit the number of blocks we consider if it's too slow.
            # For now, let's assume it's fast enough for v=19.
            
            # To speed up, we can precompute for each t-subset which blocks cover it.
            # t_sub_to_blocks = {idx: [] for idx in range(num_t_subsets)}
            # for b_idx, covered in enumerate(covered_t_subsets):
            #     for t_idx in covered:
            #         t_sub_to_blocks[t_idx].append(b_idx)
            
            # Let's build this map once outside.
            
            best_gain = 0
            best_block_idx = -1
            
            # Iterate through blocks covering the target t-subset
            for b_idx in t_sub_to_blocks[target_t_idx]:
                if b_idx not in remaining_blocks:
                    continue
                gain = 0
                for t_idx in covered_t_subsets[b_idx]:
                    if uncovered_counts[t_idx] > 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                # Fallback: pick any remaining block
                best_block_idx = random.choice(list(remaining_blocks))
            
            blocks.append(best_block_idx)
            remaining_blocks.remove(best_block_idx)
            
            # Update uncovered counts
            for t_idx in covered_t_subsets[best_block_idx]:
                uncovered_counts[t_idx] -= 1
                if uncovered_counts[t_idx] < 0:
                    uncovered_counts[t_idx] = 0
            
        return blocks

    # Precompute t_sub_to_blocks
    t_sub_to_blocks = {idx: [] for idx in range(num_t_subsets)}
    for b_idx, covered in enumerate(covered_t_subsets):
        for t_idx in covered:
            t_sub_to_blocks[t_idx].append(b_idx)

    # Remove redundant blocks
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Count coverage
        coverage_count = [0] * num_t_subsets
        for b_idx in blocks:
            for t_idx in covered_t_subsets[b_idx]:
                coverage_count[t_idx] += 1
        
        # Check for redundancy
        # A block is redundant if all its covered t-subsets have coverage_count > 1
        # Note: We must be careful. If we remove one, others might become redundant.
        # Iterate until no change.
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                is_redundant = True
                for t_idx in covered_t_subsets[b_idx]:
                    if coverage_count[t_idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    changed = True
                    # Decrement counts
                    for t_idx in covered_t_subsets[b_idx]:
                        coverage_count[t_idx] -= 1
                else:
                    new_blocks.append(b_idx)
            
            blocks = new_blocks
        
        return blocks

    # Simulated Annealing
    def simulated_annealing(initial_blocks, time_limit):
        current_blocks = list(initial_blocks)
        current_len = len(current_blocks)
        
        # Initialize coverage counts
        coverage_count = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in covered_t_subsets[b_idx]:
                coverage_count[t_idx] += 1
        
        best_blocks = list(current_blocks)
        best_len = current_len
        
        start_time = time.time()
        T = 100.0
        cooling_rate = 0.999
        
        while time.time() - start_time < time_limit:
            # Pick a random move
            # 1. Remove a block
            # 2. Add a block
            # 3. Swap a block
            
            move_type = random.random()
            
            if move_type < 0.4 and len(current_blocks) > 0:
                # Try to remove a block
                b_idx = random.choice(current_blocks)
                
                # Check if removing it keeps covering valid
                # It is valid if all t-subsets it covers have coverage_count > 1
                can_remove = True
                for t_idx in covered_t_subsets[b_idx]:
                    if coverage_count[t_idx] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove it
                    current_blocks.remove(b_idx)
                    for t_idx in covered_t_subsets[b_idx]:
                        coverage_count[t_idx] -= 1
                    delta = -1
                    # Accept if valid (always is if can_remove)
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    # Cannot remove directly. Try to repair?
                    # For now, skip or try a swap.
                    pass
                    
            elif move_type < 0.7 and len(current_blocks) > 0:
                # Swap: Remove one, add one that covers the lost subsets
                b_idx_to_remove = random.choice(current_blocks)
                
                # Identify t-subsets that will become uncovered if we remove b_idx_to_remove
                # These are t-subsets with coverage_count == 1 that are in b_idx_to_remove
                lost_t_subsets = []
                for t_idx in covered_t_subsets[b_idx_to_remove]:
                    if coverage_count[t_idx] == 1:
                        lost_t_subsets.append(t_idx)
                
                if not lost_t_subsets:
                    # Safe removal, just do the removal move
                    current_blocks.remove(b_idx_to_remove)
                    for t_idx in covered_t_subsets[b_idx_to_remove]:
                        coverage_count[t_idx] -= 1
                    delta = -1
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    # We need to find a block that covers all lost_t_subsets
                    # This is a Set Cover problem on the lost subsets.
                    # Greedy: pick a block that covers the most lost subsets.
                    
                    # Intersection of candidate blocks?
                    # Start with all blocks.
                    # Filter blocks that cover all lost_t_subsets.
                    
                    # This is expensive if done naively.
                    # Heuristic: Pick a random lost_t_subset, find blocks covering it, 
                    # then check if they cover all other lost_t_subsets.
                    
                    candidate_block_idx = -1
                    best_cover_count = 0
                    target_lost = random.choice(lost_t_subsets)
                    
                    for b_idx in t_sub_to_blocks[target_lost]:
                        if b_idx == b_idx_to_remove:
                            continue
                        # Check if b_idx covers all lost_t_subsets
                        covers_all = True
                        count = 0
                        for lt in lost_t_subsets:
                            if lt in covered_t_subsets[b_idx]: # O(1) if set, O(N) if list
                                count += 1
                            else:
                                covers_all = False
                                break
                        if covers_all and count > best_cover_count:
                            best_cover_count = count
                            candidate_block_idx = b_idx
                    
                    if candidate_block_idx != -1:
                        # Perform swap
                        current_blocks.remove(b_idx_to_remove)
                        for t_idx in covered_t_subsets[b_idx_to_remove]:
                            coverage_count[t_idx] -= 1
                        
                        current_blocks.append(candidate_block_idx)
                        for t_idx in covered_t_subsets[candidate_block_idx]:
                            coverage_count[t_idx] += 1
                        
                        # Size unchanged, but might allow future removals
                        # No immediate size improvement, so no update to best here
                        # unless we consider structural improvement, but we only track size.
                        
            else:
                # Add a random block? Unlikely to help reduce size, but helps escape?
                # Usually we want to reduce size.
                pass
            
            # Cooling
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
        
        return best_blocks

    # Main execution
    start_time = time.time()
    time_limit = 110
    
    # 1. Greedy Initialization
    initial_blocks = greedy_init()
    initial_blocks = remove_redundant(initial_blocks)
    
    best_blocks = initial_blocks
    best_len = len(initial_blocks)
    
    # 2. Simulated Annealing
    # Run SA for remaining time
    sa_time_limit = time_limit - (time.time() - start_time)
    if sa_time_limit > 5:
        sa_result = simulated_annealing(initial_blocks, sa_time_limit)
        if len(sa_result) < best_len:
            best_blocks = sa_result
            best_len = len(sa_result)
    
    # 3. Final Redundancy Removal
    best_blocks = remove_redundant(best_blocks)
    
    # Convert indices back to block lists
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    
    return final_blocks
