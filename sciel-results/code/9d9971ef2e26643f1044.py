import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # Pre-compute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Helper to get index of a subset
    def get_idx(sub):
        return subset_to_idx[sub]

    # Helper to generate all t-subsets of a block
    def get_block_subsets(block):
        return [get_idx(s) for s in itertools.combinations(block, t)]

    # Greedy construction
    def greedy_construct():
        uncovered = set(range(num_subsets))
        blocks = []
        counts = [0] * num_subsets
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered subset
            target_idx = random.choice(list(uncovered))
            target_subset = all_t_subsets[target_idx]
            
            # Find best block containing target_subset
            remaining_points = [p for p in range(v) if p not in target_subset]
            best_block = None
            best_gain = -1
            
            # Iterate candidates
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(target_subset + combo))
                # Calculate gain
                gain = 0
                for sub_idx in get_block_subsets(candidate_block):
                    if sub_idx in uncovered:
                        gain += 1
                    if gain > best_gain:
                        # Can't break early as we need max, but if gain == len(uncovered) we can
                        pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block is None:
                break
            
            blocks.append(best_block)
            for sub_idx in get_block_subsets(best_block):
                counts[sub_idx] += 1
                uncovered.discard(sub_idx)
        
        return blocks, counts

    # Local optimization: remove redundant blocks
    def local_optimize(blocks, counts):
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if b is redundant
                redundant = True
                block_subs = get_block_subsets(b)
                for sub_idx in block_subs:
                    if counts[sub_idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    changed = True
                    # Update counts
                    for sub_idx in block_subs:
                        counts[sub_idx] -= 1
                else:
                    new_blocks.append(b)
            blocks = new_blocks
        return blocks, counts

    # Perturb and Repair
    def perturb_repair(blocks, counts):
        # Deep copy
        blocks = blocks.copy()
        counts = counts.copy()
        
        if not blocks:
            return blocks, counts
        
        # Remove a random block
        rem_idx = random.randint(0, len(blocks) - 1)
        rem_block = blocks.pop(rem_idx)
        
        # Update counts
        uncovered = set()
        for sub_idx in get_block_subsets(rem_block):
            counts[sub_idx] -= 1
            if counts[sub_idx] < 1:
                uncovered.add(sub_idx)
        
        # Repair
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered subset
            target_idx = random.choice(list(uncovered))
            target_subset = all_t_subsets[target_idx]
            
            # Find best block to add
            remaining_points = [p for p in range(v) if p not in target_subset]
            best_block = None
            best_gain = -1
            
            for combo in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(target_subset + combo))
                gain = 0
                for sub_idx in get_block_subsets(candidate_block):
                    if sub_idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block is None:
                break
            
            blocks.append(best_block)
            for sub_idx in get_block_subsets(best_block):
                counts[sub_idx] += 1
                uncovered.discard(sub_idx)
        
        # Final local optimization
        blocks, counts = local_optimize(blocks, counts)
        
        return blocks, counts

    # Main Loop
    best_blocks, best_counts = greedy_construct()
    best_blocks, best_counts = local_optimize(best_blocks, best_counts)
    best_score = len(best_blocks)
    
    # Iterated Local Search
    while time.time() < deadline:
        new_blocks, new_counts = perturb_repair(best_blocks, best_counts)
        
        if len(new_blocks) < best_score:
            best_score = len(new_blocks)
            best_blocks = new_blocks
            best_counts = new_counts
    
    return [list(b) for b in best_blocks]
