import itertools
import random
import time

def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    total_subsets = len(all_t_subsets)
    
    # Map subset to index for faster lookup if needed, but set of tuples is fine for v=19
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    def greedy_once():
        # uncovered_mask or list?
        # Using a boolean list is faster for membership if we map to index.
        # But we need to pick a random uncovered one.
        # Let's keep a list of indices of uncovered subsets.
        uncovered_indices = list(range(total_subsets))
        random.shuffle(uncovered_indices)
        
        blocks = []
        
        while uncovered_indices:
            # Pick a random uncovered subset
            # To be faster, just pick the first one? No, random is better for diversity.
            # But shuffling the whole list each time is expensive.
            # Let's pick a random index from the current list of uncovered.
            # We can just pick a random element from the list.
            # But we need to remove it.
            
            # Optimization: Instead of removing from a large list, we can use a "lazy" deletion or just pop from a random position.
            # random.shuffle is O(N).
            # Let's just pick a random index from the current list length.
            
            if not uncovered_indices:
                break
                
            # Pick random uncovered subset
            # We can just take the last one and swap with a random one?
            # Or just random.choice.
            # Let's use a different strategy: 
            # Maintain a set of uncovered.
            # Actually, for v=19, C(19,5) = 11628.
            # A list of 11628 items is small.
            
            # Let's stick to the set approach but optimize the "find one" part.
            # The previous code used random.randint and checked a set.
            # Let's try: pick a random subset from all_t_subsets. If covered, try again.
            # This is what the current code does.
            
            start_subset = None
            attempts = 0
            max_attempts = 50
            while start_subset is None:
                idx = random.randint(0, total_subsets - 1)
                if idx in uncovered_indices: # This check is O(N) if list, O(1) if set.
                    start_subset = all_t_subsets[idx]
                    # Remove from uncovered_indices?
                    # Removing from a list is O(N).
                    # Let's use a set for uncovered_indices.
                attempts += 1
                if attempts > max_attempts:
                    # Fallback: find first uncovered
                    for s in all_t_subsets:
                        if s not in {all_t_subsets[i] for i in uncovered_indices}: # This is wrong logic
                            pass
                    # Let's restart with a cleaner data structure.

    # Revised Greedy with Set of Uncovered
    def greedy_once_v2():
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random element from the set
            start_subset = random.choice(list(uncovered))
            uncovered.remove(start_subset)
            
            block = list(start_subset)
            
            # Greedily add points
            while len(block) < k:
                block_set = set(block)
                best_gain = -1
                best_points = []
                
                for x in points:
                    if x in block_set:
                        continue
                    
                    gain = 0
                    if len(block) >= t - 1:
                        for y_tuple in itertools.combinations(block, t - 1):
                            s = tuple(sorted(y_tuple + (x,)))
                            if s in uncovered:
                                gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        if gain > 0:
                            best_points.append(x)
                        if gain == 0 and best_gain == 0:
                            best_points.append(x)
                
                if not best_points:
                    break
                    
                block.append(random.choice(best_points))
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for s in itertools.combinations(block, t):
                uncovered.discard(s)
                
        return blocks

    def local_search(blocks):
        # Convert to list of tuples for hashing
        blocks = [tuple(b) for b in blocks]
        
        # Precompute subsets for each block
        block_subsets = [list(itertools.combinations(b, t)) for b in blocks]
        
        # Coverage count
        coverage_count = {}
        for subs in block_subsets:
            for s in subs:
                coverage_count[s] = coverage_count.get(s, 0) + 1
        
        changed = True
        while changed:
            changed = False
            random.shuffle(blocks)
            random.shuffle(block_subsets)
            
            # 1. Try to remove redundant block
            for i in range(len(blocks)):
                redundant = True
                for s in block_subsets[i]:
                    if coverage_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for s in block_subsets[i]:
                        coverage_count[s] -= 1
                    del blocks[i]
                    del block_subsets[i]
                    changed = True
                    break
            
            if changed:
                continue
            
            # 2. Try to swap a block
            # Pick a random block index
            i = random.randint(0, len(blocks) - 1)
            # Generate a random new block
            # To make it effective, bias towards covering subsets with low coverage?
            # Let's just pick a random k-subset.
            new_block = tuple(sorted(random.sample(points, k)))
            
            if new_block in blocks:
                continue
                
            # Check if swapping is valid
            # Remove old block's coverage
            for s in block_subsets[i]:
                coverage_count[s] -= 1
            
            # Check if any subset becomes 0
            valid_swap = True
            for s in block_subsets[i]:
                if coverage_count[s] == 0:
                    valid_swap = False
                    break
            
            if not valid_swap:
                # Revert
                for s in block_subsets[i]:
                    coverage_count[s] += 1
                continue
            
            # Add new block's coverage
            new_subs = list(itertools.combinations(new_block, t))
            for s in new_subs:
                coverage_count[s] = coverage_count.get(s, 0) + 1
            
            # Perform swap
            blocks[i] = new_block
            block_subsets[i] = new_subs
            changed = True
            
        return [list(b) for b in blocks]

    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    time_limit = 115
    
    while time.time() - start_time < time_limit:
        blocks = greedy_once_v2()
        blocks = local_search(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # Optional: Print or log? No, just keep.

    return best_blocks if best_blocks else []
