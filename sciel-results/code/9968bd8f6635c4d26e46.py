import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices for fast counting
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Helper to get t-subsets of a block
    def get_block_t_subsets(block):
        return list(itertools.combinations(block, t))
        
    # Greedy construction with randomization
    def greedy_construct():
        # uncovered is a boolean list
        uncovered = [1] * num_t_subsets
        # List of indices of uncovered subsets for random selection
        # We'll maintain a list of candidates. 
        # Initially all indices.
        # When we cover a subset, we don't remove it from the list immediately.
        # We just check if it's still uncovered when we pick one.
        candidates = list(range(num_t_subsets))
        random.shuffle(candidates)
        
        blocks = []
        
        # We need to ensure we don't infinite loop if candidates becomes empty 
        # but there are still uncovered subsets (shouldn't happen if we cover correctly).
        # However, if we pick a covered one, we skip.
        
        # To be safe and efficient, let's just find a random uncovered one.
        # Since N is small, we can just:
        # while True:
        #   idx = random.randint(0, num_t_subsets-1)
        #   if uncovered[idx]: break
        
        while True:
            # Find an uncovered t-subset
            found = False
            # Try a few random picks, or just scan if random fails often?
            # With 3876 items, random is fine.
            while not found:
                idx = random.randint(0, num_t_subsets - 1)
                if uncovered[idx]:
                    found = True
                    break
            
            if not found:
                # This implies all are covered, but the loop condition should have caught it?
                # No, the while True needs a break condition.
                # Let's check if any uncovered remain.
                # Actually, if we exit the inner while, found is True.
                # If we never find one, we loop forever.
                # So we need a global check or a counter.
                # Better: maintain a count of uncovered.
                pass # Logic below handles it
                
            # Start a new block with this t-subset
            base_subset = all_t_subsets[idx]
            block = list(base_subset)
            
            # Greedily add points
            while len(block) < k:
                best_point = None
                max_gain = -1
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    for sub in itertools.combinations(block, t - 1):
                        new_t_sub = tuple(sorted(sub + (x,)))
                        sub_idx = t_subset_to_idx[new_t_sub]
                        if uncovered[sub_idx]:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_point = x
            
            if best_point is not None:
                block.append(best_point)
            else:
                candidates = [x for x in range(v) if x not in block_set]
                if candidates:
                    block.append(random.choice(candidates))
                else:
                    break
            
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered
            for sub in get_block_t_subsets(block):
                sub_idx = t_subset_to_idx[sub]
                if uncovered[sub_idx]:
                    uncovered[sub_idx] = 0
                    
            # Check if all covered
            # To avoid O(N) check every time, we can keep a count.
            # But O(N) check of 3876 is fast.
            if not any(uncovered):
                break
                
        return blocks

    # Local Search: Remove redundant blocks
    def local_search(blocks):
        cover_counts = [0] * num_t_subsets
        
        for b in blocks:
            for sub in get_block_t_subsets(b):
                cover_counts[t_subset_to_idx[sub]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                b = blocks[i]
                is_redundant = True
                for sub in get_block_t_subsets(b):
                    idx = t_subset_to_idx[sub]
                    if cover_counts[idx] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for sub in get_block_t_subsets(b):
                        idx = t_subset_to_idx[sub]
                        cover_counts[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        return blocks

    best_blocks = None
    best_size = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Deterministic first pass
    random.seed(42)
    blocks = greedy_construct()
    blocks = local_search(blocks)
    if len(blocks) < best_size:
        best_size = len(blocks)
        best_blocks = blocks
        
    while time.time() < deadline:
        random.seed()
        blocks = greedy_construct()
        blocks = local_search(blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
    return [list(b) for b in best_blocks]
