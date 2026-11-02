import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_size = float('inf')
    
    def is_valid(blocks):
        covered = [0] * num_t_subsets
        for b in blocks:
            for sub in itertools.combinations(b, t):
                covered[subset_to_idx[sub]] += 1
        return all(c > 0 for c in covered)

    def reduce_blocks(blocks):
        blocks = list(blocks)
        improved = True
        while improved:
            improved = False
            for i in range(len(blocks)):
                candidate_blocks = blocks[:i] + blocks[i+1:]
                if is_valid(candidate_blocks):
                    blocks = candidate_blocks
                    improved = True
                    break
        return blocks

    def greedy_construct(seed):
        random.seed(seed)
        cover_count = [0] * num_t_subsets
        uncovered = set(range(num_t_subsets))
        blocks = []
        all_points = list(range(v))
        
        while uncovered:
            # Pick a random uncovered t-subset
            # Converting to list is O(N), but N is small enough
            target_idx = random.choice(list(uncovered))
            target_subset = all_t_subsets[target_idx]
            
            current_block = set(target_subset)
            
            while len(current_block) < k:
                best_gain = -1
                best_point = -1
                # Candidates are points not in current_block
                # Using list comprehension is fast
                candidates = [x for x in all_points if x not in current_block]
                
                for p in candidates:
                    new_block_set = current_block | {p}
                    gain = 0
                    for sub in itertools.combinations(new_block_set, t):
                        idx = subset_to_idx[sub]
                        if cover_count[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                
                current_block.add(best_point)
            
            block_tuple = tuple(sorted(current_block))
            blocks.append(block_tuple)
            
            # Update cover counts and uncovered set
            for sub in itertools.combinations(block_tuple, t):
                idx = subset_to_idx[sub]
                if cover_count[idx] == 0:
                    uncovered.discard(idx)
                cover_count[idx] += 1
                
        return blocks

    seed = 0
    while time.time() < deadline:
        blocks = greedy_construct(seed)
        reduced_blocks = reduce_blocks(blocks)
        
        if len(reduced_blocks) < best_size:
            best_size = len(reduced_blocks)
            best_blocks = reduced_blocks
            
        seed += 1
        
    if best_blocks is None:
        # Fallback: return a valid cover from the last run if best wasn't updated
        # This shouldn't happen if the loop runs at least once
        blocks = greedy_construct(seed)
        reduced_blocks = reduce_blocks(blocks)
        return [list(b) for b in reduced_blocks]
        
    return [list(b) for b in best_blocks]
