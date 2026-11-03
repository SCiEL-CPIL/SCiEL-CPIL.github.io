import itertools
import time
import random

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    index_map = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    def greedy_construct(seed):
        random.seed(seed)
        # uncovered is a set of indices for fast removal and membership
        uncovered = set(range(n_t_subsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_idx = random.choice(list(uncovered))
            start_sub = all_t_subsets[start_idx]
            block = list(start_sub)
            
            # Grow the block to size k
            while len(block) < k:
                best_gain = -1
                best_x = -1
                # Candidates are points not in block
                candidates = [x for x in range(v) if x not in block]
                
                # Precompute (t-1) subsets of current block
                t_minus_1_subs = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    gain = 0
                    for S in t_minus_1_subs:
                        # T = S U {x}
                        T = S + (x,)
                        idx = index_map[T]
                        if idx in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x != -1:
                        # Tie-breaking: random
                        if random.random() < 0.5:
                            best_x = x
                            
                if best_x == -1:
                    break
                block.append(best_x)
                
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                idx = index_map[sub]
                if idx in uncovered:
                    uncovered.remove(idx)
                    
        return blocks

    def local_search_reduce(blocks, deadline):
        # Convert blocks to sets of indices for faster lookup
        block_indices = []
        for b in blocks:
            idxs = set(index_map[sub] for sub in itertools.combinations(b, t))
            block_indices.append(idxs)
            
        # cover_count[i] = number of blocks covering t-subset i
        cover_count = [0] * n_t_subsets
        for idxs in block_indices:
            for i in idxs:
                cover_count[i] += 1
                
        current_blocks = list(blocks)
        current_block_indices = list(block_indices)
        
        # Iterative reduction
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove a block
            order = list(range(len(current_blocks)))
            random.shuffle(order)
            
            for i in order:
                if time.time() > deadline:
                    break
                idxs = current_block_indices[i]
                # Check if all subsets in this block are covered by other blocks
                redundant = True
                for j in idxs:
                    if cover_count[j] <= 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove block i
                    for j in idxs:
                        cover_count[j] -= 1
                    current_blocks.pop(i)
                    current_block_indices.pop(i)
                    improved = True
                    break # Restart loop
                    
        return current_blocks

    deadline = time.time() + 110
    best_blocks = None
    best_count = float('inf')
    seed = 0
    
    # Run multiple greedy constructions
    while time.time() < deadline:
        seed += 1
        blocks = greedy_construct(seed)
        
        # Reduce
        reduced_blocks = local_search_reduce(blocks, deadline)
        
        if len(reduced_blocks) < best_count:
            best_count = len(reduced_blocks)
            best_blocks = reduced_blocks
            
    return best_blocks
