import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145
    
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_to_t_indices = {}
    for b in all_k_blocks:
        block_to_t_indices[b] = [t_subsets_set[s] for s in itertools.combinations(b, t)]
        
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b, t_indices in block_to_t_indices.items():
        for idx in t_indices:
            t_to_blocks[idx].append(b)

    best_blocks = None
    best_len = float('inf')
    
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        random.seed(iterations)
        
        coverage_count = [0] * num_t_subsets
        uncovered_count = num_t_subsets
        current_blocks = []
        
        while uncovered_count > 0:
            # Pick a random uncovered t-subset
            # Use rejection sampling
            while True:
                idx = random.randint(0, num_t_subsets - 1)
                if coverage_count[idx] == 0:
                    break
            
            # Find best block containing idx
            best_block = None
            max_gain = -1
            candidates = t_to_blocks[idx]
            
            for b in candidates:
                gain = 0
                for t_idx in block_to_t_indices[b]:
                    if coverage_count[t_idx] == 0:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
            
            if best_block is None:
                # Should not happen
                break
                
            current_blocks.append(best_block)
            
            for t_idx in block_to_t_indices[best_block]:
                if coverage_count[t_idx] == 0:
                    uncovered_count -= 1
                coverage_count[t_idx] += 1

        # Pruning
        changed = True
        while changed:
            changed = False
            random.shuffle(current_blocks)
            for i in range(len(current_blocks) - 1, -1, -1):
                b = current_blocks[i]
                can_remove = True
                for t_idx in block_to_t_indices[b]:
                    if coverage_count[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    for t_idx in block_to_t_indices[b]:
                        coverage_count[t_idx] -= 1
                    current_blocks.pop(i)
                    changed = True
        
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]
            
    if best_blocks is None:
        return []
    return best_blocks
