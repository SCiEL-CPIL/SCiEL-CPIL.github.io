import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # 5s margin
    
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Precompute all k-subsets and their covered t-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    block_data = []
    for block in k_subsets:
        covered = set()
        for t_sub in itertools.combinations(block, t):
            covered.add(t_subset_to_idx[t_sub])
        block_data.append((block, covered))
        
    # Greedy Set Cover
    uncovered = set(range(num_t_subsets))
    blocks = [] # List of (block, covered_set)
    
    while uncovered and time.time() < deadline:
        best_block = None
        best_covered = None
        best_gain = 0
        
        # Iterate through all possible blocks
        for block, covered in block_data:
            gain = len(covered & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_covered = covered
        
        if best_gain == 0:
            break
        
        blocks.append((best_block, best_covered))
        uncovered.difference_update(best_covered)
        
    # Local Search: Remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_blocks = []
        for i, (block_i, covered_i) in enumerate(blocks):
            redundant = True
            for t_sub_idx in covered_i:
                covered_by_other = False
                for j, (block_j, covered_j) in enumerate(blocks):
                    if i == j:
                        continue
                    if t_sub_idx in covered_j:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if not redundant:
                new_blocks.append((block_i, covered_i))
            else:
                changed = True
        
        blocks = new_blocks
        
    return [list(block) for block, _ in blocks]
