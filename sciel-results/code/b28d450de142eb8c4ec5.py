import itertools
import random
import time

def construct(v, k, t):
    all_points = list(range(v))
    # Precompute all t-combinations and their masks
    t_combos = list(itertools.combinations(all_points, t))
    mask_to_idx = {}
    for i, combo in enumerate(t_combos):
        m = 0
        for p in combo:
            m |= (1 << p)
        mask_to_idx[m] = i
    
    n_t = len(t_combos)
    
    def get_mask(points):
        m = 0
        for p in points:
            m |= (1 << p)
        return m

    def greedy_construct(seed):
        random.seed(seed)
        # coverage_count[i] is the number of blocks covering t-combo i
        coverage_count = [0] * n_t
        blocks = []
        
        # Cache for block -> t-indices
        block_t_indices_cache = {}
        
        def get_block_t_indices(block):
            b_tuple = tuple(sorted(block))
            if b_tuple in block_t_indices_cache:
                return block_t_indices_cache[b_tuple]
            indices = []
            for sub in itertools.combinations(b_tuple, t):
                m = 0
                for p in sub:
                    m |= (1 << p)
                indices.append(mask_to_idx[m])
            block_t_indices_cache[b_tuple] = indices
            return indices

        # Uncovered indices set
        uncovered_indices = set(range(n_t))
        
        while uncovered_indices:
            # Pick a random uncovered t-subset to anchor the new block
            # Convert to list for random.choice (O(N) but N is ~11k, acceptable for <300 blocks)
            anchor_idx = random.choice(list(uncovered_indices))
            anchor_points = set(t_combos[anchor_idx])
            
            current_block = list(anchor_points)
            
            # Complete the block greedily
            while len(current_block) < k:
                best_gain = -1
                best_points = []
                
                # Precompute (t-1) subsets of current_block
                t_minus_1_subs = list(itertools.combinations(current_block, t-1))
                
                for x in range(v):
                    if x in current_block:
                        continue
                    
                    gain = 0
                    for sub in t_minus_1_subs:
                        m = 0
                        for p in sub:
                            m |= (1 << p)
                        m |= (1 << x)
                        idx = mask_to_idx.get(m)
                        if idx is not None and idx in uncovered_indices:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain and gain > 0:
                        best_points.append(x)
                    
                if not best_points:
                    # If no gain, pick a random valid point
                    candidates = [x for x in range(v) if x not in current_block]
                    if candidates:
                        best_points = [random.choice(candidates)]
                    else:
                        break
                
                current_block.append(random.choice(best_points))
            
            # Add block
            block_indices = get_block_t_indices(current_block)
            blocks.append(sorted(current_block))
            
            for idx in block_indices:
                coverage_count[idx] += 1
                if coverage_count[idx] == 1:
                    uncovered_indices.discard(idx)

        # Pruning: Remove blocks that are not essential
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                essential = False
                for idx in get_block_t_indices(b):
                    if coverage_count[idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove block
                    for idx in get_block_t_indices(b):
                        coverage_count[idx] -= 1
                        if coverage_count[idx] == 0:
                            uncovered_indices.add(idx)
                    blocks.pop(i)
                    changed = True
        
        return blocks

    # Run multiple restarts
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    timeout = 115 # Leave 5s buffer
    
    seed = 42
    while time.time() - start_time < timeout:
        blocks = greedy_construct(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        seed += 1
            
    return best_blocks
