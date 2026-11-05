import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    all_t_subsets = []
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for p in combo:
            mask |= (1 << p)
        all_t_subsets.append(mask)
        
    uncovered_set = set(all_t_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Greedy block construction
            # Pick first point: max frequency in uncovered
            counts = [0] * v
            for mask in uncovered:
                m = mask
                while m:
                    lsb = m & -m
                    p = lsb.bit_length() - 1
                    counts[p] += 1
                    m ^= lsb
            
            max_count = max(counts)
            candidates = [p for p in range(v) if counts[p] == max_count]
            block = [random.choice(candidates)]
            
            while len(block) < k:
                best_x = -1
                best_gain = -1
                block_set = set(block)
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub in t_minus_1_subsets:
                        mask = 0
                        for p in sub:
                            mask |= (1 << p)
                        mask |= (1 << x)
                        if mask in uncovered:
                            gain += 1
                    
                    # Random tie-breaking / noise
                    noise = random.uniform(0, 0.5)
                    if gain + noise > best_gain:
                        best_gain = gain + noise
                        best_x = x
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                mask = 0
                for p in sub:
                    mask |= (1 << p)
                uncovered.discard(mask)
        
        # Post-processing: Remove redundant blocks using coverage counts
        # Initialize coverage counts
        coverage = {mask: 0 for mask in all_t_subsets}
        for block in blocks:
            for sub in itertools.combinations(block, t):
                mask = 0
                for p in sub:
                    mask |= (1 << p)
                coverage[mask] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for block in blocks:
                # Check if all t-subsets of this block have coverage > 1
                redundant = True
                for sub in itertools.combinations(block, t):
                    mask = 0
                    for p in sub:
                        mask |= (1 << p)
                    if coverage[mask] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove this block
                    for sub in itertools.combinations(block, t):
                        mask = 0
                        for p in sub:
                            mask |= (1 << p)
                        coverage[mask] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks.copy()
            
            if best_count <= 265: # Lower bound is 264
                break

    if best_blocks is None:
        # Fallback
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            block = [0]
            while len(block) < k:
                best_x = -1
                best_gain = -1
                t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for sub in t_minus_1_subsets:
                        mask = 0
                        for p in sub:
                            mask |= (1 << p)
                        mask |= (1 << x)
                        if mask in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for sub in itertools.combinations(block, t):
                mask = 0
                for p in sub:
                    mask |= (1 << p)
                uncovered.discard(mask)
        best_blocks = blocks

    return best_blocks
