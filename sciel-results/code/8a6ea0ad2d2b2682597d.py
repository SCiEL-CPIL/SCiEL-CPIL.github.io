import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(v), t))
    n_triples = len(all_triples)
    
    # Map triple to index
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    
    def get_block_triples_indices(block):
        return [triple_to_idx[tr] for tr in itertools.combinations(block, t)]
        
    def greedy_construct(seed_offset=0):
        np.random.seed(42 + seed_offset)
        count = np.zeros(n_triples, dtype=np.int32)
        blocks = []
        
        while np.any(count == 0):
            if time.time() > deadline - 10:
                break
            
            uncovered_idx = np.where(count == 0)[0]
            if len(uncovered_idx) == 0:
                break
                
            # Pick a few random uncovered triples to seed
            seeds = np.random.choice(uncovered_idx, size=min(10, len(uncovered_idx)), replace=False)
            
            best_block = None
            best_gain = -1
            
            for seed_idx in seeds:
                seed_triple = all_triples[seed_idx]
                current_block = list(seed_triple)
                remaining = set(range(v))
                for p in seed_triple:
                    remaining.discard(p)
                
                for _ in range(k - t):
                    best_cand = None
                    best_cand_gain = -1
                    cands = list(np.random.choice(list(remaining), size=min(50, len(remaining)), replace=False))
                    
                    for cand in cands:
                        new_block = current_block + [cand]
                        gain = 0
                        for tr in itertools.combinations(new_block, t):
                            if count[triple_to_idx[tr]] == 0:
                                gain += 1
                        if gain > best_cand_gain:
                            best_cand_gain = gain
                            best_cand = cand
                            
                    if best_cand is None:
                        best_cand = random.choice(list(remaining))
                    current_block.append(best_cand)
                    remaining.discard(best_cand)
                    
                block_gain = 0
                for tr in itertools.combinations(current_block, t):
                    if count[triple_to_idx[tr]] == 0:
                        block_gain += 1
                        
                if block_gain > best_gain:
                    best_gain = block_gain
                    best_block = tuple(sorted(current_block))
                    
            if best_block is None:
                best_block = tuple(sorted(random.sample(range(v), k)))
                
            blocks.append(best_block)
            for tr in itertools.combinations(best_block, t):
                count[triple_to_idx[tr]] += 1
                
        return blocks, count

    # Initial greedy
    blocks, count = greedy_construct(0)
    
    best_blocks = blocks
    best_count = count
    best_len = len(blocks)
    
    last_improvement_time = time.time()
    
    while time.time() < deadline - 5:
        improved = False
        
        # 1. Try to remove blocks
        i = 0
        while i < len(blocks):
            block = blocks[i]
            is_removable = True
            block_triples = get_block_triples_indices(block)
            for tr_idx in block_triples:
                if count[tr_idx] == 1:
                    is_removable = False
                    break
            
            if is_removable:
                for tr_idx in block_triples:
                    count[tr_idx] -= 1
                blocks.pop(i)
                improved = True
            else:
                i += 1
                
        if improved:
            last_improvement_time = time.time()
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [b[:] for b in blocks]
                best_count = count.copy()
            continue # Restart removal pass
            
        # 2. If no removals, try to replace blocks
        # Calculate critical count for each block
        block_critical_counts = []
        for block in blocks:
            crit_count = 0
            for tr_idx in get_block_triples_indices(block):
                if count[tr_idx] == 1:
                    crit_count += 1
            block_critical_counts.append(crit_count)
            
        candidates = [i for i, c in enumerate(block_critical_counts) if c > 0]
        
        if not candidates:
            # All blocks are removable? Should have been removed.
            break
            
        # Pick the block with the most critical triples
        # Or random? Most critical is better to fix.
        target_idx = max(candidates, key=lambda i: block_critical_counts[i])
        
        old_block = blocks[target_idx]
        old_triples = get_block_triples_indices(old_block)
        old_critical_triples = [tr_idx for tr_idx in old_triples if count[tr_idx] == 1]
        
        found_replacement = False
        # Try 20 random replacements
        for _ in range(20):
            required_points = set()
            for tr_idx in old_critical_triples:
                tr = all_triples[tr_idx]
                for p in tr:
                    required_points.add(p)
                    
            if len(required_points) > k:
                continue
                
            remaining_points = set(range(v)) - required_points
            fill_needed = k - len(required_points)
            if fill_needed > 0:
                fill_points = random.sample(list(remaining_points), fill_needed)
            else:
                fill_points = []
                
            new_block = tuple(sorted(list(required_points) + fill_points))
            
            if new_block == old_block:
                continue
                
            # Simulate
            for tr_idx in old_triples:
                count[tr_idx] -= 1
            new_triples = get_block_triples_indices(new_block)
            for tr_idx in new_triples:
                count[tr_idx] += 1
                
            # Check validity (should be valid)
            # Check if it allows removals? 
            # We just accept it to explore.
            blocks[target_idx] = new_block
            found_replacement = True
            break
            
        if found_replacement:
            last_improvement_time = time.time()
            # Don't update best yet, wait for removals
        else:
            # Stuck
            # Check time
            if time.time() - last_improvement_time > 5:
                break
            # Try random restart of local search?
            # Or just break.
            break

    # Return the best found
    # Re-verify best_blocks
    # If best_blocks is not valid, fallback to blocks
    # But we maintained validity.
    
    return [list(b) for b in best_blocks]
