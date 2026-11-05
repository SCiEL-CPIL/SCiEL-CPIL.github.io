import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Pre-compute all blocks and their 4-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    block_to_subsets = {}
    subset_to_blocks = {s: [] for s in itertools.combinations(range(v), t)}
    
    for b in all_blocks:
        b_set = set(b)
        subs = list(itertools.combinations(b, t))
        block_to_subsets[b] = subs
        for s in subs:
            subset_to_blocks[s].append(b)
            
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple trials with different random seeds
    for trial in range(50):
        if time.time() > deadline:
            break
            
        random.seed(trial * 12345 + 67890)
        
        # 1. Fast Random Greedy Construction
        uncovered = set(itertools.combinations(range(v), t))
        current_blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            s = random.choice(list(uncovered))
            # Pick a random block covering it
            # To improve quality, we could pick the block that covers the most uncovered subsets,
            # but that's slow. Let's stick to random for speed, or a simple heuristic.
            # Let's try a slightly better heuristic: pick a block from the candidates that
            # has the highest "potential" (e.g., random is okay for now, let's see).
            candidates = subset_to_blocks[s]
            b = random.choice(candidates)
            
            if b not in current_blocks:
                current_blocks.append(b)
                # Update uncovered
                for sub in block_to_subsets[b]:
                    if sub in uncovered:
                        uncovered.discard(sub)
                        
        # 2. Greedy Deletion
        # Build count of how many times each subset is covered
        cover_count = {s: 0 for s in itertools.combinations(range(v), t)}
        for b in current_blocks:
            for s in block_to_subsets[b]:
                cover_count[s] += 1
                
        blocks_set = set(current_blocks)
        changed = True
        while changed:
            changed = False
            for b in list(blocks_set):
                # Check if b is redundant
                redundant = True
                for s in block_to_subsets[b]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove b
                    blocks_set.remove(b)
                    for s in block_to_subsets[b]:
                        cover_count[s] -= 1
                    changed = True
                    
        # 3. Local Search (Swap)
        # We try to replace blocks to enable further deletions
        # We maintain blocks as a list for ordering, but use set for lookup
        blocks_list = list(blocks_set)
        
        while time.time() < deadline:
            improved = False
            for i, b in enumerate(blocks_list):
                if time.time() > deadline:
                    break
                    
                # Find unique subsets for b
                unique_subs = [s for s in block_to_subsets[b] if cover_count[s] == 1]
                
                if not unique_subs:
                    # b is redundant, remove it
                    blocks_list.pop(i)
                    blocks_set.remove(b)
                    for s in block_to_subsets[b]:
                        cover_count[s] -= 1
                    improved = True
                    break # Restart loop
                    
                # Try to find a replacement block b' not in blocks_set
                # such that b' covers all unique_subs
                # Intersection of candidate blocks for each unique sub
                candidates = None
                for s in unique_subs:
                    cand_set = set(subset_to_blocks[s])
                    if candidates is None:
                        candidates = cand_set
                    else:
                        candidates &= cand_set
                        
                if candidates:
                    # Remove current blocks from candidates
                    candidates -= blocks_set
                    if candidates:
                        # Pick a random candidate
                        b_prime = random.choice(list(candidates))
                        
                        # Swap b with b_prime
                        blocks_list[i] = b_prime
                        blocks_set.remove(b)
                        blocks_set.add(b_prime)
                        
                        # Update counts
                        for s in block_to_subsets[b]:
                            cover_count[s] -= 1
                        for s in block_to_subsets[b_prime]:
                            cover_count[s] += 1
                            
                        improved = True
                        break # Restart loop
                        
            if not improved:
                break
                
        # Update best
        if len(blocks_list) < best_len:
            best_len = len(blocks_list)
            best_blocks = blocks_list.copy()
            
    if best_blocks is None:
        # Fallback: return any valid covering (should not happen)
        return []
        
    return [list(b) for b in best_blocks]
