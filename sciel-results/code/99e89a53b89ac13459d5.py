import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(v)
    all_4sets = list(itertools.combinations(points, t))
    n_4sets = len(all_4sets)
    set_to_idx = {s: i for i, s in enumerate(all_4sets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage
    block_to_4sets = []
    for b in all_blocks:
        covered = [set_to_idx[s] for s in itertools.combinations(b, t)]
        block_to_4sets.append(covered)
        
    set_to_blocks = [[] for _ in range(n_4sets)]
    for b_idx, covered in enumerate(block_to_4sets):
        for s_idx in covered:
            set_to_blocks[s_idx].append(b_idx)

    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        # Greedy Construction
        # is_uncovered[s_idx] is True if s_idx is not yet covered
        is_uncovered = [True] * n_4sets
        uncovered_count = n_4sets
        blocks_chosen = []
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered 4-set
            # To avoid O(N) scan for random pick, we can just pick a random index and check
            # But as uncovered_count drops, this becomes inefficient.
            # Let's maintain a list of uncovered indices.
            # Actually, the list filtering approach:
            # We don't have a list here yet. Let's create one initially.
            # But updating it requires knowing which ones were covered.
            
            # Strategy: Pick a random index from 0..n_4sets-1. 
            # If is_uncovered[idx], use it.
            # Else, retry.
            # To prevent infinite loop if uncovered_count is very small, we can 
            # just pick from a pre-computed list of uncovered indices if we maintain it.
            
            # Let's maintain `uncovered_list`
            # Initialization
            if not hasattr(construct, '_uncovered_list_init'):
                pass # Just do it inline
            
            # Re-initialize uncovered_list for this run? 
            # It's expensive to build it every restart? 
            # `uncovered_list = list(range(n_4sets))` is fast.
            # But we need to update it.
            
            # Let's stick to the boolean array + random retry, but with a cap.
            # If we fail to find an uncovered index in 100 tries, scan the array.
            # Given n=3876, scanning is cheap.
            
            target_idx = -1
            for _ in range(10):
                idx = random.randint(0, n_4sets - 1)
                if is_uncovered[idx]:
                    target_idx = idx
                    break
            if target_idx == -1:
                # Scan
                for idx in range(n_4sets):
                    if is_uncovered[idx]:
                        target_idx = idx
                        break
            
            if target_idx == -1:
                break # Should not happen if uncovered_count > 0
                
            candidates = set_to_blocks[target_idx]
            best_block_idx = -1
            best_score = -1
            
            for b_idx in candidates:
                score = 0
                for s_idx in block_to_4sets[b_idx]:
                    if is_uncovered[s_idx]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_block_idx = b_idx
                    
            if best_block_idx != -1:
                blocks_chosen.append(best_block_idx)
                # Update is_uncovered
                for s_idx in block_to_4sets[best_block_idx]:
                    if is_uncovered[s_idx]:
                        is_uncovered[s_idx] = False
                        uncovered_count -= 1
        
        # Reduction Phase
        if blocks_chosen and time.time() < deadline:
            cover_count = [0] * n_4sets
            for b_idx in blocks_chosen:
                for s_idx in block_to_4sets[b_idx]:
                    cover_count[s_idx] += 1
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_blocks = []
                for b_idx in blocks_chosen:
                    redundant = True
                    for s_idx in block_to_4sets[b_idx]:
                        if cover_count[s_idx] == 1:
                            redundant = False
                            break
                    if redundant:
                        for s_idx in block_to_4sets[b_idx]:
                            cover_count[s_idx] -= 1
                        changed = True
                    else:
                        new_blocks.append(b_idx)
                blocks_chosen = new_blocks
            
            if len(blocks_chosen) < best_count:
                best_count = len(blocks_chosen)
                best_blocks = [list(all_blocks[i]) for i in blocks_chosen]
                
        if time.time() > deadline - 5.0:
            break

    if best_blocks is None:
        return []
        
    return best_blocks
