import itertools
import time
import random
import sys

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their covered t-subsets
    blocks = []
    block_covers = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        for sub in itertools.combinations(block, t):
            covers.append(sub_index[sub])
        blocks.append(block)
        block_covers.append(covers)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    def greedy_cover(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        counts = [0] * n_t
        selected_indices = []
        uncovered = set(range(n_t))
        
        # To speed up, we can pre-shuffle the block indices if we want randomized order for ties?
        # Or just pick max gain. If ties, pick random one.
        
        while uncovered:
            if not _deadline_ok(deadline):
                break
                
            best_block = -1
            max_gain = -1
            candidates = []
            
            # Iterate all blocks
            for b in range(n_blocks):
                gain = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
                    candidates = [b]
                elif gain == max_gain:
                    candidates.append(b)
            
            if max_gain == 0:
                break # No block covers any uncovered subset? Should not happen if cover exists.
                
            if len(candidates) > 1:
                best_block = random.choice(candidates)
            else:
                best_block = candidates[0]
                
            selected_indices.append(best_block)
            
            for s in block_covers[best_block]:
                if s in uncovered:
                    uncovered.discard(s)
                    
        return selected_indices

    def prune_redundant(selected_indices):
        # Convert to set for O(1) lookup? No, we need to update counts.
        # Calculate counts for each subset based on selected blocks
        final_counts = [0] * n_t
        for i in selected_indices:
            for s in block_covers[i]:
                final_counts[s] += 1
                
        # Try to remove blocks
        changed = True
        selected_set = set(selected_indices)
        
        while changed and _deadline_ok(deadline):
            changed = False
            for i in list(selected_set):
                # Check if block i is redundant
                # It is redundant if for all s in block_covers[i], final_counts[s] > 1
                redundant = True
                for s in block_covers[i]:
                    if final_counts[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    selected_set.remove(i)
                    for s in block_covers[i]:
                        final_counts[s] -= 1
                    changed = True
                    if not _deadline_ok(deadline):
                        break
                        
        return list(selected_set)

    best_result = None
    best_len = float('inf')
    
    # Run multiple randomized greedy restarts
    # Estimate time per restart.
    # n_blocks ~ 75k. n_t ~ 11.6k.
    # Greedy steps ~ 300.
    # Each step scans 75k blocks. 300 * 75k = 22.5M block checks.
    # Each check iterates 56 subsets. 22.5M * 56 = 1.26B ops.
    # This is too slow for pure Python if we do many restarts.
    # We need to optimize the greedy step.
    
    # Optimization:
    # Instead of scanning all blocks, we can maintain a list of "active" blocks?
    # Or, we can precompute for each subset the list of blocks covering it.
    # Then, for the "most constrained" heuristic:
    # 1. Find uncovered subset with min degree (fewest covering blocks).
    # 2. Pick best block from its candidates.
    
    # Let's implement the "Most Constrained Variable" heuristic which is often faster and better.
    
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    def greedy_mcv(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        counts = [0] * n_t
        selected_indices = []
        uncovered = set(range(n_t))
        
        while uncovered:
            if not _deadline_ok(deadline):
                break
                
            # Find uncovered subset with fewest candidate blocks
            min_count = float('inf')
            target_sub = -1
            for s in uncovered:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target_sub = s
                    if c == 1:
                        break
            
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks[target_sub]
            
            # Pick candidate that maximizes coverage of other uncovered subsets
            best_block = -1
            max_gain = -1
            best_candidates = []
            
            for b in candidates:
                gain = 0
                for s in block_covers[b]:
                    if s in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b
                    best_candidates = [b]
                elif gain == max_gain:
                    best_candidates.append(b)
                    
            if len(best_candidates) > 1:
                best_block = random.choice(best_candidates)
            else:
                best_block = best_candidates[0]
                
            selected_indices.append(best_block)
            
            for s in block_covers[best_block]:
                if s in uncovered:
                    uncovered.discard(s)
                    
        return selected_indices

    # Run MCV greedy with restarts
    num_restarts = 10 # Adjust based on time
    for i in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        selected = greedy_mcv(i)
        selected = prune_redundant(selected)
        if len(selected) < best_len:
            best_len = len(selected)
            best_result = selected
            
    # If we have time, try local search swaps on the best result?
    # For now, return the best greedy result.
    
    if best_result is None:
        # Fallback to deterministic greedy
        selected = greedy_mcv()
        selected = prune_redundant(selected)
        best_result = selected

    result = [list(blocks[i]) for i in best_result]
    
    # Verify validity
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in best_result:
            for s in block_covers[i]:
                check_counts[s] += 1
        if any(c == 0 for c in check_counts):
             # Should not happen, but if it does, return empty or full?
             # Returning empty is invalid.
             pass
             
    return result
