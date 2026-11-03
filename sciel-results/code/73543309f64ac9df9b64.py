import itertools
import time
import random
import sys
import numpy as np
from math import comb

def _deadline_ok(deadline):
    return time.time() < deadline

def _greedy_cover(block_covers, n_t, sub_to_blocks, deadline):
    """
    Standard greedy set cover.
    block_covers: list of lists, block_covers[i] is the list of t-subset indices covered by block i.
    n_t: number of t-subsets.
    sub_to_blocks: list of lists, sub_to_blocks[s] is the list of block indices covering t-subset s.
    """
    uncovered = set(range(n_t))
    selected = set()
    
    while uncovered:
        if not _deadline_ok(deadline):
            break
        
        # Find the uncovered t-subset with the fewest covering blocks (most constrained)
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
        
        # Pick the candidate that covers the most uncovered t-subsets
        best_block = -1
        best_gain = -1
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if s in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
        
        if best_block != -1:
            selected.add(best_block)
            for s in block_covers[best_block]:
                uncovered.discard(s)
    
    return selected

def _local_search(selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=1000):
    """
    Local search to minimize the number of blocks.
    Tries to remove a block and replace it with a better one if necessary.
    """
    if not selected:
        return selected
    
    # counts[s] is the number of selected blocks covering t-subset s
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
            
    best_selected = set(selected)
    best_size = len(best_selected)
    
    # Precompute block sizes (number of t-subsets covered)
    block_sizes = [len(covers) for covers in block_covers]
    
    no_improve_count = 0
    iterations = 0
    
    while _deadline_ok(deadline) and iterations < max_iterations:
        iterations += 1
        improved = False
        
        # Sort selected blocks by redundancy (descending) to try removing "heavy" blocks first
        # Redundancy of a block i is the number of t-subsets it covers that are covered by >1 block.
        # Approximation: just try removing blocks in random order or by size.
        # Let's try removing blocks that cover the most "redundant" subsets.
        
        # Calculate redundancy score for each selected block
        redundancy_scores = {}
        for i in best_selected:
            score = 0
            for s in block_covers[i]:
                if counts[s] > 1:
                    score += 1
            redundancy_scores[i] = score
            
        # Try removing blocks with highest redundancy
        candidates_for_removal = sorted(best_selected, key=lambda x: redundancy_scores[x], reverse=True)
        
        for i in candidates_for_removal:
            if not _deadline_ok(deadline):
                break
            
            # Check if we can remove block i
            uncovered_if_removed = []
            for s in block_covers[i]:
                if counts[s] == 1:
                    uncovered_if_removed.append(s)
            
            if not uncovered_if_removed:
                # Safe to remove
                best_selected.remove(i)
                for s in block_covers[i]:
                    counts[s] -= 1
                best_size -= 1
                improved = True
                no_improve_count = 0
                break # Restart the loop as counts changed
            
            # If we can't remove it directly, try to find a replacement
            # We need a block that covers all subsets in uncovered_if_removed
            # And ideally covers other uncovered or low-count subsets.
            
            # Find the most constrained subset in uncovered_if_removed
            min_count = float('inf')
            target_sub = -1
            for s in uncovered_if_removed:
                c = len(sub_to_blocks[s])
                if c < min_count:
                    min_count = c
                    target_sub = s
                    if c == 1:
                        break
            
            if target_sub == -1:
                continue
                
            candidates = sub_to_blocks[target_sub]
            
            best_replacement = -1
            best_gain = -1
            
            for b in candidates:
                if b in best_selected:
                    continue
                
                # Check if b covers all uncovered_if_removed
                covers_all = True
                for s in uncovered_if_removed:
                    if s not in block_covers[b]:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                # Calculate gain: how many new subsets does b cover that are currently uncovered or low count?
                # Actually, we just want to minimize the total count increase.
                # Since we remove i and add b, the net change in size is 0.
                # We want to choose b such that it "helps" the most.
                # A good metric is the number of subsets s where counts[s] == 1 (critical) that b covers.
                # Or simply, the number of subsets in block_covers[b] that are in uncovered_if_removed.
                
                gain = 0
                for s in block_covers[b]:
                    if counts[s] == 0: # Should be 0 if it was uncovered, but we are simulating removal
                        # Actually, if s is in uncovered_if_removed, counts[s] is currently 1.
                        # If we remove i, counts[s] becomes 0.
                        # So b must cover s.
                        pass
                    
                # Let's use a simpler metric: prefer blocks that cover more subsets from uncovered_if_removed
                # and have a high overlap with existing selected blocks (to be redundant in the future? No, we want to keep counts low).
                
                # Better metric: 
                # 1. Must cover all uncovered_if_removed.
                # 2. Minimize the number of subsets where counts[s] becomes > 1.
                
                new_counts_increase = 0
                for s in block_covers[b]:
                    if s in uncovered_if_removed:
                        # counts[s] will be 0 after removing i, then 1 after adding b. No increase in redundancy.
                        continue
                    else:
                        if counts[s] >= 1:
                            new_counts_increase += 1
                
                if new_counts_increase < best_gain or best_gain == -1:
                    best_gain = new_counts_increase
                    best_replacement = b
            
            if best_replacement != -1:
                # Perform swap
                best_selected.remove(i)
                for s in block_covers[i]:
                    counts[s] -= 1
                best_selected.add(best_replacement)
                for s in block_covers[best_replacement]:
                    counts[s] += 1
                # Size remains same, but structure improved (lower redundancy)
                improved = True
                no_improve_count = 0
                break # Restart loop
            
        if not improved:
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    return best_selected

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
    
    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    # 3. Initial Greedy Solution
    best_selected = _greedy_cover(block_covers, n_t, sub_to_blocks, deadline)
    best_size = len(best_selected)
    
    # 4. Local Search Optimization
    if _deadline_ok(deadline):
        best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=2000)
        best_size = len(best_selected)
        
    # 5. Randomized Restarts
    # Try a few times with random tie-breaking in greedy
    for restart in range(5):
        if not _deadline_ok(deadline):
            break
        
        # Simple randomized greedy
        uncovered = set(range(n_t))
        selected = set()
        
        while uncovered and _deadline_ok(deadline):
            # Pick a random uncovered subset
            s = random.choice(list(uncovered))
            candidates = sub_to_blocks[s]
            
            # Pick a random candidate
            b = random.choice(candidates)
            selected.add(b)
            for cov in block_covers[b]:
                uncovered.discard(cov)
                
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected
            
        # Refine the best solution found so far
        if _deadline_ok(deadline):
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline, max_iterations=500)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
        
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
