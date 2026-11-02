import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_covers, n_t, deadline, counts=None):
    """Remove redundant blocks from selected set. 
    Returns (new_selected, new_counts) if counts provided, else just new_selected.
    """
    if not selected:
        if counts is not None:
            return selected, counts
        return selected
    
    if counts is None:
        counts = [0] * n_t
        for i in selected:
            for s in block_covers[i]:
                counts[s] += 1
            
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in list(selected):
            redundant = True
            for s in block_covers[i]:
                if counts[s] <= 1:
                    redundant = False
                    break
            if redundant:
                selected.remove(i)
                for s in block_covers[i]:
                    counts[s] -= 1
                changed = True
                if not _deadline_ok(deadline):
                    break
    if counts is None:
        return selected
    return selected, counts

def _local_search(selected, block_covers, n_t, sub_to_blocks, deadline):
    """
    Local search with 1-for-1 swaps and 2-for-1 reductions.
    """
    if not selected:
        return selected
    
    # Build initial counts
    counts = [0] * n_t
    for i in selected:
        for s in block_covers[i]:
            counts[s] += 1
            
    best_selected = set(selected)
    best_size = len(best_selected)
    
    # We will loop until no improvement or deadline
    # Improvement is either: size reduction (2-for-1) or structural change (1-for-1) that allows future reduction
    # Since 1-for-1 doesn't reduce size, we need to be careful not to loop forever on neutral moves.
    # We will limit the number of neutral swaps.
    
    max_neutral_swaps = 100
    neutral_swaps = 0
    
    while _deadline_ok(deadline):
        improved = False
        neutral_swaps = 0
        
        # 1. Prune redundant blocks
        best_selected, counts = _prune_redundant(best_selected, block_covers, n_t, deadline, counts)
        if len(best_selected) < best_size:
            best_size = len(best_selected)
            improved = True
            neutral_swaps = 0 # Reset neutral counter on improvement
            
        if not _deadline_ok(deadline):
            break
            
        if not best_selected:
            break

        # 2. Try 2-for-1 reductions
        # Pick random pairs to check
        # To be efficient, we can limit the number of pairs checked per iteration
        max_pairs_checked = 50
        pairs_checked = 0
        
        # Create a list of blocks to iterate over (shuffled for randomness)
        block_list = list(best_selected)
        random.shuffle(block_list)
        
        for i in range(len(block_list)):
            if not _deadline_ok(deadline):
                break
            if pairs_checked >= max_pairs_checked:
                break
            a = block_list[i]
            # Check against a few other blocks
            for j in range(i+1, min(i+5, len(block_list))): # Check next 4 blocks
                if pairs_checked >= max_pairs_checked:
                    break
                pairs_checked += 1
                b = block_list[j]
                
                # Calculate unique subsets for A and B
                # A subset s is unique to A if counts[s] == 1 and s in block_covers[A]
                # But we need the union of unique subsets of A and B.
                # Actually, if we remove A and B, the uncovered subsets are those with count 1 in A or B.
                # Let U be the set of subsets s such that s in covers(A) or s in covers(B) and counts[s] <= 2?
                # No.
                # If we remove A and B, a subset s becomes uncovered if:
                # 1. s in covers(A) and s not in covers(B) and counts[s] == 1 (only A covered it)
                # 2. s in covers(B) and s not in covers(A) and counts[s] == 1 (only B covered it)
                # 3. s in covers(A) and s in covers(B) and counts[s] == 2 (only A and B covered it)
                
                unique_union = set()
                
                for s in block_covers[a]:
                    if counts[s] == 1:
                        unique_union.add(s)
                    elif counts[s] == 2 and s in block_covers[b]:
                        # Only covered by A and B
                        unique_union.add(s)
                        
                for s in block_covers[b]:
                    if counts[s] == 1:
                        unique_union.add(s)
                    # If counts[s] == 2 and s in covers(a), already handled above
                    # If counts[s] > 2, it's covered by others, so safe to remove A/B for this s.
                
                if not unique_union:
                    # Both A and B are redundant? Should have been pruned.
                    # But maybe they are mutually redundant?
                    # If unique_union is empty, removing A and B leaves all subsets covered by others.
                    # So we can remove both.
                    best_selected.remove(a)
                    best_selected.remove(b)
                    for s in block_covers[a]:
                        counts[s] -= 1
                    for s in block_covers[b]:
                        counts[s] -= 1
                    best_size -= 2
                    improved = True
                    neutral_swaps = 0
                    # Break out of pair loops and restart main loop
                    break
                else:
                    # Try to find a block C that covers all of unique_union
                    # Intersect sub_to_blocks for each s in unique_union
                    candidates = None
                    for s in unique_union:
                        if candidates is None:
                            candidates = set(sub_to_blocks[s])
                        else:
                            candidates &= set(sub_to_blocks[s])
                            
                            if not candidates:
                                break
                            
                    if candidates:
                        # Check if any candidate C is valid
                        # C must not be A or B
                        # And C must cover all unique_union (which we ensured by intersection)
                        # And C must not "uncover" anything else?
                        # Wait, if C covers unique_union, and we remove A and B,
                        # do we uncover anything else?
                        # No, because unique_union contains ALL subsets that would become uncovered.
                        # Any subset not in unique_union is covered by at least one other block (not A, not B).
                        # So if C covers unique_union, the solution remains valid.
                        
                        # Pick a random candidate
                        c = random.choice(list(candidates))
                        
                        if c == a or c == b:
                            continue
                            
                        # Perform swap: remove A, B; add C
                        best_selected.remove(a)
                        best_selected.remove(b)
                        best_selected.add(c)
                        
                        # Update counts
                        for s in block_covers[a]:
                            counts[s] -= 1
                        for s in block_covers[b]:
                            counts[s] -= 1
                        for s in block_covers[c]:
                            counts[s] += 1
                            
                        best_size -= 1
                        improved = True
                        neutral_swaps = 0
                        # Break out and restart
                        break
                        
            if improved:
                break
                
        if improved:
            continue # Restart main loop
            
        # 3. Try 1-for-1 swaps to rearrange
        # If we didn't improve, we try to rearrange to enable future improvements
        for a in list(best_selected):
            if not _deadline_ok(deadline):
                break
            if neutral_swaps >= max_neutral_swaps:
                break
                
            # Find unique subsets of A
            unique_subs = []
            for s in block_covers[a]:
                if counts[s] == 1:
                    unique_subs.append(s)
                    
            if not unique_subs:
                # Redundant, should have been pruned
                continue
                
            # Find block C that covers all unique_subs
            candidates = None
            for s in unique_subs:
                if candidates is None:
                    candidates = set(sub_to_blocks[s])
                else:
                    candidates &= set(sub_to_blocks[s])
                    if not candidates:
                        break
                        
            if candidates:
                # Filter out A itself
                candidates.discard(a)
                if candidates:
                    c = random.choice(list(candidates))
                    
                    # Check validity: C covers all unique_subs.
                    # Does swapping A -> C uncover anything?
                    # No, because unique_subs are the only ones uniquely covered by A.
                    # All other subsets covered by A have count > 1, so removing A is safe.
                    # Adding C might cover some, but that's fine.
                    
                    # Perform swap
                    best_selected.remove(a)
                    best_selected.add(c)
                    
                    for s in block_covers[a]:
                        counts[s] -= 1
                    for s in block_covers[c]:
                        counts[s] += 1
                        
                    neutral_swaps += 1
                    # We don't mark 'improved' because size didn't change
                    # But we changed the structure.
                    # Break and restart to re-evaluate redundancies?
                    # Yes, a swap might make another block redundant.
                    break
                    
        if neutral_swaps > 0:
            continue # Restart main loop to check for new redundancies
            
        # No improvement, no neutral swaps
        break
        
    return best_selected

def _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, randomize=False, top_k=10):
    """Greedy set cover with optional randomization."""
    uncovered = set(range(n_t))
    selected = set()
    
    while uncovered:
        if not _deadline_ok(deadline):
            break
        
        # Find the uncovered subset with fewest candidate blocks
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
        
        # Calculate gains for all candidates
        gains = []
        for b in candidates:
            gain = 0
            for s in block_covers[b]:
                if s in uncovered:
                    gain += 1
            gains.append((gain, b))
        
        # Sort by gain descending
        gains.sort(reverse=True)
        
        if not randomize:
            best_block = gains[0][1]
        else:
            top_k_actual = min(top_k, len(gains))
            best_block = random.choice(gains[:top_k_actual])[1]
        
        selected.add(best_block)
        
        for s in block_covers[best_block]:
            uncovered.discard(s)
    
    return selected

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

    # 3. Greedy Set Cover with multiple random restarts
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 10
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy
        selected = _greedy_cover(block_covers, n_t, sub_to_blocks, deadline, 
                                  randomize=(restart > 0), top_k=10)
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
                
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_covers, n_t, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
