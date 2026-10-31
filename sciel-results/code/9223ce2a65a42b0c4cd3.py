import time
import random
import itertools
import sys

# Global cache
_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_subsets = list(itertools.combinations(points, t))
    
    n_blocks = len(all_blocks)
    n_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # For each block, list of subset indices
    block_subsets = []
    # Bitmask for each block
    block_masks = []
    
    for b in all_blocks:
        subs = []
        mask = 0
        for s in itertools.combinations(b, t):
            idx = subset_to_idx[s]
            subs.append(idx)
            mask |= (1 << idx)
        block_subsets.append(subs)
        block_masks.append(mask)
        
    # For each subset, list of block indices covering it
    blocks_covering_subset = [[] for _ in range(n_subsets)]
    for i, subs in enumerate(block_subsets):
        for s_idx in subs:
            blocks_covering_subset[s_idx].append(i)
            
    _CACHE[key] = (all_blocks, block_subsets, block_masks, blocks_covering_subset, n_blocks, n_subsets)
    return _CACHE[key]

def _greedy_construct(v, k, t, deadline, rng):
    all_blocks, block_subsets, block_masks, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    # Use numpy for fast gain updates? Or just lists?
    # n_subsets is 3876. n_blocks is 50388.
    # Gains update: for each selected block, for each subset it covers, decrement gain of all blocks covering that subset.
    # This is O(sum of degrees).
    
    gains = [len(block_subsets[0])] * n_blocks
    uncovered = [True] * n_subsets
    
    selected = []
    selected_set = set()
    
    while time.time() < deadline:
        # Check if done
        # Count uncovered
        # This is O(n_subsets). We can maintain a count.
        uncovered_count = sum(1 for u in uncovered if u)
        if uncovered_count == 0:
            break
            
        # Find max gain
        # To add randomness, pick from top few
        # Sort indices by gain? Expensive.
        # Just find max, then pick random among those with gain close to max?
        # Or add noise to gains.
        
        # For simplicity and speed, let's find the actual max, then pick randomly from blocks with gain == max_gain * 0.9?
        # Or just pick random from top 10?
        
        # Let's just find max gain and pick randomly from blocks with that gain.
        max_gain = max(gains)
        if max_gain == 0:
            break
            
        candidates = [i for i in range(n_blocks) if gains[i] == max_gain and i not in selected_set]
        
        if not candidates:
            # Fallback: pick any with positive gain
            candidates = [i for i in range(n_blocks) if gains[i] > 0 and i not in selected_set]
            if not candidates:
                break
                
        b = rng.choice(candidates)
        
        selected.append(b)
        selected_set.add(b)
        
        # Update gains
        for s_idx in block_subsets[b]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                for j in blocks_covering_subset[s_idx]:
                    gains[j] -= 1
                    
    return selected, selected_set

def _local_search(selected, selected_set, block_subsets, block_masks, blocks_covering_subset, n_subsets, deadline, rng):
    # cover_count[s] = number of blocks covering s
    cover_count = [0] * n_subsets
    
    for b in selected:
        for s_idx in block_subsets[b]:
            cover_count[s_idx] += 1
            
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove Redundant Blocks
        # A block is redundant if it has no unique subsets (cover_count == 1)
        i = 0
        while i < len(selected):
            b = selected[i]
            is_redundant = True
            for s_idx in block_subsets[b]:
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                for s_idx in block_subsets[b]:
                    cover_count[s_idx] -= 1
                selected.pop(i)
                selected_set.remove(b)
                improved = True
            else:
                i += 1
                
        if time.time() > deadline:
            break
            
        # Phase 2: Pair Swaps
        # For each pair (b1, b2), find if a single block b3 can replace them.
        # Critical subsets: s in (b1 U b2) such that cover_count[s] == 1.
        # We need b3 to cover all critical subsets.
        
        n_sel = len(selected)
        # Shuffle to break symmetry
        indices = list(range(n_sel))
        rng.shuffle(indices)
        
        swap_made = False
        for i_idx in range(n_sel):
            b1 = selected[i_idx]
            # Get subsets of b1
            subs_b1 = block_subsets[b1]
            
            for j_idx in range(i_idx + 1, n_sel):
                if time.time() > deadline:
                    return selected, selected_set
                    
                b2 = selected[j_idx]
                subs_b2 = block_subsets[b2]
                
                # Find critical subsets
                # We can use a set for speed if lists are small
                # But lists are size C(7,4)=35.
                # Union is at most 70.
                
                # Identify subsets with count 1
                # We need to check all subsets in b1 and b2.
                # To avoid duplicates, we can use a set of indices or just check both lists.
                
                crit_subs = []
                seen = set()
                
                for s in subs_b1:
                    if cover_count[s] == 1:
                        crit_subs.append(s)
                        seen.add(s)
                        
                for s in subs_b2:
                    if s not in seen and cover_count[s] == 1:
                        crit_subs.append(s)
                        
                if not crit_subs:
                    # This pair is redundant? Should have been caught in Phase 1 if they were individually redundant?
                    # No, if b1 covers s1 (count 1) and b2 covers s2 (count 1), but b1 also covers s2 (count 2) and b2 covers s1 (count 2)?
                    # Then s1 is covered by b1, b2. count=2.
                    # If cover_count[s]==1, it is covered by exactly one block.
                    # So if s in b1 and cover_count[s]==1, then b2 does NOT cover s.
                    # So crit_subs from b1 and b2 are disjoint.
                    # If crit_subs is empty, it means no subset in b1 or b2 has count 1.
                    # This implies b1 and b2 are both redundant?
                    # If b1 has no subset with count 1, it is redundant.
                    # So this case should be handled by Phase 1.
                    continue
                    
                # Find candidate b3
                # Intersect blocks_covering_subset for all s in crit_subs
                # Sort crit_subs by length of blocks_covering_subset to minimize intersection cost
                crit_subs_sorted = sorted(crit_subs, key=lambda s: len(blocks_covering_subset[s]))
                
                candidates_in = set(blocks_covering_subset[crit_subs_sorted[0]])
                
                for s in crit_subs_sorted[1:]:
                    candidates_in &= set(blocks_covering_subset[s])
                    if not candidates_in:
                        break
                        
                if not candidates_in:
                    continue
                    
                # Remove currently selected blocks
                candidates_in -= selected_set
                
                if not candidates_in:
                    continue
                    
                # Pick a candidate
                # Heuristic: Pick one that covers the most other subsets?
                # Or just pick the first one?
                # Let's pick the one with the smallest index to be deterministic-ish, or random.
                # Random might help escape loops?
                b3 = rng.choice(list(candidates_in))
                
                # Perform Swap
                # Remove b1, b2
                for s in subs_b1:
                    cover_count[s] -= 1
                for s in subs_b2:
                    cover_count[s] -= 1
                    
                selected.remove(b1)
                selected.remove(b2)
                selected_set.remove(b1)
                selected_set.remove(b2)
                
                # Add b3
                for s in block_subsets[b3]:
                    cover_count[s] += 1
                selected.append(b3)
                selected_set.add(b3)
                
                swap_made = True
                improved = True
                break # Restart loop after one swap
                
            if swap_made:
                break
                
        if not swap_made:
            # No pair swap found.
            # Could try triple swaps, but might be too slow.
            # Break to save time? Or try a different strategy?
            # If we are stuck, maybe we should stop.
            break
            
    return selected, selected_set

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    
    all_blocks, block_subsets, block_masks, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    best_selected = None
    best_len = float('inf')
    
    # We can run multiple greedy restarts to find a good starting point
    # Then apply local search to the best one.
    # Or apply local search to each? Local search is expensive.
    # Strategy:
    # 1. Run greedy for 5 seconds.
    # 2. Run local search for remaining time on the result.
    # 3. If local search finishes early, restart with a different seed?
    #    But local search is the main reducer.
    
    # Let's do:
    # 1. Greedy (2s)
    # 2. Local Search (110s)
    
    rng = random.Random(42)
    
    # Greedy
    cons_deadline = t0 + 2.0
    selected, selected_set = _greedy_construct(v, k, t, cons_deadline, rng)
    
    # Local Search
    selected, selected_set = _local_search(
        selected, selected_set,
        block_subsets, block_masks, blocks_covering_subset,
        n_subsets, deadline, rng
    )
    
    result = [list(all_blocks[i]) for i in selected]
    
    return result
