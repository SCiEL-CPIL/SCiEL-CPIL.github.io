import itertools
import time
import random
import sys
import numpy as np

def _popcount(x):
    return bin(x).count('1')

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
    block_covers_bitmasks = []
    
    for block in itertools.combinations(range(v), k):
        covers = []
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            covers.append(idx)
            mask |= (1 << idx)
        blocks.append(block)
        block_covers.append(covers)
        block_covers_bitmasks.append(mask)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build sub_to_blocks: for each t-subset, list of blocks covering it
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, covers in enumerate(block_covers):
        for j in covers:
            sub_to_blocks[j].append(i)
            
    if not _deadline_ok(deadline):
        return []

    # 3. Greedy Set Cover for initial solution
    def _greedy_cover():
        uncovered_mask = (1 << n_t) - 1
        selected = set()
        
        while uncovered_mask:
            if not _deadline_ok(deadline):
                break
            
            # Find the uncovered subset with fewest candidate blocks
            min_count = float('inf')
            target_sub = -1
            for s in range(n_t):
                if (uncovered_mask >> s) & 1:
                    c = len(sub_to_blocks[s])
                    if c < min_count:
                        min_count = c
                        target_sub = s
                        if c == 1:
                            break
            
            if target_sub == -1:
                break
            
            candidates = sub_to_blocks[target_sub]
            
            # Calculate gains for all candidates using bitmask
            gains = []
            for b in candidates:
                gain = _popcount(block_covers_bitmasks[b] & uncovered_mask)
                gains.append((gain, b))
            
            # Sort by gain descending
            gains.sort(reverse=True)
            best_block = gains[0][1]
            
            selected.add(best_block)
            uncovered_mask &= ~block_covers_bitmasks[best_block]
        
        return selected, uncovered_mask

    selected, uncovered_mask = _greedy_cover()
    
    best_selected = set(selected)
    best_size = len(best_selected)
    best_uncovered = uncovered_mask
    
    # 4. WalkSAT-style Local Search
    # Objective: Minimize alpha * popcount(uncovered) + popcount(selected)
    # We prioritize covering all subsets (alpha large) then minimize count.
    
    alpha = n_blocks + 1
    
    def _get_cost(sel_mask, unc_mask):
        return alpha * _popcount(unc_mask) + _popcount(sel_mask)
    
    current_sel_mask = 0
    for b in selected:
        current_sel_mask |= (1 << b)
        
    current_unc_mask = uncovered_mask
    current_cost = _get_cost(current_sel_mask, current_unc_mask)
    best_cost = current_cost
    
    # Precompute list of all blocks for random selection
    all_blocks = list(range(n_blocks))
    
    # WalkSAT parameters
    p_walk = 0.2 # Probability of taking a random step
    max_steps = 100000
    
    for step in range(max_steps):
        if not _deadline_ok(deadline):
            break
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_sel_mask = current_sel_mask
            best_unc_mask = current_unc_mask
            best_size = _popcount(best_sel_mask)
            
        # If valid, try to remove blocks
        if current_unc_mask == 0:
            # Try to remove a random selected block
            sel_blocks = [i for i in range(n_blocks) if (current_sel_mask >> i) & 1]
            if sel_blocks:
                b = random.choice(sel_blocks)
                # Check if removing b keeps it valid
                new_unc = current_unc_mask | block_covers_bitmasks[b]
                # Actually, we need to check if any subset becomes uncovered.
                # A subset s is uncovered after removing b if it was ONLY covered by b.
                # This is hard to check quickly with just masks.
                # Alternative: Calculate new uncovered mask.
                # We don't have a "coverage count" array here.
                # Let's use a different heuristic for valid states:
                # Try to remove a block that has low "criticality".
                # Or, just try to remove a random block and check validity by recomputing uncovered?
                # Recomputing uncovered from scratch is O(n_blocks * n_t / word_size).
                # n_blocks ~ 75k, n_t ~ 11k. Too slow.
                
                # Better approach for valid states:
                # 1. Pick a random selected block b.
                # 2. Tentatively remove it.
                # 3. Check if any subset s in block_covers[b] is now uncovered.
                #    We need to know if there is another selected block covering s.
                #    We can precompute for each subset s, the list of blocks.
                #    Then for each s in block_covers[b], check if any other block in sub_to_blocks[s] is selected.
                #    This is O(56 * avg_degree). avg_degree is ~ 11628*56 / 75582 ~ 8.6.
                #    So 56 * 8.6 ~ 500 ops. Very fast.
                
                can_remove = True
                for s in block_covers[b]:
                    # Check if any other selected block covers s
                    found = False
                    for other_b in sub_to_blocks[s]:
                        if other_b != b and ((current_sel_mask >> other_b) & 1):
                            found = True
                            break
                    if not found:
                        can_remove = False
                        break
                
                if can_remove:
                    current_sel_mask &= ~(1 << b)
                    current_cost -= 1 # alpha*0 + (-1)
                    # Uncovered mask remains 0
                else:
                    # Try to swap: remove b, add some other block?
                    # Or just do a random walk step.
                    pass
            
            # Also, try to improve by swapping?
            # For now, just rely on random removals when valid.
            
        else:
            # Invalid state: try to add a block that covers most uncovered subsets
            # Candidates: blocks that cover at least one uncovered subset
            # To speed up, pick the most constrained uncovered subset
            min_count = float('inf')
            target_sub = -1
            for s in range(n_t):
                if (current_unc_mask >> s) & 1:
                    c = len(sub_to_blocks[s])
                    if c < min_count:
                        min_count = c
                        target_sub = s
                        if c == 1:
                            break
            
            if target_sub == -1:
                # Should not happen if current_unc_mask != 0
                break
                
            candidates = sub_to_blocks[target_sub]
            
            # Calculate gain for each candidate
            best_gain = -1
            best_block = -1
            second_best_gain = -1
            second_best_block = -1
            
            for b in candidates:
                gain = _popcount(block_covers_bitmasks[b] & current_unc_mask)
                # Cost change: -alpha * gain + 1 (if not already selected)
                # If already selected, gain is 0 for uncovered, cost change 0?
                # If b is already selected, flipping it OFF would increase uncovered.
                # We only consider flipping OFF if it reduces cost?
                # In invalid state, we prioritize reducing uncovered.
                # So we only consider adding blocks (flipping 0->1) or swapping.
                # Let's stick to adding blocks for now.
                
                if not ((current_sel_mask >> b) & 1):
                    if gain > best_gain:
                        if best_gain > second_best_gain:
                            second_best_gain = best_gain
                            second_best_block = best_block
                        best_gain = gain
                        best_block = b
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_block = b
            
            if best_block == -1:
                # No unselected block covers target_sub? Should not happen.
                break
                
            # WalkSAT decision
            if random.random() < p_walk:
                # Random choice from candidates (unselected ones)
                unselected_cands = [b for b in candidates if not ((current_sel_mask >> b) & 1)]
                if unselected_cands:
                    chosen_b = random.choice(unselected_cands)
                else:
                    chosen_b = best_block
            else:
                chosen_b = best_block
                
            # Flip chosen_b
            current_sel_mask |= (1 << chosen_b)
            current_unc_mask &= ~block_covers_bitmasks[chosen_b]
            current_cost += 1 - alpha * _popcount(block_covers_bitmasks[chosen_b] & ~current_unc_mask) # This cost update is tricky, let's just recalculate or approximate.
            # Actually, simpler: just recalculate cost periodically or trust the mask.
            # Let's just update cost incrementally properly.
            # old_unc = current_unc_mask before update
            # new_unc = current_unc_mask after update
            # delta_unc = popcount(old_unc) - popcount(new_unc)
            # delta_sel = 1
            # delta_cost = -alpha * delta_unc + delta_sel
            
            # We didn't store old_unc. Let's store it.
            # Refactor:
            pass # We will rewrite the loop body more carefully.

    # The above loop body was a sketch. Let's implement a cleaner version.
    
    # Reset for clean implementation
    selected, uncovered_mask = _greedy_cover()
    current_sel_mask = 0
    for b in selected:
        current_sel_mask |= (1 << b)
    current_unc_mask = uncovered_mask
    current_cost = _get_cost(current_sel_mask, current_unc_mask)
    best_cost = current_cost
    best_sel_mask = current_sel_mask
    best_unc_mask = current_unc_mask
    best_size = _popcount(best_sel_mask)
    
    for step in range(max_steps):
        if not _deadline_ok(deadline):
            break
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_sel_mask = current_sel_mask
            best_unc_mask = current_unc_mask
            best_size = _popcount(best_sel_mask)
            
        if current_unc_mask == 0:
            # Valid state
            # Try to remove a block
            sel_blocks = [i for i in range(n_blocks) if (current_sel_mask >> i) & 1]
            if sel_blocks:
                b = random.choice(sel_blocks)
                can_remove = True
                for s in block_covers[b]:
                    found = False
                    for other_b in sub_to_blocks[s]:
                        if other_b != b and ((current_sel_mask >> other_b) & 1):
                            found = True
                            break
                    if not found:
                        can_remove = False
                        break
                
                if can_remove:
                    current_sel_mask &= ~(1 << b)
                    current_cost -= 1
                else:
                    # Try to swap: remove b, add a better one?
                    # Or just do a random walk: flip a random block
                    # Let's try to find a block to add that allows removing b?
                    # This is complex. Let's just try to add a random block and see if we can remove b later.
                    # Or, just do a random flip of a selected block to unselected? No, that breaks validity.
                    # Let's try to add a random unselected block, then try to remove b.
                    unselected = [i for i in range(n_blocks) if not ((current_sel_mask >> i) & 1)]
                    if unselected:
                        add_b = random.choice(unselected)
                        # Tentatively add add_b
                        current_sel_mask |= (1 << add_b)
                        current_unc_mask &= ~block_covers_bitmasks[add_b] # Still 0
                        current_cost += 1
                        # Now try to remove b
                        can_remove_b = True
                        for s in block_covers[b]:
                            found = False
                            for other_b in sub_to_blocks[s]:
                                if other_b != b and ((current_sel_mask >> other_b) & 1):
                                    found = True
                                    break
                            if not found:
                                can_remove_b = False
                                break
                        if can_remove_b:
                            current_sel_mask &= ~(1 << b)
                            current_cost -= 1
                        else:
                            # Revert add_b
                            current_sel_mask &= ~(1 << add_b)
                            current_cost -= 1
                            # Try a different random walk: flip a random selected block?
                            # No, that breaks validity.
                            # Let's just accept the current state and try again next step.
                            pass
            else:
                # No selected blocks? Should not happen if valid and n_t > 0.
                break
        else:
            # Invalid state
            # Find best block to add
            min_count = float('inf')
            target_sub = -1
            for s in range(n_t):
                if (current_unc_mask >> s) & 1:
                    c = len(sub_to_blocks[s])
                    if c < min_count:
                        min_count = c
                        target_sub = s
                        if c == 1:
                            break
            
            if target_sub == -1:
                break
                
            candidates = sub_to_blocks[target_sub]
            
            best_gain = -1
            best_block = -1
            second_best_gain = -1
            second_best_block = -1
            
            for b in candidates:
                if not ((current_sel_mask >> b) & 1):
                    gain = _popcount(block_covers_bitmasks[b] & current_unc_mask)
                    if gain > best_gain:
                        if best_gain > second_best_gain:
                            second_best_gain = best_gain
                            second_best_block = best_block
                        best_gain = gain
                        best_block = b
                    elif gain > second_best_gain:
                        second_best_gain = gain
                        second_best_block = b
            
            if best_block == -1:
                break
                
            if random.random() < p_walk:
                unselected_cands = [b for b in candidates if not ((current_sel_mask >> b) & 1)]
                if unselected_cands:
                    chosen_b = random.choice(unselected_cands)
                else:
                    chosen_b = best_block
            else:
                chosen_b = best_block
                
            old_unc = current_unc_mask
            current_sel_mask |= (1 << chosen_b)
            current_unc_mask &= ~block_covers_bitmasks[chosen_b]
            
            delta_unc = _popcount(old_unc) - _popcount(current_unc_mask)
            current_cost += 1 - alpha * delta_unc

    # Final result
    if best_sel_mask != 0:
        result_blocks = [i for i in range(n_blocks) if (best_sel_mask >> i) & 1]
        return [list(blocks[i]) for i in result_blocks]
    else:
        return []

