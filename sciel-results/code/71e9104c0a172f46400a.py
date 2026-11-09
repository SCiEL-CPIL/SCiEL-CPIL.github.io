"""Combined program: Lexicographic Greedy with Bitmask Efficiency and Local Search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _to_mask(combo):
    """Convert a tuple of elements to a bitmask."""
    mask = 0
    for x in combo:
        mask |= (1 << x)
    return mask

def _mask_to_list(mask):
    """Convert a bitmask back to a sorted list of elements."""
    return [i + 1 for i in range(16) if mask & (1 << i)]

def _has_sunflower(candidate_mask, chosen_masks):
    """
    Check if adding candidate_mask to chosen_masks creates a sunflower.
    A sunflower is a triple (A, B, C) such that A&B == A&C == B&C.
    We check if there exist A, B in chosen_masks such that:
    A & candidate == B & candidate == A & B
    """
    n = len(chosen_masks)
    if n < 2:
        return False
    
    # Precompute intersections of candidate with all chosen masks
    # inter_i = chosen_masks[i] & candidate_mask
    inter_with_cand = [m & candidate_mask for m in chosen_masks]
    
    # Check all pairs (i, j)
    for i in range(n):
        ac = inter_with_cand[i]
        a = chosen_masks[i]
        for j in range(i + 1, n):
            bc = inter_with_cand[j]
            # For a sunflower, we need A&C == B&C == A&B
            # So ac must equal bc
            if ac != bc:
                continue
            # And A&B must also equal ac
            ab = a & chosen_masks[j]
            if ab == ac:
                return True
    return False

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # 1. Generate all 4-subsets in lexicographic order and convert to bitmasks
    all_combos = list(combinations(range(16), 4))
    all_masks = [_to_mask(c) for c in all_combos]
    n_total = len(all_masks)
    
    chosen_masks = []
    chosen_set = set() # For O(1) membership check if needed, though we iterate by index
    
    # 2. Deterministic Lexicographic Greedy Pass
    # This is the core structural idea from Inspiration 1, optimized with bitmasks from Current Program
    for idx in range(n_total):
        if time.time() > deadline - 10:
            break
        mask = all_masks[idx]
        if mask in chosen_set:
            continue
            
        if not _has_sunflower(mask, chosen_masks):
            chosen_masks.append(mask)
            chosen_set.add(mask)
            
    # 3. Local Search / Refinement Phase
    # Try to add any remaining sets that might have been skipped or try swaps.
    # The greedy pass adds sets in lex order. Sometimes a later set could fit if an earlier one was removed?
    # Or simply, did we miss any? No, greedy checks all. 
    # However, we can try to "fill in" gaps if we removed any (we didn't in greedy).
    # Let's try a simple improvement: iterate through unchosen sets again. 
    # In a pure greedy, if a set was rejected, it's because of existing pairs. 
    # Removing a set might allow two others to be added.
    
    # Strategy: Try swaps. Pick a random set in chosen, remove it, see if we can add 2 new sets.
    # To keep it fast and deterministic-ish, we'll do a limited number of swap attempts.
    
    improved = True
    swap_attempts = 0
    max_swap_attempts = 500
    
    while improved and swap_attempts < max_swap_attempts and time.time() < deadline - 5:
        improved = False
        swap_attempts += 1
        
        # Try to add any unchosen sets first (cheap check)
        # This handles cases where the initial greedy might have been blocked by a set that is now "less blocking" 
        # (though in pure addition, nothing is removed, so this is redundant unless we did removals).
        # Since we only do swaps, let's focus on swaps.
        
        if len(chosen_masks) < 2:
            break
            
        # Randomly pick a set to remove
        # To be structured, let's iterate through a random permutation of indices for the first few tries
        # but for efficiency, just pick random indices.
        
        # We need to be careful: if we remove a set, we must ensure the remaining set is still sunflower-free.
        # It is, because it was a subset of a sunflower-free family.
        
        # Pick a candidate index to remove
        rem_idx = random.randint(0, len(chosen_masks) - 1)
        removed_mask = chosen_masks[rem_idx]
        
        # Create a temporary list without the removed mask
        # To avoid O(N) copy every time, we can just pass a flag or use a list comprehension.
        # Given N is small (~200-250), copying is fine.
        temp_chosen = [m for i, m in enumerate(chosen_masks) if i != rem_idx]
        temp_set = chosen_set - {removed_mask}
        
        # Find unchosen masks
        # We want to add 2 masks from the unchosen pool that don't conflict with temp_chosen or each other.
        
        # Get list of unchosen masks
        # To make this faster, we can pre-compute the list of all masks not in chosen_set.
        unchosen_masks = [m for m in all_masks if m not in temp_set]
        
        # Shuffle to find a good pair quickly
        random.shuffle(unchosen_masks)
        
        added = 0
        new_chosen = list(temp_chosen)
        new_set = set(temp_set)
        
        # Try to add 2 sets
        for mask in unchosen_masks:
            if added >= 2:
                break
            # Check if mask can be added to new_chosen
            if not _has_sunflower(mask, new_chosen):
                new_chosen.append(mask)
                new_set.add(mask)
                added += 1
                
        if added >= 2:
            # Accept the swap: we removed 1, added 2, net gain +1
            chosen_masks = new_chosen
            chosen_set = new_set
            improved = True
        elif added == 1:
            # Net gain 0. We could accept it if we want to change the structure for future swaps,
            # but it risks getting stuck. Let's only accept if we can add 2 (net +1) or if we are stuck.
            # For now, only accept net positive changes to ensure progress.
            pass
        else:
            # No improvement from this removal
            pass

    # Final check: try to add any remaining sets one by one (in case the swap phase left room)
    # This is a cheap "fill" step.
    for mask in all_masks:
        if mask in chosen_set:
            continue
        if not _has_sunflower(mask, chosen_masks):
            chosen_masks.append(mask)
            chosen_set.add(mask)

    # Convert back to lists
    result = [_mask_to_list(m) for m in chosen_masks]
    
    return result
# EVOLVE-BLOCK-END
