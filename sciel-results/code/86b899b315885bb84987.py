"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy initial construction: Iterate moduli 20..200, adding a modulus if a valid residue exists.
    2. Local Search (Hill Climbing): Try to improve the set of moduli by:
       a. Adding a missing modulus if a valid residue can be found.
       b. Swapping a selected modulus (with lower 1/m) for a missing modulus (with higher 1/m) 
          if the swap results in a valid assignment and higher total score.
    3. Repeat until no improvement is found or time limit approaches.
    """
    deadline = time.time() + 140.0
    
    # Helper: Given a list of (m, r) pairs, check if a new (m_new, r_new) is valid
    def is_valid(candidate_m, candidate_r, current_classes):
        for m_i, r_i in current_classes:
            g = gcd(candidate_m, m_i)
            if candidate_r % g == r_i % g:
                return False
        return True

    # Helper: Find any valid residue for m given current_classes. Returns r or None.
    def find_valid_residue(m, current_classes):
        # To maximize future flexibility, we might want a specific residue, 
        # but for now, any valid one works for the score. 
        # However, the choice of residue affects which OTHER moduli can be added later.
        # A simple heuristic: try residues 0..m-1. 
        # To improve search, we could try to pick the residue that conflicts with the fewest 
        # potential future moduli, but that's expensive. Let's stick to first-found for speed,
        # but note that in local search, we re-evaluate the whole set.
        for r in range(m):
            if is_valid(m, r, current_classes):
                return r
        return None

    # Helper: Calculate score
    def calc_score(classes):
        return sum(1.0 / m for m, _ in classes)

    # Initial Greedy Phase
    # Start with an empty set
    selected = []
    
    # We iterate moduli in ascending order (small m = high value)
    # But we should be careful: adding a small m might block many others.
    # The greedy "add if possible" is a good start.
    for m in range(20, 201):
        if time.time() > deadline - 10.0:
            break
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append((m, r))

    current_score = calc_score(selected)
    best_selected = [tuple(c) for c in selected]
    best_score = current_score

    # Local Search Phase
    # We represent the state as a list of (m, r).
    # Moves:
    # 1. Try to add a modulus not in selected.
    # 2. Try to swap a modulus in selected with one not in selected.
    
    # Precompute list of all moduli
    all_moduli = list(range(20, 201))
    
    # To make swap evaluation faster, we can cache gcds or just rely on the small N (180 moduli).
    # N is small, so O(N^2) per step is fine.
    
    improvement_found = True
    while improvement_found and time.time() < deadline - 5.0:
        improvement_found = False
        
        # 1. Try Additions
        current_m_set = set(m for m, _ in selected)
        for m_new in all_moduli:
            if m_new in current_m_set:
                continue
            # Try to find a residue for m_new that doesn't conflict
            # We need to check against current selected.
            # If we find one, we add it.
            # Note: Adding a modulus increases score by 1/m_new. Always positive.
            # So if we can add, we should.
            r_new = find_valid_residue(m_new, selected)
            if r_new is not None:
                new_selected = selected + [(m_new, r_new)]
                new_score = current_score + 1.0 / m_new
                # Since we are just adding, and validity is checked, score always increases.
                selected = new_selected
                current_score = new_score
                current_m_set.add(m_new)
                improvement_found = True
                # Update best if better
                if current_score > best_score:
                    best_score = current_score
                    best_selected = [tuple(c) for c in selected]
        
        # 2. Try Swaps
        # We look for a pair (m_old, m_new) where m_old is in selected, m_new is not,
        # and 1/m_new > 1/m_old (i.e., m_new < m_old).
        # We remove m_old, then try to find a residue for m_new.
        # If valid, we swap.
        
        # To be efficient, iterate potential m_new (smaller moduli) and m_old (larger moduli).
        # Actually, any swap where the new modulus is smaller is beneficial in terms of score, 
        # PROVIDED it is feasible.
        
        # Let's iterate through all missing moduli m_new.
        # For each, iterate through all selected moduli m_old where m_old > m_new.
        # Try to swap.
        
        # Optimization: Sort missing moduli ascending (highest value first).
        missing_moduli = [m for m in all_moduli if m not in current_m_set]
        missing_moduli.sort() # Smallest first (highest value)
        
        selected_moduli = [m for m, _ in selected]
        
        # We need to be careful with the order of updates in the swap loop to avoid using stale data.
        # We'll collect potential swaps and apply the best one, or just apply the first one that works?
        # Hill climbing usually applies one move at a time.
        
        # Let's try to find the "best" swap (max score gain) in one pass? 
        # Or just first valid? First valid is faster. Since all valid swaps with m_new < m_old increase score,
        # the magnitude doesn't matter for validity, only for how much we improve.
        # However, a swap with a very small m_new might block future additions more than a swap with a slightly larger m_new.
        # But we are in a loop, so we'll keep searching.
        
        # Let's just try to find ANY improving swap.
        swapped = False
        for m_new in missing_moduli:
            if swapped:
                break
            # We only care about m_new < m_old for score increase
            for i, (m_old, r_old) in enumerate(selected):
                if m_new >= m_old:
                    continue # No score gain (or loss if m_new > m_old, which we avoid)
                
                # Remove m_old temporarily
                temp_selected = selected[:i] + selected[i+1:]
                
                # Try to find residue for m_new in temp_selected
                r_new = find_valid_residue(m_new, temp_selected)
                if r_new is not None:
                    # Swap is valid
                    new_selected = temp_selected + [(m_new, r_new)]
                    # Score change: -1/m_old + 1/m_new. Since m_new < m_old, this is positive.
                    new_score = current_score - 1.0/m_old + 1.0/m_new
                    
                    selected = new_selected
                    current_score = new_score
                    current_m_set.discard(m_old)
                    current_m_set.add(m_new)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = [tuple(c) for c in selected]
                    
                    improvement_found = True
                    swapped = True
                    break # Restart outer loop or just break inner? 
                    # Breaking inner means we try next m_new. 
                    # But the set has changed. It's safer to break both and restart the while loop logic?
                    # Or just continue checking other swaps on the NEW set?
                    # Let's break the inner loop and continue to next m_new in the SAME iteration?
                    # No, the list `selected` has changed, so indices in the outer loop `for i...` are stale.
                    # So we must break the outer `for m_new` loop as well if we modified the set?
                    # Actually, `missing_moduli` is static for this iteration of `while`.
                    # If we swap, `selected` changes. The next `m_new` in `missing_moduli` might still be missing.
                    # But the `selected` list indices are messed up for the inner loop.
                    # So, if we swap, we should break out of the swap-finding block and restart the `while` loop?
                    # Or just set a flag and break.
                    pass
        
        # If we made a swap, we break out of the "try swaps" section and go back to top of while loop
        # to re-evaluate additions and swaps with the new state.
        if swapped:
            continue # Go to next iteration of while loop
            
    # Return the best result found
    # Convert to required format [[m, r], ...]
    return [[m, r] for m, r in best_selected]
# EVOLVE-BLOCK-END
