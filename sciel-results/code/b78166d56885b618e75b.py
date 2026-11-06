import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Candidate moduli in [20, 200]
    # We want to maximize sum(1/m), so smaller moduli are more valuable.
    # However, smaller moduli have fewer residues and are more likely to conflict.
    # We will use a heuristic that balances value and feasibility.
    
    all_moduli = list(range(20, 201))
    
    # We'll use a "Max-Weight Bipartite Matching" like approach.
    # Left side: Moduli. Right side: Residues? No, residues depend on modulus.
    # Instead, we view it as: Select a subset of moduli S and assign residues r_m.
    # Constraint: For all m1, m2 in S, r_m1 % gcd(m1, m2) != r_m2 % gcd(m1, m2).
    
    # This is equivalent to finding a large set of "compatible" moduli.
    # A simple greedy fails because it commits early.
    # A better strategy:
    # 1. Sort moduli by 1/m descending (smallest m first).
    # 2. Use a backtracking search with pruning, but limit depth/time.
    # 3. Or, use a local search on a set of selected moduli.
    
    # Let's try a randomized local search / simulated annealing style approach.
    # State: A set of (m, r) pairs.
    # Objective: Sum(1/m).
    # Moves: 
    #   1. Add a new modulus m with a valid residue r.
    #   2. Remove a modulus m.
    #   3. Change residue r for an existing modulus m (if still valid).
    #   4. Swap: Remove m1, add m2.
    
    # To make it efficient, we need a fast validity check.
    # Precompute gcds? Or just check on the fly.
    # N is at most 181. Pairwise check is O(N^2) per step. 181^2 ~ 32k ops. 
    # 120s is plenty for thousands of steps.
    
    # Strategy:
    # 1. Initialize with a greedy solution (evens, or all sorted by 1/m).
    # 2. Iteratively try to improve:
    #    a. Try to add a missing modulus (sorted by 1/m desc).
    #    b. Try to reassign residues to free up space for higher value moduli.
    #    c. Try to remove low-value moduli to make room for high-value ones.
    
    # Let's implement a more robust local search.
    
    def get_score(selected):
        return sum(1.0 / m for m, r in selected)
    
    def is_valid(m, r, selected):
        for m_sel, r_sel in selected:
            g = gcd(m, m_sel)
            if r % g == r_sel % g:
                return False
        return True
        
    def try_add(selected, m):
        # Try to find a valid residue for m
        # Randomize residue order to escape local minima
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_valid(m, r, selected):
                return (m, r)
        return None

    # Initial solution: Greedy on all moduli sorted by 1/m desc
    # This is the same as sorting by m asc.
    selected = []
    for m in sorted(all_moduli, key=lambda x: -1.0/x):
        if time.time() > deadline: break
        pair = try_add(selected, m)
        if pair:
            selected.append(pair)
            
    best_score = get_score(selected)
    best_selected = [x for x in selected]
    
    # Local Search Phase
    # We will try to improve the score by swapping low-value moduli for high-value ones
    # or by re-optimizing residues.
    
    # Sort current selection by value descending (smallest m first)
    # Identify "low value" moduli (large m) that might be blocking "high value" ones.
    
    iterations = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        iterations += 1
        current_score = get_score(selected)
        
        # 1. Try to add any missing high-value modulus
        added = False
        # Check moduli in order of 1/m desc
        for m in sorted(all_moduli, key=lambda x: -1.0/x):
            if any(sm == m for sm, sr in selected):
                continue
            # Try to add m
            pair = try_add(selected, m)
            if pair:
                selected.append(pair)
                added = True
                break
        
        if added:
            no_improve_count = 0
            continue
        
        # 2. Try to swap a low-value modulus with a high-value one
        # Get list of selected moduli sorted by value (asc, so lowest value first)
        selected_sorted = sorted(selected, key=lambda x: 1.0/x)
        
        swapped = False
        # Try removing the lowest value modulus (largest m)
        if selected_sorted:
            low_m, low_r = selected_sorted[0]
            # Try to add a higher value modulus that is currently missing
            # Sort missing moduli by value desc
            missing = [m for m in all_moduli if not any(sm == m for sm, sr in selected)]
            missing.sort(key=lambda x: -1.0/x)
            
            for m_new in missing:
                if 1.0/m_new <= 1.0/low_m:
                    break # No better value available
                
                # Tentatively remove low_m
                selected.remove((low_m, low_r))
                
                # Try to add m_new
                pair_new = try_add(selected, m_new)
                if pair_new:
                    selected.append(pair_new)
                    swapped = True
                    break
                else:
                    # Restore low_m
                    selected.append((low_m, low_r))
            
            if swapped:
                no_improve_count = 0
                continue
        
        # 3. Try to reassign residues to potentially allow additions
        # This is harder. Let's just randomize a residue of a random selected modulus
        # and see if it helps future additions.
        if selected:
            idx = random.randint(0, len(selected)-1)
            m, r = selected[idx]
            # Try a different residue
            new_r = random.randint(0, m-1)
            if new_r != r:
                selected[idx] = (m, new_r)
                # Check if this new assignment allows adding a missing modulus
                # (We don't actually add it here, just see if the state is "better" in some sense?
                # No, we need to measure score. Reassignment doesn't change score directly.
                # It only changes feasibility for future adds.
                # So we should only keep it if it leads to an add?
                # Let's revert for now and rely on the add/swap logic.
                selected[idx] = (m, r)
        
        # 4. If no improvement, count it
        if not added and not swapped:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a more aggressive move: remove 2 low value, add 1 high value?
                # Or just break to save time if stuck.
                # Let's try to remove 2 lowest and add best missing.
                if len(selected) >= 2:
                    s1 = selected_sorted[0]
                    s2 = selected_sorted[1]
                    selected.remove(s1)
                    selected.remove(s2)
                    
                    added_pair = None
                    for m_new in missing:
                        pair_new = try_add(selected, m_new)
                        if pair_new:
                            added_pair = pair_new
                            break
                    if added_pair:
                        selected.append(added_pair)
                        no_improve_count = 0
                        continue
                    else:
                        selected.append(s1)
                        selected.append(s2)
            
        # Update best
        if get_score(selected) > best_score:
            best_score = get_score(selected)
            best_selected = [x for x in selected]
            
    # Final check: Ensure best_selected is valid and returns the best found
    # (It should be valid by construction, but just in case)
    # We return best_selected
    
    # Convert to required format
    result = [[m, r] for m, r in best_selected]
    
    # Fallback
    if not result:
        result = [[20, 0]]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} classes")
    print(f"Score: {sum(1.0/m for m, r in res)}")
