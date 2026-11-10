"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Greedy: try to include all moduli from 20 to 200
    # Process in order of increasing m (highest weight 1/m first)
    selected = []  # list of [m, r]
    
    for m in range(20, 201):
        # Check time limit
        if time.time() > deadline:
            break
        
        # Try each residue 0, 1, ..., m-1
        found = False
        for r in range(m):
            # Check compatibility with all selected classes
            compatible = True
            for m2, r2 in selected:
                g = gcd(m, m2)
                if r % g == r2 % g:
                    compatible = False
                    break
            if compatible:
                selected.append([m, r])
                found = True
                break
        # If no compatible residue found, skip this modulus
    
    # Try to improve: for each selected class, see if we can swap in a better configuration
    # Actually, let's try a local search: remove one element and try to re-add with different residue
    # But first, let's try a second pass with different ordering or residue preferences
    
    # Second pass: try to add any skipped moduli with different residue preferences
    # We can also try to improve by reassigning residues
    
    # Let's do a simple local search: try removing each element and see if we can add back
    # a different modulus or same modulus with different residue that allows more additions
    
    # Actually, the greedy approach in increasing m order should be quite good since
    # smaller m have larger 1/m weights. Let's verify we're getting a good result.
    
    # Let's try one more improvement: for each modulus we couldn't include, try all residues
    # and see if removing one existing element and adding this one improves the sum
    
    # Compute current score
    def score(sel):
        return sum(1.0/m for m, r in sel)
    
    current_score = score(selected)
    
    # Try local search: for each pair (i, j) where i is in selected and j is not,
    # see if removing i and adding j (with some residue) improves the score
    # This is O(n^2 * m * n) which might be slow, so let's be careful
    
    # Actually, let's just try a simple hill-climbing: 
    # For each modulus not in selected, try to add it by removing one element
    # that conflicts with it, and see if the net change is positive
    
    moduli_in_selected = set(m for m, r in selected)
    all_moduli = set(range(20, 201))
    missing_moduli = sorted(all_moduli - moduli_in_selected)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for m_new in missing_moduli:
            if time.time() > deadline:
                break
            
            # Try to find a residue for m_new that is compatible with most of selected
            # but we need it to be compatible with ALL of selected (after removing some)
            # Strategy: find which elements of selected conflict with m_new for each residue
            # Then try removing the one element whose removal allows a valid residue
            # and check if 1/m_new > 1/m_removed
            
            for r_new in range(m_new):
                # Check which selected elements conflict with (m_new, r_new)
                conflicts = []
                for idx, (m2, r2) in enumerate(selected):
                    g = gcd(m_new, m2)
                    if r_new % g == r2 % g:
                        conflicts.append(idx)
                
                if len(conflicts) == 0:
                    # Can add directly without removing anything
                    new_score = current_score + 1.0/m_new
                    if new_score > current_score:
                        selected.append([m_new, r_new])
                        current_score = new_score
                        missing_moduli = [x for x in missing_moduli if x != m_new]
                        improved = True
                        break
                elif len(conflicts) == 1:
                    # Remove one element and add this one
                    idx_remove = conflicts[0]
                    m_remove, r_remove = selected[idx_remove]
                    if 1.0/m_new > 1.0/m_remove:
                        # Remove the conflicting element and add the new one
                        selected.pop(idx_remove)
                        selected.append([m_new, r_new])
                        current_score = current_score - 1.0/m_remove + 1.0/m_new
                        missing_moduli = [x for x in missing_moduli if x != m_new]
                        improved = True
                        break
            
            if improved:
                break
    
    return selected
# EVOLVE-BLOCK-END
