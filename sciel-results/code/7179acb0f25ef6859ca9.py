import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for safety

    def is_compatible(selected, m, r):
        """Check if residue r for modulus m is compatible with all selected classes."""
        for (m_i, r_i) in selected:
            g = math.gcd(m, m_i)
            if r % g == r_i % g:
                return False
        return True

    def score(selected):
        """Calculate the sum of 1/m for selected classes."""
        return sum(1.0 / m for (m, r) in selected)

    # Greedy phase: iterate through moduli from smallest to largest
    # Prioritize smaller moduli since they have higher reciprocal values
    selected = []
    for m in range(20, 201):
        if time.time() > deadline:
            break
        
        # Try residues in order - prefer smaller residues for determinism
        found = False
        for r in range(m):
            if is_compatible(selected, m, r):
                selected.append((m, r))
                found = True
                break
    
    # Local search phase: try to improve by swapping residues or moduli
    best_score = score(selected)
    best_selected = list(selected)
    
    # Try swapping residues for existing moduli
    for i in range(len(selected)):
        if time.time() > deadline:
            break
        
        m_i = selected[i][0]
        old_r_i = selected[i][1]
        
        # Try all other residues for this modulus
        for new_r in range(m_i):
            if new_r == old_r_i:
                continue
            
            # Check if new residue is compatible with all other selected classes
            compatible = True
            for j in range(len(selected)):
                if j == i:
                    continue
                m_j, r_j = selected[j]
                g = math.gcd(m_i, m_j)
                if new_r % g == r_j % g:
                    compatible = False
                    break
            
            if compatible:
                # This swap is valid - update
                selected[i] = (m_i, new_r)
                new_score = score(selected)
                if new_score > best_score:
                    best_score = new_score
                    best_selected = list(selected)
                # Keep the swap if it doesn't hurt (we're just exploring)
                # Actually, let's be more careful - only keep if it improves or is neutral
                # For now, revert and continue searching
                selected[i] = (m_i, old_r_i)
    
    # Try adding moduli that were skipped
    current_moduli = set(m for (m, r) in best_selected)
    for m in range(20, 201):
        if time.time() > deadline:
            break
        
        if m in current_moduli:
            continue
        
        # Try to find a compatible residue for this modulus
        for r in range(m):
            if is_compatible(best_selected, m, r):
                # Test if adding this improves the score
                test_selected = list(best_selected) + [(m, r)]
                test_score = score(test_selected)
                if test_score > best_score:
                    best_score = test_score
                    best_selected = test_selected
                break
    
    # Try removing and re-adding moduli with different residues
    for i in range(len(best_selected)):
        if time.time() > deadline:
            break
        
        m_i, r_i = best_selected[i]
        
        # Remove this modulus
        temp_selected = [x for j, x in enumerate(best_selected) if j != i]
        
        # Try to find a better residue for the same modulus
        for new_r in range(m_i):
            if is_compatible(temp_selected, m_i, new_r):
                test_selected = temp_selected + [(m_i, new_r)]
                test_score = score(test_selected)
                if test_score > best_score:
                    best_score = test_score
                    best_selected = test_selected
                break
        
        # Also try different moduli
        for m_new in range(20, 201):
            if time.time() > deadline:
                break
            
            if m_new == m_i or m_new in [x[0] for x in temp_selected]:
                continue
            
            for r_new in range(m_new):
                if is_compatible(temp_selected, m_new, r_new):
                    test_selected = temp_selected + [(m_new, r_new)]
                    test_score = score(test_selected)
                    if test_score > best_score:
                        best_score = test_score
                        best_selected = test_selected
                    break
    
    return [[m, r] for (m, r) in best_selected]
# EVOLVE-BLOCK-END
