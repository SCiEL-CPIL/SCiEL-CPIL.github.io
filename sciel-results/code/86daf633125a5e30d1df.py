import math
import time
import random

def construct():
    """
    Constructs a set of pairwise disjoint residue classes with distinct moduli in [20, 200]
    maximizing the sum of 1/m.
    
    Returns:
        list of [m, r] pairs
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    # Precompute GCD table for moduli 20..200
    moduli = list(range(20, 201))
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def _is_compatible(m, r, selected):
        """Check if (m, r) is compatible with all entries in selected."""
        for m_prime, r_prime in selected:
            g = math.gcd(m, m_prime)
            if r % g == r_prime % g:
                return False
        return True
    
    def _score(selected):
        """Compute the score of a selection."""
        return sum(1.0 / m for m, r in selected)
    
    def _greedy_construct(order, deadline, gcd_table):
        """Greedy construction using the given order of moduli."""
        selected = []
        for m in order:
            if time.time() > deadline:
                break
            # Try residues in order 0, 1, ..., m-1
            for r in range(m):
                if _is_compatible(m, r, selected):
                    selected.append([m, r])
                    break
        return selected
    
    def _local_improve(selected, deadline, gcd_table):
        """Local improvement: try to swap residues and add missing moduli."""
        current_score = _score(selected)
        selected_moduli = set(m for m, r in selected)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add missing moduli by rearranging existing ones
            for m_new in range(20, 201):
                if m_new in selected_moduli:
                    continue
                
                # Try each residue for m_new
                for r_new in range(m_new):
                    # Check if we can fit m_new by removing one existing class
                    for i in range(len(selected)):
                        m_i, r_i = selected[i]
                        # Remove i temporarily
                        temp = selected[:i] + selected[i+1:]
                        # Try to find a residue for m_new compatible with temp
                        compatible = True
                        for m_prime, r_prime in temp:
                            g = math.gcd(m_new, m_prime)
                            if r_new % g == r_prime % g:
                                compatible = False
                                break
                        if compatible:
                            # Check if the removed class can be re-inserted elsewhere
                            # Actually, we're just swapping: remove i, add m_new
                            # But we need to make sure m_i isn't lost permanently
                            # For now, just add m_new if it fits without removing
                            pass
                    
                    # Try adding without removing
                    if _is_compatible(m_new, r_new, selected):
                        selected.append([m_new, r_new])
                        selected_moduli.add(m_new)
                        improved = True
                        break
                
                if improved:
                    break
            
            # Try swapping residues to free up space
            for i in range(len(selected)):
                m_i, r_i = selected[i]
                for r_alt in range(m_i):
                    if r_alt == r_i:
                        continue
                    # Try changing residue of i to r_alt
                    old_r = selected[i][1]
                    selected[i][1] = r_alt
                    # Check if this frees up space for a missing modulus
                    for m_new in range(20, 201):
                        if m_new in selected_moduli:
                            continue
                        for r_new in range(m_new):
                            if _is_compatible(m_new, r_new, selected):
                                # Good swap: add m_new
                                selected.append([m_new, r_new])
                                selected_moduli.add(m_new)
                                improved = True
                                break
                        if improved:
                            break
                    if not improved:
                        # Revert
                        selected[i][1] = old_r
                    if improved:
                        break
                if improved:
                    break
        
        return selected
    
    # Run multiple restarts with different orderings
    best_solution = []
    best_score = 0.0
    
    # Initial greedy with natural order (smallest first)
    moduli_natural = list(range(20, 201))
    sol1 = _greedy_construct(moduli_natural, deadline, gcd_table)
    sol1 = _local_improve(sol1, deadline, gcd_table)
    score1 = _score(sol1)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1
    
    # Random restarts with shuffled orders
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline - 3:
            break
            
        moduli_shuffled = list(range(20, 201))
        random.shuffle(moduli_shuffled)
        
        sol = _greedy_construct(moduli_shuffled, deadline - 2, gcd_table)
        
        # Local improvement if time permits
        if time.time() < deadline - 5:
            sol = _local_improve(sol, deadline - 1, gcd_table)
        
        score = _score(sol)
        if score > best_score:
            best_score = score
            best_solution = sol
    
    return best_solution
