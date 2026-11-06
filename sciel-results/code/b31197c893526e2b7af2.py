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
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute GCD table for moduli 20..200
    moduli = list(range(20, 201))
    gcd_table = {}
    for m1 in moduli:
        for m2 in moduli:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def _find_valid_residues(m, selected):
        """
        Find all valid residues for modulus m given the selected classes.
        Returns a list of valid residues.
        """
        forbidden = [False] * m
        
        for m_prime, r_prime in selected:
            g = gcd_table.get((m, m_prime), math.gcd(m, m_prime))
            # Residues r such that r % g == r_prime % g are forbidden
            # r = r_prime % g + k * g
            start_r = r_prime % g
            for r in range(start_r, m, g):
                forbidden[r] = True
        
        valid_residues = [r for r in range(m) if not forbidden[r]]
        return valid_residues

    def _greedy_construct(order, seed):
        """
        Greedy construction using the given order of moduli.
        Uses a specific random seed to choose among valid residues.
        """
        random.seed(seed)
        selected = []
        
        for m in order:
            valid_r = _find_valid_residues(m, selected)
            if valid_r:
                # Pick a random valid residue to diversify the search
                r = random.choice(valid_r)
                selected.append([m, r])
        
        return selected

    def _score(selected):
        """Compute the score of a selection."""
        return sum(1.0 / m for m, r in selected)

    def _local_improve(selected, deadline):
        """
        Local improvement:
        1. Try to add any missing modulus that fits directly.
        2. Try swaps: replace a larger modulus with a smaller missing one.
        """
        selected_moduli = set(m for m, r in selected)
        improved = True
        # Limit iterations to prevent infinite loops
        max_iter = 100
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Phase 1: Try to add missing moduli directly
            # Sort missing moduli by value (smaller m first)
            missing_moduli = [m for m in range(20, 201) if m not in selected_moduli]
            missing_moduli.sort()
            
            for m_new in missing_moduli:
                if time.time() > deadline:
                    break
                
                valid_r = _find_valid_residues(m_new, selected)
                if valid_r:
                    # Pick first valid residue (or random, but first is faster)
                    r_new = valid_r[0]
                    selected.append([m_new, r_new])
                    selected_moduli.add(m_new)
                    improved = True
                    # Restart the loop to check for more additions
                    break
            
            # Phase 2: If no direct addition, try swaps
            if not improved and missing_moduli:
                # Try to swap a selected modulus with a missing one
                # We want to swap out a large m (low value) for a small m (high value)
                # So we iterate missing moduli in increasing order (high value)
                # and selected moduli in decreasing order (low value)
                
                selected_list = sorted(selected, key=lambda x: -x[0]) # Largest m first
                swapped = False
                
                for m_new in missing_moduli:
                    if time.time() > deadline or swapped:
                        break
                    
                    # Try to swap with any selected modulus larger than m_new
                    for i in range(len(selected_list)):
                        m_old, r_old = selected_list[i]
                        if m_old <= m_new:
                            break # Since sorted descending, no more larger moduli
                    
                        # Temporarily remove m_old
                        # Find index in 'selected'
                        idx = None
                        for j in range(len(selected)):
                            if selected[j][0] == m_old:
                                idx = j
                                break
                        if idx is None: continue
                        
                        # Remove
                        selected.pop(idx)
                        
                        # Try to add m_new
                        valid_r = _find_valid_residues(m_new, selected)
                        if valid_r:
                            selected.append([m_new, valid_r[0]])
                            selected_moduli.remove(m_old)
                            selected_moduli.add(m_new)
                            improved = True
                            swapped = True
                            break
                        else:
                            # Revert
                            selected.insert(idx, [m_old, r_old])
                
        return selected

    # Run multiple restarts with different orderings and seeds
    best_solution = []
    best_score = 0.0
    
    # Initial greedy with natural order (smallest first)
    moduli_natural = list(range(20, 201))
    
    # We will run a number of greedy passes with different random seeds
    # to diversify residue choices, then pick the best few for local improvement.
    
    num_greedy = 50
    greedy_solutions = []
    
    for i in range(num_greedy):
        if time.time() > deadline - 10: # Leave time for local improve
            break
        sol = _greedy_construct(moduli_natural, i)
        greedy_solutions.append(sol)
    
    # Sort greedy solutions by score and keep top K
    greedy_solutions.sort(key=_score, reverse=True)
    top_k = min(10, len(greedy_solutions))
    
    for i in range(top_k):
        if time.time() > deadline - 2:
            break
        
        sol = greedy_solutions[i]
        # Deep copy to avoid modification issues
        sol_copy = [list(x) for x in sol]
        
        sol_copy = _local_improve(sol_copy, deadline - 0.5)
        score = _score(sol_copy)
        
        if score > best_score:
            best_score = score
            best_solution = sol_copy
            
    # If no solution was found (shouldn't happen), fallback to simple greedy
    if not best_solution:
        best_solution = _greedy_construct(moduli_natural, 0)

    return best_solution
