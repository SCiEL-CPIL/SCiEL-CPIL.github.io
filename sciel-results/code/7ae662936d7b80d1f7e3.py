import random
import time
from math import gcd
from fractions import Fraction

# EVOLVE-BLOCK-START

def _is_compatible(m, r, chosen):
    """Check if (m, r) is compatible with all chosen (m_i, r_i)."""
    for (m_i, r_i) in chosen:
        g = gcd(m, m_i)
        if r % g == r_i % g:
            return False
    return True

def _find_compatible_residue(m, chosen):
    """
    Find a residue r for m that is compatible with 'chosen'.
    Returns a random compatible residue or None if none exists.
    """
    compatible_rs = []
    for r in range(m):
        if _is_compatible(m, r, chosen):
            compatible_rs.append(r)
    
    if not compatible_rs:
        return None
    
    return random.choice(compatible_rs)

def _greedy_construct():
    """
    Construct a solution by iterating moduli from 20 to 200.
    """
    moduli = list(range(20, 201))
    chosen = []
    for m in moduli:
        r = _find_compatible_residue(m, chosen)
        if r is not None:
            chosen.append((m, r))
    return chosen

def _local_search(chosen, deadline):
    """
    Improve the solution by:
    1. Trying to add any missing modulus.
    2. Trying to swap a larger modulus for a smaller one.
    """
    current_sum = sum(Fraction(1, m) for m, r in chosen)
    all_moduli = set(range(20, 201))
    selected_moduli = set(m for m, r in chosen)
    missing_moduli = sorted(list(all_moduli - selected_moduli))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add a missing modulus
        # Iterate missing moduli in ascending order (highest weight first)
        for m_new in missing_moduli:
            r_new = _find_compatible_residue(m_new, chosen)
            if r_new is not None:
                chosen.append((m_new, r_new))
                current_sum += Fraction(1, m_new)
                missing_moduli.remove(m_new)
                improved = True
                break # Restart loop to add more
        
        # 2. Try to swap a selected modulus for a smaller missing one
        if not improved and chosen and missing_moduli:
            # Try a limited number of random swaps to avoid getting stuck
            # We want to replace a large m with a small m.
            # Sort chosen by m descending, missing by m ascending
            candidates_to_remove = sorted([m for m, r in chosen], reverse=True)[:5]
            candidates_to_add = missing_moduli[:5]
            
            for m_old in candidates_to_remove:
                idx = None
                for i, (m, r) in enumerate(chosen):
                    if m == m_old:
                        idx = i
                        break
                if idx is None: continue
                
                m_old_val, r_old_val = chosen[idx]
                # Temporarily remove
                chosen.pop(idx)
                
                swap_made = False
                for m_new in candidates_to_add:
                    if m_new >= m_old_val: continue
                    
                    r_new = _find_compatible_residue(m_new, chosen)
                    if r_new is not None:
                        chosen.append((m_new, r_new))
                        current_sum += Fraction(1, m_new) - Fraction(1, m_old_val)
                        missing_moduli.remove(m_new)
                        improved = True
                        swap_made = True
                        break
                
                if not swap_made:
                    chosen.insert(idx, (m_old_val, r_old_val))
                
                if improved:
                    break # Restart loop with new state
            # If no swap made, loop continues but improved is False, so it will exit unless additions were made earlier
            # But we set improved=False at start. If we didn't add or swap, improved stays False.
            # However, if we tried swaps and failed, we should exit the while loop to avoid infinite loop.
            # The `while improved` handles this.

    return chosen, float(current_sum)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_chosen = []
    best_score = 0.0
    
    # Run multiple greedy + local search iterations
    num_iterations = 50
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
        
        # Greedy construction
        chosen = _greedy_construct()
        
        # Local search
        chosen, score = _local_search(chosen, deadline)
        
        if score > best_score:
            best_score = score
            best_chosen = chosen
        
        # Early stop if we found a very high score
        if best_score > 2.2:
            break

    # If no solution found, fallback to simple greedy
    if not best_chosen:
        best_chosen = _greedy_construct()
        best_chosen, _ = _local_search(best_chosen, deadline)

    # Format result
    result = [[m, r] for m, r in best_chosen]
    return result

# EVOLVE-BLOCK-END
