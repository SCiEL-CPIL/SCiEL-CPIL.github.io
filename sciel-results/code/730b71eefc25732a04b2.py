import time
from math import gcd
from fractions import Fraction
import random

# EVOLVE-BLOCK-START

def _precompute_gcds(max_m):
    """Precompute gcd for all pairs in [20, max_m]."""
    gcds = {}
    for i in range(20, max_m + 1):
        for j in range(20, max_m + 1):
            if i not in gcds:
                gcds[i] = {}
            gcds[i][j] = gcd(i, j)
    return gcds

def _is_compatible_fast(m, r, chosen, gcds):
    """Check if (m, r) is compatible with all chosen (m_i, r_i) using precomputed GCDs."""
    for (m_i, r_i) in chosen:
        g = gcds[m][m_i]
        if r % g == r_i % g:
            return False
    return True

def _find_compatible_residue(m, chosen, gcds):
    """
    Find a residue r for m that is compatible with 'chosen'.
    Returns a compatible residue or None if none exists.
    We iterate through residues. To be deterministic and potentially better for future steps,
    we can just pick the first one or a random one. 
    Given the set is fixed, any valid residue is equally good for the current score.
    However, different residues might allow different future additions.
    For now, we return the first found to be fast, or random for diversity.
    Let's use random to avoid biasing towards low residues which might block others?
    Actually, low residues are often more "constrained" in a way? No.
    Let's stick to finding *a* valid residue.
    """
    # Optimization: If m is coprime to all m_i in chosen, any r works.
    # If not, we check.
    
    # Quick check: if chosen is empty, return 0
    if not chosen:
        return 0

    # Check if there's any compatible residue
    # We can just iterate. m <= 200, so max 200 checks.
    compatible_rs = []
    for r in range(m):
        if _is_compatible_fast(m, r, chosen, gcds):
            compatible_rs.append(r)
            
    if not compatible_rs:
        return None
    
    # Random choice to explore different paths in restarts
    return random.choice(compatible_rs)

def _greedy_construct(gcds, random_order=False):
    """
    Construct a solution by iterating moduli.
    If random_order is False, iterate 20..200.
    If True, iterate in a shuffled order but biased towards small? 
    Actually, adding small moduli first is generally better because they are more restrictive 
    (fewer residues) and have higher weight. 
    So we always try to add small moduli first.
    """
    moduli = list(range(20, 201))
    if random_order:
        # Shuffle but keep small ones prioritized? 
        # Let's just shuffle for diversity in restarts, but the "add if fits" 
        # will naturally skip large ones if they block small ones? 
        # No, if we add a large one first, it might block a small one later.
        # So we should always try small ones first.
        # Let's stick to ascending order for the main greedy.
        # Randomness comes from the residue choice.
        pass
        
    chosen = []
    for m in moduli:
        r = _find_compatible_residue(m, chosen, gcds)
        if r is not None:
            chosen.append((m, r))
    return chosen

def _local_search(chosen, gcds, deadline):
    """
    Improve the solution by:
    1. Trying to add any missing modulus (ascending order).
    2. Trying to swap a larger modulus for a smaller missing one.
    """
    current_sum = sum(Fraction(1, m) for m, r in chosen)
    all_moduli = set(range(20, 201))
    selected_moduli = set(m for m, r in chosen)
    missing_moduli = sorted(list(all_moduli - selected_moduli))
    
    improved = True
    # Limit the number of full passes to prevent infinite loops if logic is flawed
    max_passes = 100
    
    while improved and time.time() < deadline:
        improved = False
        pass_count = 0
        
        # 1. Try to add a missing modulus
        # Iterate missing moduli in ascending order (highest weight first)
        # We need to re-scan because adding one might block/unblock others?
        # Adding a modulus only restricts others, it never unblocks them.
        # So if m is not compatible now, it won't be compatible later (unless we remove something).
        # So we can just iterate through missing once?
        # No, if we add m1, it might make m2 incompatible? Yes.
        # But if m2 was compatible before, it might not be after.
        # So we should check each missing modulus against the *current* chosen set.
        
        added_any = False
        # We iterate from smallest to largest.
        # If we add a small one, it's high value.
        for i, m_new in enumerate(missing_moduli):
            if time.time() > deadline:
                return chosen, float(current_sum)
            
            r_new = _find_compatible_residue(m_new, chosen, gcds)
            if r_new is not None:
                chosen.append((m_new, r_new))
                current_sum += Fraction(1, m_new)
                missing_moduli.pop(i)
                added_any = True
                # Restart the scan? 
                # Adding m_new might make other missing moduli incompatible.
                # But we are iterating ascending. 
                # If we add m_new, we should check if the *next* ones are still compatible.
                # Since we are in a for loop, we continue to the next one.
                # But the list `missing_moduli` changed. 
                # It's safer to break and restart the loop if we added something,
                # to ensure we always try the smallest available first.
                improved = True
                break # Restart loop to add more, starting from smallest again
                
        # 2. Try to swap a selected modulus for a smaller missing one
        if not improved and chosen and missing_moduli:
            # We want to replace a large m with a small m.
            # For each missing m_new (ascending), check if we can remove some m_old 
            # such that m_new becomes compatible with the rest.
            
            swap_made = False
            # Try missing moduli in ascending order (highest gain)
            for m_new in missing_moduli:
                if time.time() > deadline:
                    return chosen, float(current_sum)
                
                # Find a candidate m_old to remove
                # We want to remove the largest m_old that conflicts with m_new?
                # Or just any m_old that allows m_new to be added.
                # If we remove m_old, m_new must be compatible with chosen \ {m_old}.
                
                # Sort chosen by m descending to try removing large ones first
                # (since we want to replace large with small, removing a large one is good
                # if it allows adding a small one, but we also lose 1/m_old.
                # We only swap if 1/m_new > 1/m_old, i.e., m_new < m_old).
                
                candidates_to_remove = sorted([m for m, r in chosen], reverse=True)
                
                for m_old in candidates_to_remove:
                    if m_new >= m_old:
                        # No gain in swapping
                        continue
                    
                    # Find index of m_old
                    idx = None
                    for i, (m, r) in enumerate(chosen):
                        if m == m_old:
                            idx = i
                            break
                    if idx is None:
                        continue
                    
                    # Temporarily remove m_old
                    m_old_val, r_old_val = chosen[idx]
                    chosen.pop(idx)
                    
                    # Check if m_new is compatible with the rest
                    r_new = _find_compatible_residue(m_new, chosen, gcds)
                    if r_new is not None:
                        # Perform swap
                        chosen.append((m_new, r_new))
                        current_sum += Fraction(1, m_new) - Fraction(1, m_old_val)
                        
                        # Update missing_moduli
                        if m_new in missing_moduli:
                            missing_moduli.remove(m_new)
                        if m_old_val not in missing_moduli:
                            missing_moduli.append(m_old_val)
                        missing_moduli.sort()
                        
                        improved = True
                        swap_made = True
                        break # Restart loop with new state
                    
                    # If not compatible, restore m_old
                    chosen.insert(idx, (m_old_val, r_old_val))
                    
                    if swap_made:
                        break
                
                if swap_made:
                    break # Restart outer loop

    return chosen, float(current_sum)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s/150s limit
    
    gcds = _precompute_gcds(200)
    
    best_chosen = []
    best_score = 0.0
    
    # Run multiple greedy + local search iterations
    # The randomization is in _find_compatible_residue (random choice of valid residue)
    num_iterations = 50
    
    for i in range(num_iterations):
        if time.time() > deadline:
            break
        
        # Greedy construction
        # We don't randomize the order of moduli (always ascending) because small moduli are more critical.
        # Randomness comes from residue selection.
        chosen = _greedy_construct(gcds)
        
        # Local search
        chosen, score = _local_search(chosen, gcds, deadline)
        
        if score > best_score:
            best_score = score
            best_chosen = chosen
        
        # Early stop if we found a very high score?
        # The theoretical max is sum(1/m) for all m if they were all coprime.
        # But they aren't. 
        # Let's not early stop, just use the time.
        
    # If no solution found, fallback to simple greedy
    if not best_chosen:
        best_chosen = _greedy_construct(gcds)
        best_chosen, _ = _local_search(best_chosen, gcds, deadline)

    # Format result
    result = [[m, r] for m, r in best_chosen]
    return result

# EVOLVE-BLOCK-END
