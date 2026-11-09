import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    
    # Precompute gcd
    gcd_table = {}
    for i in range(n_mod):
        for j in range(i, n_mod):
            gcd_table[(moduli[i], moduli[j])] = gcd(moduli[i], moduli[j])
            gcd_table[(moduli[j], moduli[i])] = gcd(moduli[i], moduli[j])
            
    def get_gcd(m1, m2):
        return gcd_table[(m1, m2)]

    def can_assign_residue(classes, m_new):
        """
        Try to find a valid residue for m_new given existing classes.
        Returns the residue if found, else None.
        classes: list of [m, r]
        """
        for r in range(m_new):
            valid = True
            for (m_old, r_old) in classes:
                g = get_gcd(m_new, m_old)
                if (r - r_old) % g == 0:
                    valid = False
                    break
            if valid:
                return r
        return None

    def is_valid_set(classes):
        """Verify the set is valid. Used for debugging or final check."""
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                g = get_gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True

    def score(classes):
        total = Fraction(0, 1)
        for m, r in classes:
            total += Fraction(1, m)
        return total

    # 1. Greedy Initialization
    # Sort moduli by 1/m descending
    sorted_moduli = sorted(moduli, key=lambda m: 1/m, reverse=True)
    
    best_classes = []
    for m in sorted_moduli:
        r = can_assign_residue(best_classes, m)
        if r is not None:
            best_classes.append([m, r])
            
    best_score = score(best_classes)
    
    # 2. Local Search
    # We will try to improve the set by swapping elements.
    # State: list of [m, r]
    
    # Pre-sort moduli by weight for easy access to "good" candidates
    # moduli_by_weight: list of (m, 1/m)
    
    used_moduli = set(m for m, r in best_classes)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        improved = False
        
        # Strategy: Try to add a high-weight modulus not in the set,
        # possibly by removing one or more low-weight moduli.
        
        # Get candidates for addition (not in used_moduli)
        candidates_add = [m for m in sorted_moduli if m not in used_moduli]
        
        if not candidates_add:
            break
            
        # Pick a random high-weight candidate to add
        # We prioritize the top few
        target_m = candidates_add[random.randint(0, min(10, len(candidates_add)-1))]
        
        # Try to add target_m to the current set
        # If it fits, great. If not, try to remove some low-weight moduli to make space.
        
        # First, try adding directly
        r_new = can_assign_residue(best_classes, target_m)
        if r_new is not None:
            new_classes = best_classes + [[target_m, r_new]]
            new_score = score(new_classes)
            if new_score > best_score:
                best_classes = new_classes
                best_score = new_score
                used_moduli.add(target_m)
                improved = True
                continue
        
        # If it doesn't fit directly, try swapping.
        # Identify "bad" moduli in current set (low weight, i.e., large m)
        # Sort current classes by weight ascending (largest m first) to find removal candidates
        current_sorted_by_weight = sorted(best_classes, key=lambda c: c[0]) # Ascending m -> Ascending weight? No, m ascending -> weight descending.
        # We want to remove LOW weight (LARGE m). So sort by m descending.
        current_sorted_by_m_desc = sorted(best_classes, key=lambda c: c[0], reverse=True)
        
        # Try removing 1 or 2 low-weight moduli
        removal_candidates = []
        for i in range(min(3, len(current_sorted_by_m_desc))):
            removal_candidates.append(current_sorted_by_m_desc[i])
            
        # Try single removals
        for m_remove in removal_candidates:
            temp_classes = [c for c in best_classes if c[0] != m_remove]
            r_new = can_assign_residue(temp_classes, target_m)
            if r_new is not None:
                new_classes = temp_classes + [[target_m, r_new]]
                new_score = score(new_classes)
                if new_score > best_score:
                    best_classes = new_classes
                    best_score = new_score
                    used_moduli.remove(m_remove)
                    used_moduli.add(target_m)
                    improved = True
                    break
        if improved:
            continue
            
        # Try double removals (if not improved)
        if len(removal_candidates) >= 2:
            for i in range(len(removal_candidates)):
                for j in range(i+1, len(removal_candidates)):
                    m_r1 = removal_candidates[i][0]
                    m_r2 = removal_candidates[j][0]
                    temp_classes = [c for c in best_classes if c[0] != m_r1 and c[0] != m_r2]
                    r_new = can_assign_residue(temp_classes, target_m)
                    if r_new is not None:
                        new_classes = temp_classes + [[target_m, r_new]]
                        new_score = score(new_classes)
                        if new_score > best_score:
                            best_classes = new_classes
                            best_score = new_score
                            used_moduli.remove(m_r1)
                            used_moduli.remove(m_r2)
                            used_moduli.add(target_m)
                            improved = True
                            break
                if improved:
                    break
        if improved:
            continue

        # If no improvement, try a random restart or perturbation?
        # To avoid getting stuck, occasionally shuffle the order of greedy or restart local search from a slightly different state.
        if iteration % 100 == 0:
             # Simple perturbation: remove a random element and re-greedy the rest?
             # Or just continue searching.
             pass

    return best_classes
