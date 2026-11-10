import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    
    # We want to maximize sum(1/m).
    # Since 1/m is decreasing, we prioritize smaller m.
    # We will try to construct a valid set of (m, r) pairs.
    
    def is_compatible(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def find_valid_residue(m, selected):
        """
        Find a residue r for modulus m that is compatible with all selected pairs.
        Returns the residue or None.
        """
        for r in range(m):
            compatible = True
            for m2, r2 in selected:
                if not is_compatible(m, r, m2, r2):
                    compatible = False
                    break
            if compatible:
                return r
        return None

    def calc_score(selected):
        return sum(1.0 / m for m, r in selected)

    # Strategy 1: Greedy with Backtracking
    # We process moduli in increasing order (highest weight first).
    # We maintain a list of selected (m, r).
    
    best_selected = []
    best_score = 0.0
    
    # Try to include all moduli
    # Use a simple backtracking search
    # State: index in moduli list, current selection
    
    def backtrack(idx, selected):
        nonlocal best_selected, best_score
        
        if time.time() - start_time > time_limit:
            return
        
        if idx == len(moduli):
            score = calc_score(selected)
            if score > best_score:
                best_score = score
                best_selected = list(selected)
            return
        
        m = moduli[idx]
        
        # Try to include m
        # Find a valid residue
        # To speed up, we can try residues in a specific order
        # Smaller residues are often "safer" or just arbitrary.
        
        found_residue = False
        # Try all residues
        for r in range(m):
            # Check compatibility with current selection
            compatible = True
            for m2, r2 in selected:
                if not is_compatible(m, r, m2, r2):
                    compatible = False
                    break
            
            if compatible:
                # Add (m, r) and recurse
                selected.append((m, r))
                backtrack(idx + 1, selected)
                selected.pop()
                found_residue = True
                # Optimization: If we found one, do we need to try others?
                # For maximum weight, including m is always better than skipping it.
                # However, the choice of r might affect future moduli.
                # So we MUST try all valid residues to ensure we don't block future high-value moduli.
                
        # Try to skip m
        # Only if including m didn't lead to a better solution or if we ran out of time?
        # Actually, skipping m is always an option.
        # But since 1/m > 0, including m is strictly better IF it doesn't prevent including future moduli.
        # If including m blocks a future m' with 1/m' > 0, it might be bad?
        # No, we just lose 1/m'. We gain 1/m.
        # Since 1/m > 1/m' for m < m', including m is generally good.
        # But if including m blocks MANY future moduli, it might be bad.
        
        # So we must also branch on skipping m.
        # However, to save time, we can prioritize including m.
        
        # If we didn't find any residue, we must skip.
        # If we did find residues, we explored them. Now try skipping.
        # Optimization: If the current best score + sum(1/m for m in moduli[idx:]) <= best_score, prune.
        
        # Simple prune:
        # Upper bound: current score + sum of all remaining 1/m
        # This is expensive to compute every time.
        
        # For now, just recurse on skip.
        backtrack(idx + 1, selected)

    # The backtracking above might be too slow (2^181 states).
    # We need a better heuristic.
    
    # Heuristic: Greedy with Local Search
    # 1. Greedy: Add m if possible.
    # 2. Local Search: Try to improve.
    
    selected = []
    for m in moduli:
        r = find_valid_residue(m, selected)
        if r is not None:
            selected.append((m, r))
    
    current_score = calc_score(selected)
    best_selected = list(selected)
    best_score = current_score
    
    # Local Search
    # Try to add skipped moduli by changing residues of existing ones.
    
    used_moduli = set(m for m, r in selected)
    skipped = [m for m in moduli if m not in used_moduli]
    
    attempts = 0
    max_attempts = 100000
    
    while time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        
        # Randomly pick a skipped modulus
        if skipped:
            m_skip = random.choice(skipped)
            # Try to find a residue for m_skip
            # If none, try to change a residue of an existing modulus to make space
            
            r_new = find_valid_residue(m_skip, selected)
            if r_new is not None:
                selected.append((m_skip, r_new))
                used_moduli.add(m_skip)
                skipped.remove(m_skip)
                current_score += 1.0 / m_skip
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(selected)
            else:
                # Try to swap residue of a random existing modulus
                if selected:
                    idx = random.randint(0, len(selected) - 1)
                    m_old, r_old = selected[idx]
                    
                    # Try changing r_old to r_old_new
                    # We want r_old_new to be compatible with others AND allow m_skip to be added
                    # This is complex. Simpler: just change r_old to a random valid one.
                    
                    valid_rs = []
                    for r in range(m_old):
                        compatible = True
                        for i, (m2, r2) in enumerate(selected):
                            if i == idx:
                                continue
                            if not is_compatible(m_old, r, m2, r2):
                                compatible = False
                                break
                        if compatible:
                            valid_rs.append(r)
                    
                    if valid_rs:
                        r_new_old = random.choice(valid_rs)
                        if r_new_old != r_old:
                            selected[idx] = (m_old, r_new_old)
                            # Now check if m_skip can be added
                            r_skip = find_valid_residue(m_skip, selected)
                            if r_skip is not None:
                                selected.append((m_skip, r_skip))
                                used_moduli.add(m_skip)
                                skipped.remove(m_skip)
                                current_score += 1.0 / m_skip
                                if current_score > best_score:
                                    best_score = current_score
                                    best_selected = list(selected)
                            else:
                                # Revert if it didn't help? Or keep if it's valid?
                                # Keeping a valid state is fine.
                                pass

    return [[m, r] for m, r in best_selected]
