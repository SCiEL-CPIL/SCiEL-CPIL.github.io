import math
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    
    def is_compatible(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def find_valid_residue(m, selected, future_moduli=None):
        """
        Find a residue r for modulus m that is compatible with all selected pairs.
        If future_moduli is provided, use look-ahead to pick the best r.
        Returns the residue or None.
        """
        valid_rs = []
        for r in range(m):
            compatible = True
            for m2, r2 in selected:
                if not is_compatible(m, r, m2, r2):
                    compatible = False
                    break
            if compatible:
                valid_rs.append(r)
        
        if not valid_rs:
            return None
            
        if future_moduli is None:
            return random.choice(valid_rs)
            
        # Look-ahead: Pick the residue that leaves the most options for future moduli
        best_r = None
        max_future_options = -1
        
        # To speed up, we might not check all future moduli if the list is long,
        # but here the list is small enough (181 items).
        # We check how many valid residues each future modulus would have
        # if we chose a specific r for m.
        
        # Optimization: We can just check a few random future moduli or all of them.
        # Let's check all future moduli for accuracy in selection.
        
        # Actually, checking all valid_rs against all future_moduli is O(m * |selected| * |future| * |selected|)
        # which might be too slow if done naively in a tight loop.
        # Let's simplify: Pick the residue that minimizes conflicts with the *next* few moduli.
        
        # Simple heuristic: Pick the residue that has the fewest conflicts with the immediate next moduli.
        # Or just pick a random one if look-ahead is too expensive.
        
        # Let's try a simpler look-ahead:
        # For each candidate r in valid_rs, count how many residues are blocked for the next 5 moduli.
        # Choose the r that blocks the fewest.
        
        next_mods = future_moduli[:5]
        
        best_r = valid_rs[0]
        min_blocked = float('inf')
        
        for r in valid_rs:
            blocked_count = 0
            for m_next in next_mods:
                # Check how many residues for m_next are incompatible with (m, r)
                # and also compatible with current selected? 
                # No, just incompatible with (m, r).
                # If a residue for m_next is incompatible with (m, r), it's blocked by this choice.
                # We want to minimize the number of blocked residues.
                
                # Actually, we want to maximize the number of available residues.
                # So minimize blocked.
                
                g = math.gcd(m, m_next)
                # Residues for m_next incompatible with r are those r_next such that r_next % g == r % g
                # There are m_next / g such residues.
                # So the number of blocked residues is constant for a given pair (m, m_next)!
                # Wait, the number of residues r_next in [0, m_next) such that r_next % g == r % g is exactly m_next / g.
                # This does NOT depend on r.
                # So look-ahead based on "number of blocked residues" is useless because it's constant.
                
                # The look-ahead must be based on whether the *specific* residues needed for other constraints are blocked.
                # This is complex.
                
                # Alternative Look-ahead:
                # Greedy usually works well. Let's stick to a simple heuristic:
                # Pick the residue that is "least likely" to be a multiple of small primes? No.
                
                # Let's just pick a random valid residue. The local search will fix it.
                pass
            
            # Since the simple metric is constant, let's just pick randomly to avoid bias.
            # The local search is the main driver.
            break # Just pick the first one or random? Random is better for diversity.
        
        return random.choice(valid_rs)

    def calc_score(selected):
        return sum(1.0 / m for m, r in selected)
    
    # Strategy: Simulated Annealing / Randomized Local Search
    # Start with a greedy solution.
    # Then perform local search moves:
    # 1. Add a random skipped modulus if possible.
    # 2. Change residue of a random selected modulus to a different valid one.
    # 3. Remove a random selected modulus (if it allows adding a better one later, or just to escape local min).
    # 4. Swap: Remove one, add another.
    
    selected = []
    for m in moduli:
        r = find_valid_residue(m, selected, moduli[moduli.index(m)+1:])
        if r is not None:
            selected.append((m, r))
            
    current_score = calc_score(selected)
    best_selected = list(selected)
    best_score = current_score
    
    used_moduli = set(m for m, r in selected)
    skipped = [m for m in moduli if m not in used_moduli]
    
    # Annealing parameters
    T = 0.1
    T_min = 0.001
    alpha = 0.99
    
    attempts = 0
    max_attempts = 500000
    
    while time.time() - start_time < time_limit and attempts < max_attempts:
        attempts += 1
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.5 and skipped:
            # Try to add a skipped modulus
            m_skip = random.choice(skipped)
            r_new = find_valid_residue(m_skip, selected)
            if r_new is not None:
                # Calculate new score
                new_score = current_score + 1.0 / m_skip
                # Always accept if it improves
                selected.append((m_skip, r_new))
                used_moduli.add(m_skip)
                skipped.remove(m_skip)
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = list(selected)
            else:
                # Try to change a residue to make space
                if selected:
                    idx = random.randint(0, len(selected) - 1)
                    m_old, r_old = selected[idx]
                    # Find a new residue for m_old that is compatible with others
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
                                # Revert
                                selected[idx] = (m_old, r_old)
                                
        elif move_type < 0.8 and selected:
            # Change residue of a random selected modulus
            idx = random.randint(0, len(selected) - 1)
            m_old, r_old = selected[idx]
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
                r_new = random.choice(valid_rs)
                if r_new != r_old:
                    selected[idx] = (m_old, r_new)
                    # Score doesn't change, but state does. 
                    # This helps escape local minima in terms of "addability".
                    # We don't update best_score here as it's the same.
                    
        else:
            # Remove a random modulus (to potentially free up space for a better combination later)
            # This is risky, but helps with diversification.
            if selected:
                idx = random.randint(0, len(selected) - 1)
                m_old, r_old = selected[idx]
                selected.pop(idx)
                used_moduli.remove(m_old)
                skipped.append(m_old)
                current_score -= 1.0 / m_old
                
                # If we removed it, maybe we can add a different one?
                # The next iteration might add it back or a better one.
                # To prevent immediate re-adding of the same thing, we could keep it removed for a few steps,
                # but simple random walk is often sufficient.

        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min

    return [[m, r] for m, r in best_selected]
