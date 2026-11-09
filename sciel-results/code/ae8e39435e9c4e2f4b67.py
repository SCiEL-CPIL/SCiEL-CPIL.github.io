import time
import math
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = 0.0
    best_classes = []
    
    # Helper to check if residue r for modulus m is compatible with current list
    def is_compatible(current, m, r):
        for m2, r2 in current:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    # Helper to find a valid residue for m given current
    def find_residue(current, m):
        # Try residues in a randomized order or standard order
        # Standard order is deterministic, random order helps in restarts
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_compatible(current, m, r):
                return r
        return None

    # Strategy 1: Greedy Ascending
    # This is a strong baseline
    current = []
    score = 0.0
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = find_residue(current, m)
        if r is not None:
            current.append([m, r])
            score += 1.0 / m
    if score > best_score:
        best_score = score
        best_classes = current

    # Strategy 2: Randomized Greedy with multiple restarts
    # We try to find a better set of moduli by randomizing the order of consideration
    num_restarts = 2000
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random permutation of moduli
        moduli = list(range(20, 201))
        random.shuffle(moduli)
        
        current = []
        score = 0.0
        
        for m in moduli:
            # Optimization: if we already have a very high score, we might skip
            # But let's just run it
            r = find_residue(current, m)
            if r is not None:
                current.append([m, r])
                score += 1.0 / m
        
        if score > best_score:
            best_score = score
            best_classes = current

    # Strategy 3: Local Search on the best result
    # Try to add unselected moduli or swap
    if best_classes:
        selected_moduli = set(m for m, r in best_classes)
        unselected_moduli = [m for m in range(20, 201) if m not in selected_moduli]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add any unselected modulus
            for m in unselected_moduli:
                if time.time() > deadline:
                    break
                r = find_residue(best_classes, m)
                if r is not None:
                    best_classes.append([m, r])
                    selected_moduli.add(m)
                    unselected_moduli.remove(m)
                    best_score += 1.0 / m
                    improved = True
            
            # Try to swap: remove a large modulus, try to add a smaller one
            # Sort best_classes by modulus descending to try removing large ones first
            # This is a heuristic: smaller moduli have higher weight
            if not improved:
                # Create a list of (m, r) sorted by m descending
                sorted_classes = sorted(best_classes, key=lambda x: x[0], reverse=True)
                for m_remove, r_remove in sorted_classes:
                    if time.time() > deadline:
                        break
                    # Try to remove m_remove
                    trial_classes = [c for c in best_classes if c[0] != m_remove]
                    # Try to add any unselected modulus that is smaller than m_remove
                    for m_add in unselected_moduli:
                        if m_add >= m_remove:
                            continue
                        r_add = find_residue(trial_classes, m_add)
                        if r_add is not None:
                            # Check if score improves
                            new_score = best_score - 1.0/m_remove + 1.0/m_add
                            if new_score > best_score:
                                best_classes = trial_classes + [[m_add, r_add]]
                                selected_moduli.remove(m_remove)
                                selected_moduli.add(m_add)
                                unselected_moduli.remove(m_add)
                                unselected_moduli.append(m_remove)
                                best_score = new_score
                                improved = True
                                break # Restart the loop with new state
                    if improved:
                        break

    return best_classes
