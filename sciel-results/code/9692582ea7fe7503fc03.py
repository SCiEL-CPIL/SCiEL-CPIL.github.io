"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s limit
    
    def is_valid(m, r, selected):
        """Check if residue r for modulus m is valid given current selection."""
        for (m_i, r_i) in selected:
            g = gcd(m, m_i)
            if r % g == r_i % g:
                return False
        return True
    
    def find_valid_residue(m, selected):
        """Find a valid residue for modulus m, or None if no valid residue exists."""
        for r in range(m):
            if is_valid(m, r, selected):
                return r
        return None
    
    # Multiple random restarts with greedy + local search
    best_result = []
    best_score = 0.0
    
    import random
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Randomly shuffle the order of moduli
        moduli = list(range(20, 201))
        random.shuffle(moduli)
        
        selected = []
        for m in moduli:
            if time.time() > deadline - 10:
                break
            r = find_valid_residue(m, selected)
            if r is not None:
                selected.append((m, r))
        
        # Local search: try to add any missing moduli
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_mods = set(m for m, r in selected)
            for m in range(20, 201):
                if m in current_mods:
                    continue
                if time.time() > deadline - 10:
                    break
                r = find_valid_residue(m, selected)
                if r is not None:
                    selected.append((m, r))
                    current_mods.add(m)
                    improved = True
                    break
        
        # Calculate score
        score = sum(1.0 / m for m, r in selected)
        if score > best_score:
            best_score = score
            best_result = [[m, r] for m, r in selected]
    
    return best_result
# EVOLVE-BLOCK-END
