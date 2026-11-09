"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Moduli in [20, 200]
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def compute_score(classes):
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total
    
    def is_valid_residue(m, r, classes):
        """Check if residue r for modulus m is disjoint with all existing classes."""
        for (m2, r2) in classes:
            g = gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True
    
    def find_valid_residues(m, classes):
        """Find all valid residues for modulus m given existing classes."""
        valid = []
        for r in range(m):
            if is_valid_residue(m, r, classes):
                valid.append(r)
        return valid
    
    def flexibility_score(m, r, classes, used_moduli):
        """Score how flexible residue r is for modulus m - lower is better (fewer future conflicts)."""
        score = 0
        for m2 in moduli:
            if m2 in used_moduli or m2 == m:
                continue
            g = gcd(m, m2)
            if g == 1:
                continue
            # Count residues of m2 that would conflict with r mod g
            # Residues r2 where r2 ≡ r (mod g): approximately m2/g
            score += m2 // g
        return score
    
    def greedy_construct(order, used_moduli=None):
        """Greedy construction with given order."""
        if used_moduli is None:
            used_moduli = set()
        classes = []
        for m in order:
            if m in used_moduli:
                continue
            valid_residues = find_valid_residues(m, classes)
            if not valid_residues:
                continue
            # Pick residue with minimum flexibility score
            best_r = min(valid_residues, key=lambda r: flexibility_score(m, r, classes, used_moduli))
            classes.append([m, best_r])
            used_moduli.add(m)
        return classes
    
    def local_search(classes, max_time=10.0):
        """Improve via add, swap, and residue reassignment."""
        classes = [list(c) for c in classes]
        current_score = compute_score(classes)
        used_moduli = set(m for m, r in classes)
        start = time.time()
        
        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False
            
            # Try to add each unused modulus
            for m in sorted([m for m in moduli if m not in used_moduli]):
                valid_residues = find_valid_residues(m, classes)
                if valid_residues:
                    # Pick best residue
                    best_r = min(valid_residues, key=lambda r: flexibility_score(m, r, classes, used_moduli))
                    new_classes = classes + [[m, best_r]]
                    new_score = current_score + Fraction(1, m)
                    if new_score > current_score:
                        classes = new_classes
                        current_score = new_score
                        used_moduli.add(m)
                        improved = True
            
            # Try swaps: remove one, add one or more
            for i in range(len(classes)):
                m_remove = classes[i][0]
                r_remove = classes[i][1]
                remaining = [c for j, c in enumerate(classes) if j != i]
                remaining_score = current_score - Fraction(1, m_remove)
                
                # Try adding unused moduli to remaining
                for m_add in sorted([m for m in moduli if m != m_remove and m not in used_moduli]):
                    valid_residues = find_valid_residues(m_add, remaining)
                    if valid_residues:
                        best_r = min(valid_residues, key=lambda r: flexibility_score(m_add, r, remaining, used_moduli - {m_remove}))
                        new_classes = remaining + [[m_add, best_r]]
                        new_score = remaining_score + Fraction(1, m_add)
                        if new_score > current_score:
                            classes = new_classes
                            current_score = new_score
                            used_moduli.discard(m_remove)
                            used_moduli.add(m_add)
                            improved = True
                            break  # Restart swap search from this i
            
            # Try residue reassignment for existing moduli
            for i in range(len(classes)):
                m_i = classes[i][0]
                others = [c for j, c in enumerate(classes) if j != i]
                valid_residues = find_valid_residues(m_i, others)
                if len(valid_residues) > 1:
                    # Try all valid residues, keep the one with best flexibility
                    best_r = min(valid_residues, key=lambda r: flexibility_score(m_i, r, others, used_moduli))
                    if best_r != classes[i][1]:
                        classes[i][1] = best_r
                        improved = True
        
        return classes, current_score
    
    # Get initial solution with sorted order (small moduli first)
    best_classes = greedy_construct(sorted(moduli))
    best_score = compute_score(best_classes)
    
    # Local search on initial
    best_classes, best_score = local_search(best_classes, max_time=20.0)
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(80):
        if time.time() > deadline:
            break
        
        # Different ordering strategies
        if restart % 3 == 0:
            # Small moduli first with random tie-breaking
            order = sorted(moduli, key=lambda m: (m, random.random()))
        elif restart % 3 == 1:
            # Random shuffle biased towards small
            order = moduli.copy()
            random.shuffle(order)
            order.sort(key=lambda m: random.random() * 30 + m / 100)
        else:
            # Pure random
            order = moduli.copy()
            random.shuffle(order)
        
        classes = greedy_construct(order)
        classes, sc = local_search(classes, max_time=8.0)
        if sc > best_score:
            best_score = sc
            best_classes = [list(c) for c in classes]
    
    # Final polish
    if time.time() < deadline - 5:
        best_classes, best_score = local_search(best_classes, max_time=15.0)
    
    return best_classes
# EVOLVE-BLOCK-END
