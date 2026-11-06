"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy initial construction: Iterate moduli 20..200 in ascending order, adding a modulus if a valid residue exists.
    2. Local Search (Hill Climbing): Try to improve the set of moduli by:
       a. Adding a missing modulus if a valid residue can be found.
       b. Swapping a selected modulus (with lower 1/m) for a missing modulus (with higher 1/m) 
          if the swap results in a valid assignment and higher total score.
    3. Repeat until no improvement is found or time limit approaches.
    """
    deadline = time.time() + 140.0
    
    # State: list of [m, r]
    classes = []
    
    def find_valid_residue(m, current_classes):
        """Find a residue r in [0, m) such that (m, r) is disjoint from all current classes."""
        for r in range(m):
            ok = True
            for cm, cr in current_classes:
                g = gcd(m, cm)
                if r % g == cr % g:
                    ok = False
                    break
            if ok:
                return r
        return None
    
    def calc_score(cls):
        """Calculate sum of 1/m for the given classes."""
        total = Fraction(0, 1)
        for m, r in cls:
            total += Fraction(1, m)
        return total
    
    def is_valid(cls):
        """Check if all pairs are disjoint."""
        n = len(cls)
        for i in range(n):
            for j in range(i+1, n):
                m1, r1 = cls[i]
                m2, r2 = cls[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True
    
    # 1. Greedy Phase: Add moduli in ascending order
    # We want to maximize sum(1/m), so smaller moduli are more valuable.
    # However, adding a small modulus might block many larger ones? 
    # Actually, disjointness depends on residues mod gcd. 
    # Greedy ascending ensures we grab the highest weight items first.
    for m in range(20, 201):
        if time.time() > deadline - 5:
            break
        r = find_valid_residue(m, classes)
        if r is not None:
            classes.append([m, r])
    
    best_score = calc_score(classes)
    best_classes = [c[:] for c in classes]
    
    # 2. Local Search Phase
    # We try to improve the set.
    # Moves:
    # a) Add a modulus m not in the set, if a valid residue exists.
    # b) Swap a modulus m_old in the set with m_new not in the set, if valid and improves score.
    
    current_classes = [c[:] for c in best_classes]
    current_score = best_score
    
    # Precompute moduli list
    all_moduli = list(range(20, 201))
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Try adding
        for m in all_moduli:
            if any(c[0] == m for c in current_classes):
                continue
            r = find_valid_residue(m, current_classes)
            if r is not None:
                new_classes = current_classes + [[m, r]]
                new_score = current_score + Fraction(1, m)
                if new_score > current_score:
                    current_classes = new_classes
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in current_classes]
        
        # Try swapping
        # To make this efficient, we only consider swapping if it clearly improves score.
        # Since 1/m is decreasing, swapping a larger m (smaller weight) for a smaller m (larger weight) is good.
        # But we must ensure the new m is valid.
        
        # Sort current classes by m descending (least valuable first) to try removing them?
        # Or just iterate all pairs.
        
        for i, (m_old, r_old) in enumerate(current_classes):
            # Try replacing m_old with some m_new not in current_classes
            # m_new should ideally be < m_old to increase score, but validity is key.
            # We can just iterate all m_new not in set.
            for m_new in all_moduli:
                if m_new == m_old:
                    continue
                if any(c[0] == m_new for c in current_classes):
                    continue
                
                # Construct candidate set
                candidate = [c[:] for c in current_classes]
                candidate.pop(i)
                
                r_new = find_valid_residue(m_new, candidate)
                if r_new is not None:
                    candidate.append([m_new, r_new])
                    new_score = current_score - Fraction(1, m_old) + Fraction(1, m_new)
                    
                    if new_score > current_score:
                        current_classes = candidate
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_classes = [c[:] for c in current_classes]
                        # If improved, we might want to restart the swap loop or just continue?
                        # Continuing is fine, but the list of "current_classes" changed.
                        # Breaking out to re-evaluate from scratch is safer but slower.
                        # Let's just mark improved and let the while loop re-run.
                        break
            if improved:
                break # Restart the add/swap cycle with the new state
    
    # Return the best found
    # Ensure the result is a list of lists [m, r]
    return [c[:] for c in best_classes]

# EVOLVE-BLOCK-END
