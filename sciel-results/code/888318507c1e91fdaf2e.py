"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy initial construction: Iterate moduli in order of 1/m (descending), adding a modulus if a valid residue exists.
    2. Local Search (Hill Climbing): Try to improve the set of moduli by:
       a. Adding a missing modulus if a valid residue can be found.
       b. Swapping a selected modulus (with lower 1/m) for a missing modulus (with higher 1/m) if the swap results in a valid assignment and higher total score.
       c. Reassigning residues to allow more moduli to be added.
    3. Repeat until no improvement is found or time limit approaches.
    """
    deadline = time.time() + 140.0
    
    def is_valid(classes):
        """Check if all pairs of classes are disjoint."""
        n = len(classes)
        for i in range(n):
            m1, r1 = classes[i]
            for j in range(i+1, n):
                m2, r2 = classes[j]
                g = gcd(m1, m2)
                if (r1 % g) == (r2 % g):
                    return False
        return True
    
    def find_valid_residue(m, classes):
        """Find a residue r in [0, m) such that [m, r] is disjoint from all classes."""
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                g = gcd(m, m2)
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def calc_score(classes):
        """Calculate the sum of 1/m as a float."""
        return sum(1.0 / m for m, r in classes)
    
    def try_add(classes, m):
        """Try to add modulus m with a valid residue. Returns new classes or None."""
        r = find_valid_residue(m, classes)
        if r is not None:
            new_classes = classes + [[m, r]]
            return new_classes
        return None
    
    def try_swap(classes, m_remove, m_add):
        """Try to swap m_remove for m_add. Returns new classes or None."""
        # Remove m_remove
        new_classes = [c for c in classes if c[0] != m_remove]
        # Try to add m_add
        r = find_valid_residue(m_add, new_classes)
        if r is not None:
            new_classes.append([m_add, r])
            return new_classes
        return None
    
    # Initial greedy construction
    # Sort moduli by 1/m descending (i.e., m ascending)
    moduli = list(range(20, 201))
    moduli.sort(key=lambda m: -1.0/m)
    
    classes = []
    for m in moduli:
        if time.time() > deadline - 5.0:
            break
        new_classes = try_add(classes, m)
        if new_classes is not None:
            classes = new_classes
    
    best_classes = classes[:]
    best_score = calc_score(classes)
    
    # Local search
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Try adding missing moduli
        for m in moduli:
            if time.time() > deadline - 5.0:
                break
            if any(c[0] == m for c in classes):
                continue
            new_classes = try_add(classes, m)
            if new_classes is not None:
                new_score = calc_score(new_classes)
                if new_score > best_score + 1e-12:
                    classes = new_classes
                    best_score = new_score
                    best_classes = classes[:]
                    improved = True
        
        # Try swapping: replace a modulus with a missing one
        current_moduli = set(c[0] for c in classes)
        missing_moduli = [m for m in moduli if m not in current_moduli]
        
        # Sort missing by 1/m descending
        missing_moduli.sort(key=lambda m: -1.0/m)
        
        for m_add in missing_moduli:
            if time.time() > deadline - 5.0:
                break
            for m_remove in current_moduli:
                if time.time() > deadline - 5.0:
                    break
                # Only swap if m_add has higher weight
                if 1.0/m_add <= 1.0/m_remove:
                    continue
                new_classes = try_swap(classes, m_remove, m_add)
                if new_classes is not None:
                    new_score = calc_score(new_classes)
                    if new_score > best_score + 1e-12:
                        classes = new_classes
                        best_score = new_score
                        best_classes = classes[:]
                        improved = True
                        break
        
        # Try reassigning residues: for each class, try different residues
        # to see if it allows adding more moduli
        for i, (m, r) in enumerate(classes):
            if time.time() > deadline - 5.0:
                break
            # Try other residues for this class
            for r_new in range(m):
                if r_new == r:
                    continue
                # Check if r_new is valid for class i
                valid = True
                for j, (m2, r2) in enumerate(classes):
                    if i == j:
                        continue
                    g = gcd(m, m2)
                    if (r_new % g) == (r2 % g):
                        valid = False
                        break
                if not valid:
                    continue
                
                # Try reassigning
                old_classes = classes[:]
                classes[i] = [m, r_new]
                
                # Try to add missing moduli
                for m_add in missing_moduli:
                    if time.time() > deadline - 5.0:
                        break
                    if any(c[0] == m_add for c in classes):
                        continue
                    new_classes = try_add(classes, m_add)
                    if new_classes is not None:
                        new_score = calc_score(new_classes)
                        if new_score > best_score + 1e-12:
                            classes = new_classes
                            best_score = new_score
                            best_classes = classes[:]
                            improved = True
                            break
                
                if not improved:
                    classes = old_classes
    
    return best_classes
# EVOLVE-BLOCK-END
