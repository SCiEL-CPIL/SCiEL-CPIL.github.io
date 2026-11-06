"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Precompute all possible moduli
    moduli = list(range(20, 201))
    n_moduli = len(moduli)
    
    # We'll work with a set of selected moduli and their residues
    # State: list of (m, r) pairs
    
    def is_pairwise_disjoint(classes):
        """Check if all pairs are disjoint: r_i != r_j mod gcd(m_i, m_j)"""
        n = len(classes)
        for i in range(n):
            mi, ri = classes[i]
            for j in range(i + 1, n):
                mj, rj = classes[j]
                g = gcd(mi, mj)
                if ri % g == rj % g:
                    return False
        return True
    
    def find_valid_residue(m, existing_classes):
        """Find a residue r in [0, m) such that (m, r) is disjoint from all existing classes."""
        for r in range(m):
            valid = True
            for (mi, ri) in existing_classes:
                g = gcd(m, mi)
                if r % g == ri % g:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def score_of(classes):
        """Compute sum of 1/m as float."""
        return sum(1.0 / m for m, r in classes)
    
    def score_fraction(classes):
        """Compute exact rational score."""
        total = Fraction(0, 1)
        for m, r in classes:
            total += Fraction(1, m)
        return total
    
    # Strategy: Greedy + Local Search with residue reoptimization
    
    # Step 1: Greedy initial construction
    # Sort moduli by 1/m descending (i.e., m ascending)
    sorted_moduli = sorted(moduli, key=lambda m: m)
    
    best_classes = []
    for m in sorted_moduli:
        r = find_valid_residue(m, best_classes)
        if r is not None:
            best_classes.append((m, r))
    
    best_score = score_of(best_classes)
    
    # Step 2: Local search / Hill climbing
    # Operations:
    # a) Try adding a missing modulus
    # b) Try swapping a selected modulus for a missing one
    # c) Reassign residues to free up capacity
    
    def try_add_missing(classes, sorted_moduli):
        """Try to add any missing modulus. Return new classes if improved."""
        selected_set = set(m for m, r in classes)
        improved = False
        new_classes = list(classes)
        
        for m in sorted_moduli:
            if m in selected_set:
                continue
            r = find_valid_residue(m, new_classes)
            if r is not None:
                new_classes.append((m, r))
                improved = True
        
        return new_classes if improved else None
    
    def try_swap(classes, sorted_moduli):
        """Try swapping a selected modulus (higher m) for a missing one (lower m)."""
        selected_set = set(m for m, r in classes)
        best_new = None
        best_new_score = score_of(classes)
        
        # For each selected modulus (try removing higher ones first)
        for idx, (m_sel, r_sel) in enumerate(classes):
            # Try removing this and adding a better one
            remaining = classes[:idx] + classes[idx+1:]
            for m_new in sorted_moduli:
                if m_new in selected_set:
                    continue
                if m_new >= m_sel:  # only swap for smaller modulus (higher 1/m)
                    break
                r_new = find_valid_residue(m_new, remaining)
                if r_new is not None:
                    candidate = remaining + [(m_new, r_new)]
                    s = score_of(candidate)
                    if s > best_new_score + 1e-12:
                        best_new_score = s
                        best_new = candidate
        
        return best_new
    
    def reoptimize_residues(classes):
        """Try reassigning residues to allow more moduli to be added.
        This is a local search over residue assignments."""
        n = len(classes)
        if n == 0:
            return classes
        
        # Current residue assignment
        residues = [r for m, r in classes]
        moduli_list = [m for m, r in classes]
        
        def check_valid(res):
            for i in range(n):
                for j in range(i + 1, n):
                    g = gcd(moduli_list[i], moduli_list[j])
                    if res[i] % g == res[j] % g:
                        return False
            return True
        
        # Try random reassignments
        best_res = list(residues)
        best_valid = check_valid(best_res)
        
        for _ in range(500):
            new_res = list(best_res)
            # Pick a random index and try a random residue
            idx = random.randint(0, n - 1)
            m = moduli_list[idx]
            new_res[idx] = random.randint(0, m - 1)
            if check_valid(new_res):
                best_res = new_res
                best_valid = True
        
        if best_valid:
            return [(moduli_list[i], best_res[i]) for i in range(n)]
        return classes
    
    # Iterative improvement
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        # Try adding missing
        new_classes = try_add_missing(best_classes, sorted_moduli)
        if new_classes is not None:
            new_score = score_of(new_classes)
            if new_score > best_score + 1e-12:
                best_classes = new_classes
                best_score = new_score
                continue
        
        # Try swaps
        new_classes = try_swap(best_classes, sorted_moduli)
        if new_classes is not None:
            new_score = score_of(new_classes)
            if new_score > best_score + 1e-12:
                best_classes = new_classes
                best_score = new_score
                continue
        
        # Try residue reoptimization then add
        reopt = reoptimize_residues(best_classes)
        if reopt != best_classes:
            new_classes = try_add_missing(reopt, sorted_moduli)
            if new_classes is not None:
                new_score = score_of(new_classes)
                if new_score > best_score + 1e-12:
                    best_classes = new_classes
                    best_score = new_score
                    continue
        
        # If no improvement, try a more aggressive reoptimization
        if iteration % 50 == 0:
            # Try multiple random restarts of residue assignment
            for attempt in range(10):
                if time.time() > deadline:
                    break
                # Randomly perturb some residues
                perturbed = list(best_classes)
                for _ in range(3):
                    idx = random.randint(0, len(perturbed) - 1)
                    m = perturbed[idx][0]
                    perturbed[idx] = (m, random.randint(0, m - 1))
                if is_pairwise_disjoint(perturbed):
                    new_classes = try_add_missing(perturbed, sorted_moduli)
                    if new_classes is not None:
                        new_score = score_of(new_classes)
                        if new_score > best_score + 1e-12:
                            best_classes = new_classes
                            best_score = new_score
                            break
    
    # Final check
    if time.time() > deadline:
        pass  # just return what we have
    
    # Convert to list of lists
    return [[m, r] for m, r in best_classes]

# EVOLVE-BLOCK-END
