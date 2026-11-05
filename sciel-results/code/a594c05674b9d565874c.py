"""Optimized odd covering system search for moduli dividing 51975."""
import time
import random
from collections import defaultdict

def construct():
    BASE = 51975
    # Deadline guard: return best result well before 150s
    deadline = time.time() + 130.0
    
    # Get all odd divisors > 1 of 51975
    divisors = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            divisors.append(d)
    # Sort by size (smaller moduli cover more residues on average)
    divisors.sort()
    
    # Precompute residue sets for each divisor
    # For each divisor m, precompute which residues [0, BASE) are covered by class (m, r)
    # covered[r::m] for r in range(m)
    
    # Strategy: Randomized greedy with adaptive look-ahead
    # 1. Start with a good deterministic greedy solution
    # 2. Then do local search / randomized restarts to improve
    
    def compute_uncovered(covered):
        """Count uncovered residues."""
        return sum(1 for x in covered if x == 0)
    
    def greedy_solve(moduli_order, covered):
        """Given an ordering of moduli, greedily assign residues."""
        classes = []
        cov = bytearray(covered)
        for m in moduli_order:
            best_r = 0
            best_gain = -1
            for r in range(m):
                # Count how many uncovered residues this class would cover
                gain = 0
                for i in range(r, BASE, m):
                    if cov[i] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_gain > 0:
                classes.append([m, best_r])
                for i in range(best_r, BASE, m):
                    cov[i] = 1
        return classes, cov
    
    def score_of(covered):
        """Return number of uncovered residues (lower is better)."""
        return BASE - sum(1 for x in covered if x == 1)
    
    # Initial greedy solution
    best_classes = []
    best_score = BASE  # worst case: nothing covered
    
    # First pass: standard greedy with sorted moduli
    covered_init = bytearray(BASE)
    classes_init, covered_init = greedy_solve(divisors, covered_init)
    best_score = score_of(covered_init)
    best_classes = classes_init[:]
    
    # Try different orderings to find better solutions
    # Use randomized greedy with multiple restarts
    rng = random.Random(42)
    
    num_restarts = 0
    while time.time() < deadline:
        # Adaptive look-ahead: probabilistically select next modulus based on marginal gain
        # Try a randomized ordering with some bias toward smaller moduli
        moduli_order = divisors[:]
        
        # Partially shuffle: keep small moduli somewhat prioritized
        # Use a temperature-based approach
        temp = 1.0 / (1.0 + num_restarts * 0.1)
        
        for i in range(len(moduli_order) - 1):
            if rng.random() < temp:
                j = rng.randint(i, len(moduli_order) - 1)
                moduli_order[i], moduli_order[j] = moduli_order[j], moduli_order[i]
        
        classes_try, covered_try = greedy_solve(moduli_order, bytearray(BASE))
        score_try = score_of(covered_try)
        
        if score_try < best_score:
            best_score = score_try
            best_classes = classes_try[:]
        
        num_restarts += 1
        
        # Also try a local improvement: swap residues for some classes
        if score_try < best_score + 50 and time.time() < deadline - 10:
            # Local search: try changing residue for each class
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                cov_local = bytearray(covered_try)
                for idx, (m, r) in enumerate(classes_try):
                    # Try all residues for this modulus
                    best_r = r
                    best_local_score = score_of(cov_local)
                    # Remove current class
                    for i in range(r, BASE, m):
                        cov_local[i] = 0
                    base_score = score_of(cov_local)
                    for r2 in range(m):
                        if r2 == r:
                            continue
                        for i in range(r2, BASE, m):
                            cov_local[i] = 1
                        s = score_of(cov_local)
                        if s < best_local_score:
                            best_local_score = s
                            best_r = r2
                        # Restore
                        for i in range(r2, BASE, m):
                            cov_local[i] = 0
                    # Restore current
                    for i in range(r, BASE, m):
                        cov_local[i] = 1
                    if best_r != r:
                        classes_try[idx][1] = best_r
                        for i in range(best_r, BASE, m):
                            cov_local[i] = 1
                        improved = True
                        covered_try = cov_local
                        if score_of(covered_try) < best_score:
                            best_score = score_of(covered_try)
                            best_classes = [c[:] for c in classes_try]
    
    return best_classes
