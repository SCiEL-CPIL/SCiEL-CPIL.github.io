"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is disjoint from all in selected."""
        for m_old, r_old in selected:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    def find_valid_residue(m, selected, randomize=True):
        """Find a valid residue for modulus m given selected classes."""
        valid = [r for r in range(m) if is_compatible(m, r, selected)]
        if not valid:
            return None
        if randomize and len(valid) > 1:
            return random.choice(valid)
        return valid[0]
    
    def greedy_build(moduli_order, randomize=True):
        """Build a solution by processing moduli in given order."""
        selected = []
        for m in moduli_order:
            r = find_valid_residue(m, selected, randomize)
            if r is not None:
                selected.append([m, r])
        return selected
    
    def score(classes):
        """Compute the score (sum of 1/m)."""
        return sum(1.0 / m for m, r in classes)
    
    # Moduli in [20, 200]
    all_moduli = list(range(20, 201))
    
    # Best solution found
    best_score = 0.0
    best_classes = []
    
    # First, try a deterministic greedy: sort by m ascending
    moduli_sorted = sorted(all_moduli)
    classes = greedy_build(moduli_sorted, randomize=False)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = classes
    
    # Try multiple randomized restarts
    random.seed(42)
    num_restarts = 0
    while time.time() < deadline:
        # Random shuffle of moduli
        shuffled = all_moduli[:]
        random.shuffle(shuffled)
        classes = greedy_build(shuffled, randomize=True)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = classes
        num_restarts += 1
    
    # Also try a heuristic: prioritize primes and small numbers
    # Primes in [20, 200]
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    primes_in_range = [m for m in all_moduli if is_prime(m)]
    composites_in_range = [m for m in all_moduli if not is_prime(m)]
    
    # Try: primes first (sorted by m), then composites
    order1 = sorted(primes_in_range) + sorted(composites_in_range)
    classes = greedy_build(order1, randomize=False)
    s = score(classes)
    if s > best_score:
        best_score = s
        best_classes = classes
    
    # Try: all sorted by m, but with some randomization in residue choice
    for _ in range(50):
        if time.time() >= deadline:
            break
        classes = greedy_build(moduli_sorted, randomize=True)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = classes
    
    # Local search: try to swap or add
    # For each pair in best_classes, try swapping residues
    if time.time() < deadline:
        for i in range(len(best_classes)):
            if time.time() >= deadline:
                break
            m_i, r_i = best_classes[i]
            # Try different residues for this class
            for r_new in range(m_i):
                if r_new == r_i:
                    continue
                # Check if changing this residue is still valid
                temp = best_classes[:]
                temp[i] = [m_i, r_new]
                valid = True
                for j in range(len(temp)):
                    if j == i:
                        continue
                    m_j, r_j = temp[j]
                    g = math.gcd(m_i, m_j)
                    if (r_new - r_j) % g == 0:
                        valid = False
                        break
                if valid:
                    # Try to add a new modulus
                    for m_new in all_moduli:
                        if any(m_new == m for m, r in best_classes):
                            continue
                        r_candidate = find_valid_residue(m_new, temp, randomize=False)
                        if r_candidate is not None:
                            new_classes = temp + [[m_new, r_candidate]]
                            s_new = score(new_classes)
                            if s_new > best_score:
                                best_score = s_new
                                best_classes = new_classes
                            break  # Only add one at a time in this pass
    
    # Verify the best solution is valid
    if best_classes:
        for i in range(len(best_classes)):
            for j in range(i + 1, len(best_classes)):
                m1, r1 = best_classes[i]
                m2, r2 = best_classes[j]
                g = math.gcd(m1, m2)
                assert (r1 - r2) % g != 0, f"Invalid: {m1},{r1} and {m2},{r2}"
    
    return best_classes if best_classes else [[20, 0]]
# EVOLVE-BLOCK-END
