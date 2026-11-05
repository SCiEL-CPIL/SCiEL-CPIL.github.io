import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_compatible(m1, r1, m2, r2):
        g = gcd(m1, m2)
        return r1 % g != r2 % g
    
    def score(pairs):
        total = 0.0
        for m, r in pairs:
            total += 1.0 / m
        return total
    
    def try_greedy(order, randomize_residues=False):
        """Try to build a valid set following the given order of moduli."""
        selected = []
        for m in order:
            best_r = None
            if randomize_residues:
                candidates = list(range(m))
                random.shuffle(candidates)
            else:
                candidates = list(range(m))
            for r in candidates:
                compatible = True
                for (m2, r2) in selected:
                    if not is_compatible(m, r, m2, r2):
                        compatible = False
                        break
                if compatible:
                    best_r = r
                    break
            if best_r is not None:
                selected.append((m, best_r))
        return selected
    
    def local_search(pairs, max_iter=500):
        """Try to improve by swapping residues or adding/removing moduli."""
        pairs = list(pairs)
        current_score = score(pairs)
        best_pairs = list(pairs)
        best_score = current_score
        moduli_set = set(m for m, r in pairs)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try changing a residue
            for i in range(len(pairs)):
                m, r = pairs[i]
                for new_r in range(m):
                    if new_r == r:
                        continue
                    test_pairs = list(pairs)
                    test_pairs[i] = (m, new_r)
                    valid = True
                    for j in range(len(test_pairs)):
                        if j == i:
                            continue
                        m2, r2 = test_pairs[j]
                        if not is_compatible(m, new_r, m2, r2):
                            valid = False
                            break
                    if valid:
                        new_score = current_score  # same moduli, same score
                        # Actually score doesn't change with residue, only with moduli
                        # So residue changes don't improve score directly
                        # But they might enable adding more moduli later
                        break
            
            # Try adding a new modulus
            for m in range(20, 201):
                if m in moduli_set:
                    continue
                for r in range(m):
                    compatible = True
                    for (m2, r2) in pairs:
                        if not is_compatible(m, r, m2, r2):
                            compatible = False
                            break
                    if compatible:
                        new_pairs = pairs + [(m, r)]
                        new_score = current_score + 1.0 / m
                        if new_score > best_score:
                            best_score = new_score
                            best_pairs = new_pairs
                            moduli_set = set(mm for mm, rr in new_pairs)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
            pairs = best_pairs
            current_score = best_score
        
        return best_pairs
    
    # Strategy 1: Greedy with smallest moduli first
    base_order = list(range(20, 201))
    best_result = try_greedy(base_order)
    best_score = score(best_result)
    
    # Strategy 2: Randomized greedy with multiple restarts
    random.seed(42)
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        # Randomly perturb the order
        order = list(range(20, 201))
        # Partially shuffle: keep small moduli somewhat prioritized
        # Use a random priority for each modulus
        priorities = {m: random.random() / m for m in order}
        order.sort(key=lambda m: priorities[m])
        result = try_greedy(order, randomize_residues=True)
        s = score(result)
        if s > best_score:
            best_score = s
            best_result = result
    
    # Strategy 3: Local search on best result
    if time.time() < deadline - 10:
        improved = local_search(best_result)
        s = score(improved)
        if s > best_score:
            best_score = s
            best_result = improved
    
    # Strategy 4: More aggressive randomized search
    for i in range(500):
        if time.time() > deadline:
            break
        order = list(range(20, 201))
        random.shuffle(order)
        result = try_greedy(order, randomize_residues=True)
        s = score(result)
        if s > best_score:
            best_score = s
            best_result = result
    
    # Convert to required format
    return [[m, r] for m, r in best_result]
