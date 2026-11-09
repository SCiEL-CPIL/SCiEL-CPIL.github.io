import math
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    max_modulus = 128
    
    # Generate all possible residue classes
    candidates = []
    for m in range(2, max_modulus + 1):
        for r in range(m):
            candidates.append((r, m))
    
    def is_disjoint(c1, c2):
        r1, m1 = c1
        r2, m2 = c2
        # Classes are disjoint if gcd(r1 - r2, m1, m2) > 1
        g = math.gcd(r1 - r2, m1)
        return math.gcd(g, m2) > 1

    best_solution = []
    best_count = 0
    
    # Use a fixed seed for reproducibility in local tests, but random in production?
    # The evaluator might run multiple times. Randomness is good for escaping local optima.
    
    max_restarts = 2000
    for i in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        selected = []
        
        for c in shuffled:
            compatible = True
            for s in selected:
                if not is_disjoint(c, s):
                    compatible = False
                    break
            
            if compatible:
                selected.append(c)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_solution = selected[:]
            
    # Final check: try to add any remaining candidates to the best solution found
    # This is a simple local improvement
    improved = True
    while improved:
        improved = False
        current = best_solution[:]
        for c in candidates:
            if c in current:
                continue
            compatible = True
            for s in current:
                if not is_disjoint(c, s):
                    compatible = False
                    break
            if compatible:
                current.append(c)
                improved = True
                break # Restart the scan from the beginning with the new set
        
        if len(current) > len(best_solution):
            best_solution = current
            best_count = len(current)
            
    return [[r, m] for r, m in best_solution]
