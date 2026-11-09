import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def score(classes):
        return sum(1.0 / m for m, r in classes)
    
    def is_compatible(m, r, chosen):
        for (m2, r2) in chosen:
            g = gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True
    
    def greedy(order):
        """Try to build a set by processing moduli in given order."""
        chosen = []
        for m in order:
            if time.time() > deadline:
                break
            for r in range(m):
                if is_compatible(m, r, chosen):
                    chosen.append((m, r))
                    break
        return chosen
    
    # Strategy 1: moduli in increasing order (smallest first)
    best = greedy(list(range(20, 201)))
    best_score = score(best)
    
    # Strategy 2: moduli in decreasing order (largest first) - might allow more small ones later
    if time.time() < deadline - 10:
        candidate = greedy(list(range(200, 19, -1)))
        if score(candidate) > best_score:
            best = candidate
            best_score = score(best)
    
    # Strategy 3: try a few different residue selection heuristics
    # For now, return the best found
    return [[m, r] for m, r in best]
