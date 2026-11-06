import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    n = 256
    
    # Precompute conflict pairs for each y
    # For each y, list of (a, b) pairs such that {y, a, b} contains a 3AP
    conflict_pairs = [[] for _ in range(n + 1)]
    for y in range(1, n + 1):
        # Type 1: y, y+d, y+2d (y is first term)
        d = 1
        while y + 2 * d <= n:
            conflict_pairs[y].append((y + d, y + 2 * d))
            d += 1
        # Type 2: y-d, y, y+d (y is middle term)
        d = 1
        while y - d >= 1:
            conflict_pairs[y].append((y - d, y + d))
            d += 1
        # Type 3: y-2d, y-d, y (y is last term)
        d = 1
        while y - 2 * d >= 1:
            conflict_pairs[y].append((y - 2 * d, y - d))
            d += 1
    
    def is_valid_addition(S, y):
        """Check if y can be added to S without creating a 3AP."""
        for a, b in conflict_pairs[y]:
            if a in S and b in S:
                return False
        return True
    
    def is_valid_swap(S, x, y):
        """Check if (S - {x}) ∪ {y} is 3AP-free."""
        for a, b in conflict_pairs[y]:
            if a in S and b in S:
                if x == a or x == b:
                    continue
                return False
        return True
    
    def reciprocal_sum(S):
        return sum(Fraction(1, x) for x in S)
    
    # 1. Greedy initialization: add numbers from 1 to n if valid
    S = set()
    for x in range(1, n + 1):
        if is_valid_addition(S, x):
            S.add(x)
    
    best_score = reciprocal_sum(S)
    best_S = set(S)
    
    def local_search(S, max_iterations=50):
        """Perform local search to improve the set."""
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Pass 1: Try to add any missing element (smallest first)
            for y in range(1, n + 1):
                if time.time() > deadline:
                    break
                if y in S:
                    continue
                if is_valid_addition(S, y):
                    S.add(y)
                    improved = True
            
            # Pass 2: Try swaps - replace larger elements with smaller missing ones
            missing = sorted([y for y in range(1, n + 1) if y not in S])
            S_sorted = sorted(S, reverse=True)
            
            for y in missing:
                if time.time() > deadline:
                    break
                for x in S_sorted:
                    if x <= y:
                        break
                    if is_valid_swap(S, x, y):
                        S.remove(x)
                        S.add(y)
                        improved = True
                        S_sorted = sorted(S, reverse=True)
                        break
        
        return S
    
    # Run local search
    S = local_search(S)
    score = reciprocal_sum(S)
    if score > best_score:
        best_score = score
        best_S = set(S)
    
    # 2. Multiple restarts with perturbation
    attempts = 0
    while time.time() < deadline - 3.0 and attempts < 30:
        attempts += 1
        S_current = set(best_S)
        
        # Random perturbation: remove 1-4 random elements
        num_remove = random.randint(1, min(4, len(S_current)))
        for _ in range(num_remove):
            if S_current:
                x_to_remove = random.choice(list(S_current))
                S_current.remove(x_to_remove)
        
        # Run local search on perturbed set
        S_new = local_search(S_current)
        score_new = reciprocal_sum(S_new)
        
        if score_new > best_score:
            best_score = score_new
            best_S = set(S_new)
    
    # 3. Final polish pass
    S_polished = set(best_S)
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        
        # Try additions
        for y in range(1, n + 1):
            if time.time() > deadline - 1.0:
                break
            if y in S_polished:
                continue
            if is_valid_addition(S_polished, y):
                S_polished.add(y)
                improved = True
        
        # Try removals that enable additions
        S_list = sorted(S_polished)
        for x in S_list:
            if time.time() > deadline - 1.0:
                break
            S_test = set(S_polished)
            S_test.discard(x)
            for y in range(1, x):
                if y not in S_test and is_valid_addition(S_test, y):
                    S_polished = S_test
                    improved = True
                    break
            if improved:
                break
    
    score_polished = reciprocal_sum(S_polished)
    if score_polished > best_score:
        best_score = score_polished
        best_S = set(S_polished)
    
    return sorted(best_S)
