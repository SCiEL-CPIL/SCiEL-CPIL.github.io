import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    n = 256
    
    def is_valid_addition(S, y):
        """Check if y can be added to S without creating a 3AP."""
        for a in S:
            # Check if y, a, 2a-y form a 3AP (y is first term)
            c = 2 * a - y
            if c in S and c != y and c != a:
                return False
            # Check if a, y, 2y-a form a 3AP (y is middle term)
            c = 2 * y - a
            if c in S and c != y and c != a:
                return False
        return True
    
    def is_valid_swap(S, x, y):
        """Check if (S - {x}) ∪ {y} is 3AP-free."""
        S_without_x = S - {x}
        # Check if y creates a 3AP in S_without_x
        for a in S_without_x:
            c1 = 2 * a - y
            if c1 in S_without_x and c1 != y and c1 != a:
                return False
            c2 = 2 * y - a
            if c2 in S_without_x and c2 != y and c2 != a:
                return False
        return True
    
    def reciprocal_sum(S):
        return sum(1.0 / x for x in S)
    
    # 1. Greedy initialization: add numbers from 1 to n if valid
    S = set()
    for x in range(1, n + 1):
        if is_valid_addition(S, x):
            S.add(x)
    
    best_score = reciprocal_sum(S)
    best_S = set(S)
    
    def local_search(S, max_iterations=100):
        """Perform local search to improve the set."""
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Pass 1: Try to add any missing element
            for y in range(1, n + 1):
                if y in S:
                    continue
                if is_valid_addition(S, y):
                    S.add(y)
                    improved = True
            
            # Pass 2: Try swaps - replace larger elements with smaller missing ones
            missing = [y for y in range(1, n + 1) if y not in S]
            S_sorted = sorted(S, reverse=True)
            
            for y in missing:
                if time.time() > deadline:
                    break
                # Try to find x in S where x > y and swap is valid
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
    while time.time() < deadline - 3.0 and attempts < 20:
        attempts += 1
        S_current = set(best_S)
        
        # Random perturbation: remove 1-3 random elements
        num_remove = random.randint(1, min(3, len(S_current)))
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
    
    return sorted(best_S)
