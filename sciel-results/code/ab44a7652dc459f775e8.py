import time
import random

def construct():
    start_time = time.time()
    deadline = 55.0
    n = 256
    
    def is_valid_addition(S, y):
        # Optimized check
        for a in S:
            if (2 * a - y) in S or (2 * y - a) in S:
                return False
        return True

    def is_valid_swap(S, x, y):
        # Check if (S - {x}) U {y} is 3-free
        # S - {x} is 3-free.
        # Check if y creates 3AP in S - {x}
        S_new = S - {x}
        # Check y against S_new
        for a in S_new:
            if (2 * a - y) in S_new or (2 * y - a) in S_new:
                return False
        return True

    # 1. Greedy Init
    S = set()
    for x in range(1, n + 1):
        if is_valid_addition(S, x):
            S.add(x)
            
    best_score = sum(1.0/x for x in S)
    best_S = set(S)

    def local_search(S):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Pass 1: Add
            for y in range(1, n + 1):
                if y in S: continue
                if is_valid_addition(S, y):
                    S.add(y)
                    improved = True
            
            # Pass 2: Swap
            # We want to maximize sum(1/x).
            # Swap x (large) for y (small).
            # Iterate y in ascending order (smallest gain per unit? No, 1/y is largest).
            # For each y, try to swap with largest x > y.
            missing = [y for y in range(1, n + 1) if y not in S]
            # Sort missing ascending
            # Sort S descending
            S_desc = sorted(S, reverse=True)
            
            for y in missing:
                if time.time() > deadline: break
                # Try to find x in S_desc such that x > y and swap is valid
                for x in S_desc:
                    if x <= y: break
                    if is_valid_swap(S, x, y):
                        S.remove(x)
                        S.add(y)
                        improved = True
                        S_desc = sorted(S, reverse=True) # Update list
                        break # Move to next y

        return S

    S = local_search(S)
    score = sum(1.0/x for x in S)
    if score > best_score:
        best_score = score
        best_S = set(S)

    # 2. Perturbation / Restart if time allows
    # Try a few random perturbations
    while time.time() < deadline - 5.0:
        S_current = set(best_S)
        # Remove a random element
        if S_current:
            x_to_remove = random.choice(list(S_current))
            S_current.remove(x_to_remove)
            
        # Run local search on this perturbed set
        S_new = local_search(S_current)
        score_new = sum(1.0/x for x in S_new)
        if score_new > best_score:
            best_score = score_new
            best_S = set(S_new)
            
    return sorted(best_S)
