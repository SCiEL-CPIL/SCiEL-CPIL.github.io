import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 80  # safety margin before 90s
    
    def is_valid_set(s):
        """Check if set s is valid: no element divides sum of any 3 distinct others."""
        n = len(s)
        for i, a in enumerate(s):
            others = [s[j] for j in range(n) if j != i]
            if len(others) < 3:
                continue
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    return False
        return True
    
    # Strategy: Try multiple candidate orderings and keep the best valid set
    # Ordering 1: descending (large numbers first - less likely to divide sums)
    # Ordering 2: ascending
    # Ordering 3: random with seed
    
    best_set = []
    
    # Try descending order first (largest numbers are least likely to be divisors)
    candidates = list(range(192, 0, -1))
    selected = []
    for x in candidates:
        if time.time() - start_time > deadline:
            break
        candidate = selected + [x]
        if is_valid_set(candidate):
            selected.append(x)
            if len(selected) >= 48:
                break
    
    if len(selected) > len(best_set):
        best_set = selected[:]
    
    # Try ascending order
    if time.time() - start_time < deadline - 10:
        candidates = list(range(1, 193))
        selected = []
        for x in candidates:
            if time.time() - start_time > deadline:
                break
            candidate = selected + [x]
            if is_valid_set(candidate):
                selected.append(x)
                if len(selected) >= 48:
                    break
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    # Try a hybrid: start with large numbers, then fill with small ones
    if time.time() - start_time < deadline - 10:
        candidates = list(range(192, 144, -1)) + list(range(1, 145))
        selected = []
        for x in candidates:
            if time.time() - start_time > deadline:
                break
            candidate = selected + [x]
            if is_valid_set(candidate):
                selected.append(x)
                if len(selected) >= 48:
                    break
        
        if len(selected) > len(best_set):
            best_set = selected[:]
    
    return best_set
