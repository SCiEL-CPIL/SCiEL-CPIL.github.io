import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 10.0 # 10 seconds budget
    
    # Initial set: Interval 1..32
    current_set = set(range(1, 33))
    current_score = get_score(current_set)
    best_set = set(current_set)
    best_score = current_score
    
    # Local Search
    while time.time() < deadline:
        # Pick a random element to replace
        elem_to_replace = random.choice(list(current_set))
        
        # Pick a random new value
        new_val = random.randint(1, 512)
        if new_val in current_set:
            continue
            
        # Try replacement
        new_set = set(current_set)
        new_set.remove(elem_to_replace)
        new_set.add(new_val)
        
        new_score = get_score(new_set)
        
        if new_score >= current_score:
            current_set = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        else:
            # Optionally accept worse moves to escape local optima
            # Simple hill climbing is often sufficient for this size
            pass
            
    return sorted(best_set)

def get_score(S):
    S_set = set(S)
    S_sorted = sorted(S)
    d_set = set()
    n = len(S_sorted)
    for i in range(n):
        a = S_sorted[i]
        for j in range(i+1, n):
            b = S_sorted[j]
            d = b - a
            c = b + d
            if c in S_set:
                d_set.add(d)
    return len(d_set)
