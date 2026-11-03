import math
import random
import time

def score_set(U):
    """Calculate the score for a given set U."""
    if not U:
        return 0.0
    U = sorted(list(set(U)))
    if 0 not in U:
        return 0.0
    M = max(U)
    if M == 0:
        return 1.0
    
    # Calculate sumset and diffset sizes
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    num = len(diffset)
    den = len(sumset)
    
    if den == 0:
        return 0.0
        
    ratio = num / den
    if ratio <= 1.0:
        return 1.0
        
    return 1.0 + math.log(ratio) / math.log(2 * M + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 4, 7],
        [0, 1, 3, 8],
        [0, 2, 3, 7]
    ]
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    # Local search
    # We focus on small sets because the denominator grows with max(U)
    # and the numerator is bounded by log(2).
    
    current_set = best_set[:]
    current_score = best_score
    
    # Try to improve by adding/removing/swapping small numbers
    max_val_limit = 50 # Keep max value small to keep denominator small
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try swapping an element with a smaller one
        for i in range(len(current_set)):
            if current_set[i] == 0:
                continue
            old_val = current_set[i]
            # Try replacing with smaller values
            for new_val in range(1, old_val):
                new_set = current_set[:]
                new_set[i] = new_val
                new_set = sorted(list(set(new_set)))
                if len(new_set) != len(current_set):
                    continue
                s = score_set(new_set)
                if s > current_score + 1e-9:
                    current_set = new_set
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 2. Try adding a new element (if size is small)
        if len(current_set) < 6:
            for new_val in range(1, max_val_limit):
                if new_val in current_set:
                    continue
                new_set = current_set + [new_val]
                new_set = sorted(new_set)
                s = score_set(new_set)
                if s > current_score + 1e-9:
                    current_set = new_set
                    current_score = s
                    improved = True
                    break
            
        if improved:
            continue
            
        # 3. Try removing an element (if size > 3)
        if len(current_set) > 3:
            for i in range(len(current_set)):
                if current_set[i] == 0:
                    continue
                new_set = current_set[:i] + current_set[i+1:]
                s = score_set(new_set)
                if s > current_score + 1e-9:
                    current_set = new_set
                    current_score = s
                    improved = True
                    break
        
        if improved:
            continue
            
        # 4. Random perturbation to escape local minima
        if random.random() < 0.1:
            i = random.randint(1, len(current_set)-1)
            delta = random.randint(-5, 5)
            new_val = current_set[i] + delta
            if new_val > 0:
                new_set = current_set[:]
                new_set[i] = new_val
                new_set = sorted(list(set(new_set)))
                if len(new_set) == len(current_set):
                    s = score_set(new_set)
                    if s > current_score - 0.01: # Allow slight decrease to explore
                        current_set = new_set
                        current_score = s
                        improved = True
        
        if not improved and random.random() < 0.01:
            break # Stagnation

    if current_score > best_score:
        best_score = current_score
        best_set = current_set
        
    return best_set
