import time
import random
from itertools import combinations

def is_sidon_with_diagonals(subset):
    """
    Check if a subset of integers has all unordered pair sums (including diagonals) unique.
    """
    sums = set()
    k = len(subset)
    for i in range(k):
        for j in range(i, k):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(candidate_set):
    """
    Find the size of the largest subset of candidate_set that is Sidon with diagonals.
    candidate_set is a list of 12 distinct integers.
    """
    n = len(candidate_set)
    # Optimization: Check from largest k down
    for k in range(n, 0, -1):
        found = False
        for comb in combinations(candidate_set, k):
            if is_sidon_with_diagonals(comb):
                found = True
                break
        if found:
            return k
    return 0

def construct():
    start_time = time.time()
    deadline = 88.0
    
    # Initial candidates with high collision density
    candidates = [
        list(range(12)),
        [0,1,2,3,6,7,8,9,12,13,14,15],
        [0,1,2,4,5,6,8,9,10,12,13,14],
        [0,2,4,6,8,10,12,14,16,18,20,22],
        [0,1,3,4,6,7,9,10,12,13,15,16],
        [0,3,6,9,12,15,18,21,24,27,30,33],
        [0,1,2,5,6,7,10,11,12,15,16,17],
        [0,2,4,7,9,11,13,15,17,19,21,23],
        # Add more structured candidates
        [0,1,2,3,4,5,6,7,8,9,10,11],
        [0,1,2,3,4,5,6,7,8,9,10,20],
        [0,1,2,3,4,5,6,7,8,18,19,20],
        [0,1,2,3,4,5,16,17,18,19,20,21],
        [0,1,2,3,14,15,16,17,18,19,20,21],
    ]
    
    best_set = None
    best_score = 13
    
    for cand in candidates:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = max_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    if best_set is None:
        best_set = list(range(12))
        best_score = max_sidon_size(best_set)
        
    if best_score <= 3:
        return best_set
        
    current_set = best_set[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Local search: Replace 1 or 2 elements
        new_set = current_set[:]
        num_replaces = random.choice([1, 2])
        indices = random.sample(range(12), num_replaces)
        
        for idx in indices:
            # Try to replace with a value that creates a collision
            # Heuristic: pick a value close to existing ones
            # or random
            if random.random() < 0.5:
                # Near existing
                ref = random.choice(current_set)
                new_val = ref + random.randint(-3, 3)
            else:
                new_val = random.randint(0, 128)
                
            if 0 <= new_val <= 128 and new_val not in new_set:
                new_set[idx] = new_val
            else:
                # If invalid, try random
                new_val = random.randint(0, 128)
                if 0 <= new_val <= 128 and new_val not in new_set:
                    new_set[idx] = new_val
                else:
                    # Fallback
                    new_set[idx] = (new_set[idx] + 1) % 129
                    if new_set[idx] in new_set:
                        new_set[idx] = (new_set[idx] + 1) % 129
                        
        if len(set(new_set)) != 12:
            continue
            
        new_set.sort()
        new_score = max_sidon_size(new_set)
        
        if new_score < best_score:
            best_score = new_score
            best_set = new_set[:]
            current_set = new_set[:]
            current_score = new_score
            if best_score <= 3:
                break
        else:
            # Accept with probability to escape local minima
            if random.random() < 0.2:
                current_set = new_set[:]
                current_score = new_score

    return best_set
