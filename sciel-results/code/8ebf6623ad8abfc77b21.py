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
    deadline = 88.0  # Safety margin before 90s kill
    
    # Start with a structured set that creates many sum collisions
    # Use a pattern with arithmetic progressions to force collisions
    # e.g., {0,1,2,3,6,7,8,9,12,13,14,15} - has many a+(a+2)=(a+1)+(a+1) type collisions
    
    # Let's try several initial candidates and pick the best
    candidates = [
        list(range(12)),  # 0..11
        [0,1,2,3,6,7,8,9,12,13,14,15],
        [0,1,2,4,5,6,8,9,10,12,13,14],
        [0,2,4,6,8,10,12,14,16,18,20,22],  # even AP
        [0,1,3,4,6,7,9,10,12,13,15,16],
        [0,3,6,9,12,15,18,21,24,27,30,33],  # step 3
        [0,1,2,5,6,7,10,11,12,15,16,17],
        [0,2,4,7,9,11,13,15,17,19,21,23],
    ]
    
    best_set = None
    best_score = 13  # impossible max
    
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
    
    # Local search with multiple strategies
    current_set = best_set[:]
    current_score = best_score
    
    while time.time() < deadline:
        # Try different perturbation strategies
        strategy = random.random()
        
        if strategy < 0.3:
            # Replace one element with a nearby value
            new_set = current_set[:]
            idx = random.randint(0, 11)
            old_val = new_set[idx]
            # Try values near old_val
            for offset in random.sample(range(-5, 6), min(5, len(range(-5,6)))):
                new_val = old_val + offset
                if 0 <= new_val <= 128 and new_val != old_val:
                    if new_val not in new_set:
                        new_set[idx] = new_val
                        break
            if new_set == current_set:
                continue
            new_set.sort()
        elif strategy < 0.6:
            # Swap two elements with random values
            new_set = current_set[:]
            idx1, idx2 = random.sample(range(12), 2)
            val1 = random.randint(0, 128)
            val2 = random.randint(0, 128)
            
            if val1 == val2:
                continue
            if val1 in new_set and val1 != new_set[idx1]:
                continue
            if val2 in new_set and val2 != new_set[idx2]:
                continue
            if (val1 in new_set and val1 == new_set[idx2]) or (val2 in new_set and val2 == new_set[idx1]):
                continue
                
            new_set[idx1] = val1
            new_set[idx2] = val2
            new_set.sort()
        elif strategy < 0.8:
            # Replace two elements
            new_set = current_set[:]
            idx1, idx2 = random.sample(range(12), 2)
            val1 = random.randint(0, 128)
            val2 = random.randint(0, 128)
            
            if val1 == val2:
                continue
            if val1 in new_set and val1 != new_set[idx1]:
                continue
            if val2 in new_set and val2 != new_set[idx2]:
                continue
            if (val1 in new_set and val1 == new_set[idx2]) or (val2 in new_set and val2 == new_set[idx1]):
                continue
                
            new_set[idx1] = val1
            new_set[idx2] = val2
            new_set.sort()
        else:
            # Random restart: generate a new structured candidate
            base = random.randint(0, 30)
            step = random.randint(1, 5)
            new_set = []
            for i in range(12):
                val = base + i * step + random.randint(-1, 1)
                if 0 <= val <= 128 and val not in new_set:
                    new_set.append(val)
            if len(new_set) == 12:
                new_set.sort()
            else:
                continue
        
        new_score = max_sidon_size(new_set)
        
        if new_score < best_score:
            best_score = new_score
            best_set = new_set[:]
            current_set = new_set[:]
            current_score = new_score
            if best_score <= 3:
                break
        else:
            # Accept with some probability to escape local minima
            if random.random() < 0.15:
                current_set = new_set[:]
                current_score = new_score
    
    return best_set
