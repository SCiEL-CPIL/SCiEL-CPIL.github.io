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

def collision_density(candidate_set):
    """
    Count the number of additive quadruple conflicts (a+b = c+d with {a,b} != {c,d}).
    Higher density means more collisions, which should reduce max Sidon subset size.
    """
    sums = {}
    for i in range(len(candidate_set)):
        for j in range(i, len(candidate_set)):
            s = candidate_set[i] + candidate_set[j]
            if s not in sums:
                sums[s] = []
            sums[s].append((i, j))
    
    conflicts = 0
    for s, pairs in sums.items():
        if len(pairs) > 1:
            conflicts += len(pairs) * (len(pairs) - 1) // 2
    return conflicts

def construct():
    start_time = time.time()
    deadline = 85.0
    
    SMALL_RANGE = 30
    
    candidates = [
        list(range(12)),
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
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
    
    current_set = best_set[:]
    current_score = best_score
    
    all_time_best = best_score
    all_time_best_set = best_set[:]
    
    temperature = 1.0
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        new_set = current_set[:]
        
        num_replaces = random.choice([1, 1, 2, 2, 3])
        indices = random.sample(range(12), num_replaces)
        
        for idx in indices:
            if random.random() < 0.7:
                new_val = random.randint(0, SMALL_RANGE)
            else:
                ref = random.choice(current_set)
                new_val = ref + random.randint(-3, 3)
            
            attempts = 0
            while (new_val in new_set or new_val < 0 or new_val > 128) and attempts < 15:
                if random.random() < 0.7:
                    new_val = random.randint(0, SMALL_RANGE)
                else:
                    new_val = random.randint(0, 128)
                attempts += 1
            
            if new_val in new_set or new_val < 0 or new_val > 128:
                available = [v for v in range(0, SMALL_RANGE + 1) if v not in new_set]
                if not available:
                    available = [v for v in range(129) if v not in new_set]
                if available:
                    new_val = random.choice(available)
                else:
                    continue
            
            new_set[idx] = new_val
        
        if len(set(new_set)) != 12:
            continue
        
        new_set.sort()
        new_score = max_sidon_size(new_set)
        
        if new_score < current_score:
            current_set = new_set[:]
            current_score = new_score
            if new_score < all_time_best:
                all_time_best = new_score
                all_time_best_set = new_set[:]
                if all_time_best <= 3:
                    break
        elif new_score == current_score:
            old_density = collision_density(current_set)
            new_density = collision_density(new_set)
            if new_density > old_density:
                current_set = new_set[:]
            elif random.random() < 0.3:
                current_set = new_set[:]
        else:
            delta = new_score - current_score
            prob = 1.0 / (1.0 + delta * delta)
            if random.random() < prob:
                current_set = new_set[:]
        
        temperature *= cooling_rate
        if temperature < 0.05:
            temperature = 0.05
    
    return all_time_best_set
