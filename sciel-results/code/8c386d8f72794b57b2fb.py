import random
import time
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(arr):
    """Find the largest Sidon subset size in arr (12 elements)."""
    n = len(arr)
    # Check from largest possible down
    for k in range(n, 0, -1):
        found = False
        for comb in combinations(range(n), k):
            subset = [arr[i] for i in comb]
            if is_sidon(subset):
                found = True
                break
        if found:
            return k
    return 0

def count_sidon_collisions(arr):
    """Count the number of additive collision quadruples in arr.
    A collision is a+b=c+d where {a,b} != {c,d} as unordered pairs.
    More collisions means smaller max Sidon subset."""
    n = len(arr)
    collision_count = 0
    # Count all pair sums and find duplicates
    sum_count = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            sum_count[s] = sum_count.get(s, 0) + 1
    for s, cnt in sum_count.items():
        if cnt > 1:
            # Each pair of pairs with same sum is a collision
            collision_count += cnt * (cnt - 1) // 2
    return collision_count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    best_set = list(range(12))
    best_val = max_sidon_size(best_set)
    
    random.seed(42)
    
    # Try structured initializations that are likely to have small max Sidon
    structured_inits = [
        # Dense block 0-11 (likely max Sidon size 4-5)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Dense block 1-12
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        # Two dense clusters
        [0, 1, 2, 3, 4, 5, 64, 65, 66, 67, 68, 69],
        # Three clusters
        [0, 1, 2, 3, 43, 44, 45, 46, 86, 87, 88, 89],
        # Dense cluster + sparse
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 128],
        # AP with small difference
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        # Mix of dense and spread
        [0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 128],
        # Very dense: 0-9 plus two more close
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Two overlapping dense regions
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Dense with one outlier
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        # Dense with two outliers
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15],
        # Quadratic-like: i*i mod something
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121],
        # Fibonacci-like
        [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 128],
        # Powers of 2
        [0, 1, 2, 4, 8, 16, 32, 64, 0, 0, 0, 0],  # invalid, will be replaced
    ]
    
    # Filter out invalid structured inits
    valid_structured = []
    for init_set in structured_inits:
        if len(set(init_set)) == 12 and all(0 <= x <= 128 for x in init_set):
            valid_structured.append(init_set)
    
    for init_set in valid_structured:
        if time.time() > deadline:
            break
        val = max_sidon_size(init_set)
        if val < best_val:
            best_val = val
            best_set = init_set[:]
            if best_val <= 4:
                break
    
    # Random search with local improvements
    for restart in range(300):
        if time.time() > deadline:
            break
        
        # Random initialization - bias toward dense clusters
        if random.random() < 0.3:
            # Create a dense cluster
            base = random.randint(0, 117)
            current_set = sorted(random.sample(range(base, min(base+12, 129)), min(12, 129-base)))
            if len(current_set) < 12:
                current_set = random.sample(range(129), 12)
        else:
            current_set = random.sample(range(129), 12)
        
        current_val = max_sidon_size(current_set)
        
        if current_val < best_val:
            best_val = current_val
            best_set = current_set[:]
            if best_val <= 4:
                break
        
        # Local search: try single element replacements
        for step in range(300):
            if time.time() > deadline:
                break
            
            i = random.randint(0, 11)
            old_val = current_set[i]
            
            # Try values near existing elements to create collisions
            if random.random() < 0.5:
                # Pick a value near another element
                j = random.randint(0, 11)
                if j != i:
                    ref = current_set[j]
                    offset = random.choice([-3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
                    new_val = ref + offset
                    if not (0 <= new_val <= 128) or new_val in current_set:
                        new_val = random.randint(0, 128)
            else:
                new_val = random.randint(0, 128)
            
            if new_val == old_val or new_val in current_set:
                continue
            
            current_set[i] = new_val
            new_val_result = max_sidon_size(current_set)
            
            if new_val_result < current_val:
                current_val = new_val_result
                if current_val < best_val:
                    best_val = current_val
                    best_set = current_set[:]
                    if best_val <= 4:
                        break
            elif new_val_result == current_val and random.random() < 0.1:
                # Occasionally accept equal to escape local optima
                pass
            else:
                current_set[i] = old_val
            
            if best_val <= 4:
                break
    
    return best_set
