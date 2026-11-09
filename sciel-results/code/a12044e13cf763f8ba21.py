import time
import random
from itertools import combinations

def _is_sidon_with_diagonals(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _largest_sidon_subset_size(arr):
    n = len(arr)
    for size in range(n, 0, -1):
        for combo in combinations(range(n), size):
            if _is_sidon_with_diagonals([arr[i] for i in combo]):
                return size
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 70
    
    random.seed(777)
    
    best_set = None
    best_score = 13  # Start high
    
    # Try multiple initial solutions and local searches
    initial_solutions = [
        # Symmetric pairs around 64
        [2, 126, 4, 124, 6, 122, 8, 120, 10, 118, 12, 116],
        # Clumped values
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Spread with structure
        [0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98],
    ]
    
    for init in initial_solutions:
        if time.time() > deadline:
            break
        current_set = init[:]
        current_score = _largest_sidon_subset_size(current_set)
        
        if current_score < best_score:
            best_score = current_score
            best_set = current_set[:]
        
        # Local search
        all_values = set(range(129))
        no_improve = 0
        while time.time() < deadline and no_improve < 200:
            i = random.randint(0, 11)
            candidates = [v for v in all_values if v not in current_set]
            if not candidates:
                break
            new_val = random.choice(candidates)
            old_val = current_set[i]
            current_set[i] = new_val
            
            score = _largest_sidon_subset_size(current_set)
            
            if score < current_score:
                current_score = score
                no_improve = 0
                if score < best_score:
                    best_score = score
                    best_set = current_set[:]
            else:
                current_set[i] = old_val
                no_improve += 1
    
    if best_set is None:
        best_set = initial_solutions[0]
    
    return best_set
