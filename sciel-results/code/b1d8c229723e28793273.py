"""Find a 12-element set in [0,128] minimizing the largest Sidon subset (with diagonals)."""
import random
import time
from itertools import combinations

def _is_sidon_with_diag(subset):
    """Check if a sorted list has all pair sums (i<=j) unique."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size_exhaustive(arr):
    """Exact: largest subset of arr with all pair sums (i<=j) unique."""
    n = len(arr)
    best = 0
    # Try all subsets from largest size down
    for k in range(n, 0, -1):
        for combo in combinations(arr, k):
            if _is_sidon_with_diag(list(combo)):
                return k
    return 0

def _find_max_sidon_subset(arr):
    """Find an actual max Sidon subset (not just size)."""
    n = len(arr)
    for k in range(n, 0, -1):
        for combo in combinations(arr, k):
            if _is_sidon_with_diag(list(combo)):
                return set(combo)
    return set()

def _count_pair_collisions(arr):
    """Count number of duplicate pair sums (i<j) in the set."""
    n = len(arr)
    sums = {}
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    collisions = 0
    for count in sums.values():
        if count > 1:
            collisions += count - 1
    return collisions

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    best_set = list(range(12))
    best_size = _max_sidon_size_exhaustive(best_set)
    
    random.seed(42)
    
    def evaluate(s):
        """Return the exact max Sidon subset size for set s."""
        return _max_sidon_size_exhaustive(s)
    
    # Structured initial candidates
    initial_candidates = [
        list(range(12)),                          # 0..11
        [0, 1, 2, 4, 8, 16, 32, 64, 3, 5, 7, 9], # mixed
        [0, 1, 3, 7, 15, 31, 63, 10, 20, 40, 80, 120], # powers + extras
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # even numbers
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121], # squares
        [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66],  # triangular
        list(random.sample(range(129), 12)),
        list(random.sample(range(129), 12)),
    ]
    
    current = list(initial_candidates[0])
    current_size = evaluate(current)
    if current_size < best_size:
        best_size = current_size
        best_set = list(current)
    
    # Local search: swap two elements to reduce max Sidon size
    for restart in range(500):
        if time.time() > deadline:
            break
        
        if restart < len(initial_candidates):
            current = list(initial_candidates[restart])
        elif restart == len(initial_candidates):
            current = list(range(12))
        else:
            # Mix: some random, some structured
            if restart % 3 == 0:
                current = list(random.sample(range(129), 12))
            elif restart % 3 == 1:
                # Perturb a random base
                base = list(random.sample(range(129), 12))
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, 11)
                    base[idx] = random.randint(0, 128)
                    # Ensure distinctness
                    attempts = 0
                    while base[idx] in [base[k] for k in range(12) if k != idx]:
                        base[idx] = random.randint(0, 128)
                        attempts += 1
                        if attempts > 50:
                            base[idx] = random.randint(0, 128)
                current = base
            else:
                # Sparse perturbation of random set
                current = list(random.sample(range(129), 12))
                for _ in range(random.randint(0, 2)):
                    idx = random.randint(0, 11)
                    old = current[idx]
                    new = random.randint(max(0, old - 10), min(128, old + 10))
                    if new not in current:
                        current[idx] = new
        
        current_size = evaluate(current)
        if current_size < best_size:
            best_size = current_size
            best_set = list(current)
            if best_size <= 4:
                return best_set
        
        # Local search: try various moves
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Move type 1: Replace one element with a random value
            for i in range(12):
                if time.time() > deadline:
                    break
                new_val = random.randint(0, 128)
                if new_val in current:
                    continue
                old_val = current[i]
                current[i] = new_val
                new_size = evaluate(current)
                if new_size < current_size:
                    current_size = new_size
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_set = list(current)
                        if best_size <= 4:
                            return best_set
                else:
                    current[i] = old_val
            
            # Move type 2: Replace one element with a nearby value
            for i in range(12):
                if time.time() > deadline:
                    break
                old_val = current[i]
                for offset in [-3, -2, -1, 1, 2, 3]:
                    new_val = old_val + offset
                    if 0 <= new_val <= 128 and new_val not in current:
                        current[i] = new_val
                        new_size = evaluate(current)
                        if new_size < current_size:
                            current_size = new_size
                            improved = True
                            if current_size < best_size:
                                best_size = current_size
                                best_set = list(current)
                                if best_size <= 4:
                                    return best_set
                            break
                        else:
                            current[i] = old_val
            
            # Move type 3: Swap two elements
            for i in range(12):
                if time.time() > deadline:
                    break
                for j in range(i + 1, 12):
                    if time.time() > deadline:
                        break
                    current[i], current[j] = current[j], current[i]
                    new_size = evaluate(current)
                    if new_size < current_size:
                        current_size = new_size
                        improved = True
                        if current_size < best_size:
                            best_size = current_size
                            best_set = list(current)
                            if best_size <= 4:
                                return best_set
                    else:
                        current[i], current[j] = current[j], current[i]
        
        # Conflict-aware move: find max Sidon subset and try to break it
        if time.time() < deadline and current_size > 4:
            sidon_subset = _find_max_sidon_subset(current)
            if sidon_subset:
                # Try replacing elements in the Sidon subset
                for elem in list(sidon_subset):
                    if time.time() > deadline:
                        break
                    idx = current.index(elem)
                    for _ in range(5):
                        new_val = random.randint(0, 128)
                        if new_val in current:
                            continue
                        current[idx] = new_val
                        new_size = evaluate(current)
                        if new_size < current_size:
                            current_size = new_size
                            improved = True
                            if current_size < best_size:
                                best_size = current_size
                                best_set = list(current)
                                if best_size <= 4:
                                    return best_set
                            break
                        else:
                            current[idx] = elem
                    if improved:
                        break
    
    return best_set
