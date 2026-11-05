import time
import random
from itertools import combinations

def _is_sidon_with_diagonals(subset):
    """Check if subset has all pair sums (including diagonals) unique."""
    s = sorted(subset)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def _max_sidon_size(arr):
    """Find the largest Sidon subset size in arr."""
    n = len(arr)
    # Try sizes from large to small
    for size in range(n, 0, -1):
        for subset in combinations(range(n), size):
            if _is_sidon_with_diagonals([arr[i] for i in subset]):
                return size
    return 0

def _evaluate(arr):
    """Return the max Sidon subset size for the given 12-element set."""
    return _max_sidon_size(arr)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    best_set = list(range(12))
    best_score = _evaluate(best_set)
    
    if best_score <= 4:
        return best_set
    
    random.seed(42)
    
    # Local search: try to find a set with max Sidon size = 4
    current = [random.randint(0, 128) for _ in range(12)]
    # Ensure distinct
    while len(set(current)) < 12:
        current = [random.randint(0, 128) for _ in range(12)]
    
    current_score = _evaluate(current)
    if current_score < best_score:
        best_score = current_score
        best_set = current[:]
    
    # Simulated annealing / hill climbing
    temp = 10.0
    cooling = 0.995
    
    while time.time() < deadline:
        # Generate neighbor: swap or modify one element
        move_type = random.random()
        if move_type < 0.5:
            # Swap two elements
            i, j = random.sample(range(12), 2)
            neighbor = current[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        elif move_type < 0.8:
            # Change one element
            i = random.randint(0, 11)
            neighbor = current[:]
            neighbor[i] = random.randint(0, 128)
        else:
            # Change two elements
            i, j = random.sample(range(12), 2)
            neighbor = current[:]
            neighbor[i] = random.randint(0, 128)
            neighbor[j] = random.randint(0, 128)
        
        if len(set(neighbor)) != 12:
            continue
        if any(x < 0 or x > 128 for x in neighbor):
            continue
        
        neighbor_score = _evaluate(neighbor)
        
        if neighbor_score <= current_score:
            current = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
                if best_score <= 4:
                    break
        else:
            # Accept with probability based on temperature
            if random.random() < pow(2.718281828, -(neighbor_score - current_score) / max(temp, 0.01)):
                current = neighbor
                current_score = neighbor_score
        
        temp *= cooling
    
    return best_set
