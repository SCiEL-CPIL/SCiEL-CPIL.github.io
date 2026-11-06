import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if subset has all pair sums unique (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(arr):
    """Find the largest subset of arr that is Sidon."""
    n = len(arr)
    for k in range(n, 0, -1):
        for subset in combinations(range(n), k):
            if _is_sidon([arr[i] for i in subset]):
                return k
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin
    
    # Try several initial strategies
    candidates = []
    
    # Strategy 1: consecutive
    candidates.append(list(range(12)))
    
    # Strategy 2: random
    for _ in range(20):
        c = random.sample(range(129), 12)
        candidates.append(c)
    
    # Strategy 3: clustered
    for base in range(0, 120, 10):
        c = [base + i for i in range(6)] + [base + 20 + i for i in range(6)]
        if max(c) <= 128:
            candidates.append(c)
    
    # Evaluate all candidates
    best_set = candidates[0]
    best_score = _max_sidon_size(best_set)
    
    for c in candidates[1:]:
        s = _max_sidon_size(c)
        if s < best_score:
            best_score = s
            best_set = c
    
    # Local search from best
    current = best_set[:]
    current_score = best_score
    temp = 5.0
    cooling = 0.997
    
    while time.time() < deadline:
        neighbor = current[:]
        idx = random.randint(0, 11)
        new_val = random.randint(0, 128)
        
        attempts = 0
        while new_val in neighbor and attempts < 30:
            new_val = random.randint(0, 128)
            attempts += 1
        if new_val in neighbor:
            continue
        
        neighbor[idx] = new_val
        neighbor_score = _max_sidon_size(neighbor)
        
        if neighbor_score < current_score:
            current = neighbor
            current_score = neighbor_score
            if current_score < best_score:
                best_score = current_score
                best_set = neighbor[:]
        elif neighbor_score == current_score:
            if random.random() < 0.3:
                current = neighbor
        else:
            delta = neighbor_score - current_score
            if random.random() < pow(2.718281828, -delta / max(temp, 0.5)):
                current = neighbor
        
        temp *= cooling
    
    return best_set
