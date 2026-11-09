import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if subset is Sidon: all unordered pair sums (including diagonals) are unique."""
    s = sorted(subset)
    n = len(s)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def max_sidon_size(arr):
    """Find the largest Sidon subset size of arr using backtracking with pruning."""
    arr = sorted(arr)
    n = len(arr)
    best = 0
    
    # Upper bound: if remaining + current <= best, prune
    def backtrack(idx, current, sums_set):
        nonlocal best
        if len(current) + (n - idx) <= best:
            return
        if idx == n:
            if len(current) > best:
                best = len(current)
            return
        
        # Try including arr[idx]
        s_val = arr[idx]
        can_include = True
        new_sums = []
        for prev in current:
            sm = s_val + prev
            if sm in sums_set:
                can_include = False
                break
            new_sums.append(sm)
        if can_include:
            # Also check diagonal: 2*s_val
            if 2 * s_val in sums_set:
                can_include = False
            else:
                new_sums.append(2 * s_val)
        
        if can_include:
            for sm in new_sums:
                sums_set.add(sm)
            current.append(s_val)
            backtrack(idx + 1, current, sums_set)
            current.pop()
            for sm in new_sums:
                sums_set.discard(sm)
        
        # Try excluding arr[idx]
        backtrack(idx + 1, current, sums_set)
    
    backtrack(0, [], set())
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    random.seed(42)
    
    best_set = None
    best_score = 12  # worst case
    
    def evaluate(candidate):
        nonlocal best_set, best_score
        ms = max_sidon_size(candidate)
        if ms < best_score:
            best_score = ms
            best_set = list(candidate)
        return ms
    
    # Try some structured initial candidates
    # Pattern 1: multiples of a large step to create sum collisions
    candidates = []
    
    # Consecutive - baseline
    candidates.append(list(range(12)))
    
    # Arithmetic progression with step 11
    candidates.append(list(range(0, 12*11, 11)))
    
    # Quadratic-like: i^2 mod something
    for base in range(0, 128, 7):
        cand = sorted(set((base + i*i) % 129 for i in range(12)))
        if len(cand) == 12 and all(0 <= x <= 128 for x in cand):
            candidates.append(cand)
    
    # Random initial sets
    for _ in range(50):
        cand = sorted(random.sample(range(129), 12))
        candidates.append(cand)
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        evaluate(cand)
    
    # Local search: swap-based improvement
    if best_set is None:
        best_set = list(range(12))
    
    current = list(best_set)
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        # Try all single swaps
        for i in range(12):
            if time.time() > deadline:
                break
            for val in range(129):
                if val in current:
                    continue
                new_set = list(current)
                new_set[i] = val
                new_set.sort()
                ms = max_sidon_size(new_set)
                if ms < current_score:
                    current = new_set
                    current_score = ms
                    improved = True
                    if time.time() > deadline:
                        break
                elif ms == current_score and random.random() < 0.1:
                    # Sometimes accept equal to escape local optima
                    current = new_set
                    improved = True
        
        if not improved:
            # Try random restarts
            if random.random() < 0.3:
                current = sorted(random.sample(range(129), 12))
                current_score = max_sidon_size(current)
            else:
                break
    
    if current_score < best_score:
        best_set = list(current)
        best_score = current_score
    
    return best_set
