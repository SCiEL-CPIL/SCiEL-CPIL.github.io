"""Search for a 12-element set in [0,128] minimizing the largest weak-Sidon subset."""
import random
import time

def _max_sidon_subset(arr):
    """Return the size of the largest subset of arr with all pair sums (i<=j) unique."""
    n = len(arr)
    # Precompute: for each subset (bitmask), check if it's weak-Sidon
    # We want the max size subset that is weak-Sidon.
    # Since n=12, we can check all subsets from largest to smallest.
    # But more efficiently: check subsets of size k from n down to 1.
    
    # Actually, let's just check all subsets and track max size that is valid.
    # For n=12, 2^12 = 4096 subsets, each check is O(k^2) for k elements.
    # Total is manageable.
    
    # Let's do it by size: check all subsets of size s, from n down to 1.
    # First subset of size s that is weak-Sidon: return s.
    
    from itertools import combinations
    
    # Precompute pair sums for quick checking
    # For a given subset (as a tuple of indices), check if all pair sums are unique
    def is_weak_sidon(indices):
        k = len(indices)
        if k <= 2:
            return True
        sums = []
        for i in range(k):
            for j in range(i, k):
                s = arr[indices[i]] + arr[indices[j]]
                sums.append(s)
        return len(set(sums)) == len(sums)
    
    # Check from largest size down
    for s in range(n, 0, -1):
        for combo in combinations(range(n), s):
            if is_weak_sidon(combo):
                return s
    return 1

def _is_weak_sidon_set(arr):
    """Check if the entire set arr is weak-Sidon (all pair sums i<=j unique)."""
    n = len(arr)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def construct():
    deadline = time.time() + 80.0  # 80s safety margin
    best_set = None
    best_score = 12  # worst case: entire set is Sidon
    
    # Initial candidates
    candidates = []
    
    # Spaced sets
    for start in range(0, 128 - 11 * 10, 2):
        candidates.append(list(range(start, start + 12 * 10, 10)))
    
    # Modular patterns
    for base in range(0, 128, 5):
        cand = []
        for i in range(12):
            v = base + i * 11
            if v <= 128:
                cand.append(v)
        if len(cand) == 12:
            candidates.append(cand)
    
    # Random candidates
    rng = random.Random(42)
    for _ in range(200):
        cand = sorted(rng.sample(range(0, 129), 12))
        candidates.append(cand)
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        s = _max_sidon_subset(cand)
        if s < best_score:
            best_score = s
            best_set = list(cand)
        if best_score <= 4:
            break
    
    # Local search: swap one element
    if best_set is None:
        best_set = list(range(12))
    
    rng2 = random.Random(123)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Random restarts: perturb best_set
        current = list(best_set)
        cur_score = best_score
        
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a random element to replace
            idx = rng2.randint(0, 11)
            old_val = current[idx]
            new_val = rng2.randint(0, 128)
            if new_val == old_val:
                continue
            # Check uniqueness
            if new_val in current:
                continue
            
            candidate = current[:idx] + [new_val] + current[idx+1:]
            candidate.sort()
            
            s = _max_sidon_subset(candidate)
            if s < cur_score:
                cur_score = s
                current = candidate
                improved = True
                if s <= 4:
                    break
            elif s == cur_score:
                # Accept with some probability for diversity
                if rng2.random() < 0.1:
                    current = candidate
        
        if cur_score < best_score:
            best_score = cur_score
            best_set = current
        if best_score <= 4:
            break
    
    return best_set
