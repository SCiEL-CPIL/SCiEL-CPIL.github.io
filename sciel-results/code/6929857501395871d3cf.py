import itertools
import random
import time

def is_sidon_fast(subset):
    """Check if a subset has all unordered pair sums unique (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size_fast(S):
    """Compute the size of the largest Sidon subset of S."""
    n = len(S)
    # Check from largest possible size downwards
    for size in range(n, 0, -1):
        for comb in itertools.combinations(S, size):
            if is_sidon_fast(list(comb)):
                return size
    return 0

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    candidates = []
    
    # 1. Arithmetic Progressions
    # Small d creates more collisions
    for start in range(0, 129 - 11):
        candidates.append(list(range(start, start + 12)))
        
    for d in range(1, 12):
        # Max value <= 128
        max_start = 128 - 11 * d
        if max_start < 0:
            continue
        for start in range(0, max_start + 1):
            candidates.append(list(range(start, start + 12 * d, d)))
            
    # 2. Pair sets: 6 pairs summing to C
    for C in range(10, 129):
        # Try to find 6 distinct pairs (a, C-a) with a < C-a
        # and all 12 numbers distinct
        pairs = []
        used = set()
        for a in range(0, C // 2 + 1):
            b = C - a
            if a == b:
                continue
            if a in used or b in used:
                continue
            if a < 0 or a > 128 or b < 0 or b > 128:
                continue
            pairs.append((a, b))
            used.add(a)
            used.add(b)
            if len(pairs) == 6:
                break
        if len(pairs) == 6:
            cand = []
            for p in pairs:
                cand.extend(p)
            candidates.append(cand)
            
    # 3. Random sets with bias towards clustering
    random.seed(42)
    for _ in range(5000):
        # Cluster around a center
        center = random.randint(10, 118)
        cand = set()
        while len(cand) < 12:
            # Gaussian-like spread
            offset = random.randint(-5, 5)
            val = center + offset
            if 0 <= val <= 128:
                cand.add(val)
        candidates.append(list(cand))
        
    # 4. Pure random
    for seed in range(1000):
        random.seed(seed)
        candidates.append(random.sample(range(129), 12))

    best_set = None
    best_score = float('inf')
    
    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        # Ensure distinct and valid
        if len(set(cand)) != 12:
            continue
        score = max_sidon_size_fast(cand)
        if score < best_score:
            best_score = score
            best_set = list(cand)
            if best_score <= 4:
                # 4 is likely the minimum for 12 elements in [0,128]
                # We can break early if we find 4, but let's see if 3 is possible
                # 3 is very hard. Let's continue for a bit or break if we have plenty of 4s.
                pass

    if best_set is None:
        return list(range(12))

    # Local Refinement
    # Since max_sidon_size_fast is now fast (O(2^12 * 12^2) ~ 60k ops),
    # we can do more local search.
    current_best = list(best_set)
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(12):
            old_val = current_best[i]
            # Try all possible replacements
            for val in range(129):
                if val in current_best:
                    continue
                current_best[i] = val
                new_score = max_sidon_size_fast(current_best)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current_best[i] = old_val
            if time.time() > deadline - 2:
                break
        if time.time() > deadline - 2:
            break
            
    return current_best
