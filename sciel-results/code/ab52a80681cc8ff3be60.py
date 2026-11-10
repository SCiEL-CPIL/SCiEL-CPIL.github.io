import time
import itertools
import random

# EVOLVE-BLOCK-START
def _is_sidon(subset):
    """Check if subset has all pair sums (with replacement) unique."""
    m = len(subset)
    sums = set()
    for i in range(m):
        for j in range(i, m):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_subset_size(arr):
    """Compute the largest subset of arr with all unordered pair sums unique (including diagonals)."""
    n = len(arr)
    if n == 0:
        return 0
    
    # Try subsets in decreasing size order
    for size in range(n, 2, -1):
        for combo in itertools.combinations(range(n), size):
            subset = [arr[i] for i in combo]
            if _is_sidon(subset):
                return size
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Baseline: AP {0, ..., 11}
    baseline = list(range(12))
    best_candidate = baseline
    best_score = _max_sidon_subset_size(baseline)
    
    if best_score == 3:
        return baseline

    candidates = [baseline]
    
    # 1. All APs
    for diff in range(1, 11):
        for start in range(0, 128 - 11*diff + 1):
            cand = [start + i*diff for i in range(12)]
            if max(cand) <= 128:
                candidates.append(cand)
                
    # 2. Unions of two APs (as in original, but optimized)
    for d1 in range(1, 6):
        for d2 in range(1, 6):
            for offset in range(0, 128 - 5*max(d1,d2)):
                if time.time() > deadline:
                    break
                set1 = set(i*d1 for i in range(6))
                set2 = set(offset + i*d2 for i in range(6))
                combined = set1 | set2
                if len(combined) == 12 and max(combined) <= 128:
                    cand = sorted(combined)
                    candidates.append(cand)
                    
    # 3. Random sets (limited, as they are likely to have high Sidon subsets)
    random.seed(42)
    for _ in range(1000):
        if time.time() > deadline:
            break
        cand = random.sample(range(129), 12)
        candidates.append(cand)

    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = _max_sidon_subset_size(cand)
        if score < best_score:
            best_score = score
            best_candidate = cand
            if best_score == 3:
                return best_candidate

    return best_candidate
# EVOLVE-BLOCK-END
