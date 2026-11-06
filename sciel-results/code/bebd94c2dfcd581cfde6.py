import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if subset is Sidon (all pair sums a+b with a<=b are distinct)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient):
    """Find the largest Sidon subset size of ambient set."""
    n = len(ambient)
    for k in range(n, 0, -1):
        for subset in combinations(ambient, k):
            if _is_sidon(list(subset)):
                return k
    return 0

def _evaluate(ambient):
    """Return the max Sidon subset size (lower is better)."""
    return _max_sidon_size(ambient)

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    # First, evaluate the seed
    seed_set = list(range(12))
    seed_score = _evaluate(seed_set)
    
    best_set = seed_set[:]
    best_score = seed_score
    
    # Generate diverse initial candidates
    random.seed(12345)
    candidates = []
    
    # Dense clusters in various positions
    for base in range(0, 120, 3):
        for width in range(12, 25):
            if base + width - 1 <= 128:
                cand = sorted(random.sample(range(base, base + width), 12))
                candidates.append(cand)
    
    # Arithmetic progressions
    for a in range(0, 120, 2):
        for d in range(1, 11):
            cand = [a + i * d for i in range(12)]
            if max(cand) <= 128:
                candidates.append(cand)
    
    # Quadratic-like patterns
    for a in range(0, 50):
        for b in range(1, 10):
            cand = [a + i * b + i * i for i in range(12)]
            if max(cand) <= 128 and len(set(cand)) == 12:
                candidates.append(sorted(cand))
    
    # Random sets from small intervals
    for _ in range(300):
        lo = random.randint(0, 100)
        hi = min(128, lo + random.randint(12, 25))
        if hi - lo >= 11:
            cand = sorted(random.sample(range(lo, hi + 1), 12))
            candidates.append(cand)
    
    # Evaluate all candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        score = _evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
    
    # Local search from best
    if time.time() < deadline:
        current = best_set[:]
        current_score = best_score
        
        while time.time() < deadline - 5:
            improved = False
            # Try single-element changes
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                new_set = current[:]
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in new_set:
                    new_set[idx] = new_val
                    new_set.sort()
                    new_score = _evaluate(new_set)
                    if new_score < current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_set = current[:]
                        break
            if improved:
                continue
            # Try swap
            i, j = random.sample(range(12), 2)
            new_set = current[:]
            new_set[i], new_set[j] = new_set[j], new_set[i]
            if len(set(new_set)) == 12:
                new_set.sort()
                new_score = _evaluate(new_set)
                if new_score < current_score:
                    current = new_set
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current[:]
    
    return best_set
