"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sidon_subset(subset):
    """Check if a subset has all pair sums (including diagonals) unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size_bruteforce(arr):
    """Find the maximum Sidon subset size by exhaustive search."""
    n = len(arr)
    best = 1
    # Check all subsets in decreasing size order
    for size in range(n, 0, -1):
        found = False
        for subset in combinations(arr, size):
            if _is_sidon_subset(subset):
                return size
        # If we get here, no subset of this size is Sidon
        best = size - 1
    return best

def _evaluate_set(arr):
    """Evaluate a set: return the max Sidon subset size."""
    return _max_sidon_size_bruteforce(arr)

def _generate_candidate(seed):
    """Generate a candidate set designed to have many additive collisions."""
    rng = random.Random(seed)
    
    # Strategy: use a base with controlled offsets to create dense sum collisions
    # Try different structural approaches
    approach = rng.choice([0, 1, 2, 3])
    
    if approach == 0:
        # Arithmetic progression-like with perturbations
        base = rng.randint(0, 8)
        step = rng.randint(2, 10)
        arr = []
        used = set()
        for i in range(12):
            val = base + i * step + rng.randint(-2, 2)
            val = max(0, min(128, val))
            while val in used:
                val = (val + 1) % 129
                if val > 128:
                    val = 0
            used.add(val)
            arr.append(val)
        return sorted(arr)
    
    elif approach == 1:
        # Two interleaved arithmetic progressions
        base1 = rng.randint(0, 10)
        base2 = base1 + rng.randint(3, 15)
        step = rng.randint(2, 8)
        arr = []
        used = set()
        for i in range(6):
            v1 = base1 + i * step
            v2 = base2 + i * step
            for v in [v1, v2]:
                v = max(0, min(128, v))
                while v in used:
                    v = (v + 1) % 129
                if v <= 128:
                    used.add(v)
                    arr.append(v)
        while len(arr) < 12:
            v = rng.randint(0, 128)
            if v not in used:
                used.add(v)
                arr.append(v)
        return sorted(arr[:12])
    
    elif approach == 2:
        # Geometric-like spacing
        base = rng.randint(1, 5)
        ratio = rng.uniform(1.1, 1.8)
        arr = []
        used = set()
        val = base
        for i in range(12):
            v = int(val)
            v = max(0, min(128, v))
            while v in used:
                v = (v + 1) % 129
            if v <= 128:
                used.add(v)
                arr.append(v)
            val *= ratio
        while len(arr) < 12:
            v = rng.randint(0, 128)
            if v not in used:
                used.add(v)
                arr.append(v)
        return sorted(arr[:12])
    
    else:
        # Random with bias toward small values (more collisions in small range)
        arr = []
        used = set()
        for i in range(12):
            # Bias toward smaller numbers
            v = int(rng.betavariate(2, 5) * 129)
            v = min(128, max(0, v))
            while v in used:
                v = (v + 1) % 129
            used.add(v)
            arr.append(v)
        return sorted(arr)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    best_set = list(range(12))
    best_score = _evaluate_set(best_set)
    
    # Try a batch of candidates
    candidates = []
    for seed in range(200):
        candidates.append(_generate_candidate(seed))
    
    # Shuffle for diversity
    random.shuffle(candidates)
    
    # Evaluate candidates, keeping the best
    for arr in candidates:
        if time.time() > deadline:
            break
        if len(arr) != 12 or len(set(arr)) != 12:
            continue
        if any(x < 0 or x > 128 for x in arr):
            continue
        
        score = _evaluate_set(arr)
        if score < best_score:
            best_score = score
            best_set = arr
            if best_score <= 4:  # Very good, stop early
                break
    
    return best_set
# EVOLVE-BLOCK-END
