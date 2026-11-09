"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def _is_sidon(subset):
    """Check if a sorted list of integers forms a Sidon set (all pair sums unique including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient):
    """Find the maximum size of a Sidon subset of the given ambient set."""
    n = len(ambient)
    ambient_sorted = sorted(ambient)
    
    # Try all subsets from largest to smallest
    for size in range(n, 0, -1):
        for subset in itertools.combinations(ambient_sorted, size):
            if _is_sidon(list(subset)):
                return size
    return 1

def _evaluate_set(ambient):
    """Return the max Sidon subset size for the given ambient set."""
    return _max_sidon_size(ambient)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave safety margin before 90s limit
    
    # Start with a baseline: arithmetic progression
    best_set = list(range(12))
    best_size = _evaluate_set(best_set)
    
    # Try various structured sets
    candidates = []
    
    # Dense arithmetic progressions
    for start in range(0, 120):
        for step in range(1, 11):
            candidate = [start + i * step for i in range(12)]
            if max(candidate) <= 128 and min(candidate) >= 0:
                candidates.append(candidate)
    
    # Try sets with more collisions - mix of small and large gaps
    # Use greedy/random search with time limit
    import random
    random.seed(42)
    
    # Add some random candidates
    for _ in range(500):
        candidate = sorted(random.sample(range(129), 12))
        candidates.append(candidate)
    
    # Try structured patterns that maximize collisions
    # Pattern: two interleaved APs
    for a in range(0, 60):
        for b in range(1, 10):
            for c in range(1, 10):
                candidate = []
                for i in range(6):
                    candidate.append(a + i * b)
                for i in range(6):
                    candidate.append(a + b//2 + i * c) if (b//2) > 0 else candidate.append(a + i * c)
                # Simplify: just two groups
                candidate = sorted(set(a + i * b for i in range(6)) | set(a + b + i * c for i in range(6)))
                if len(candidate) == 12 and all(0 <= x <= 128 for x in candidate):
                    candidates.append(candidate)
    
    # Evaluate candidates
    for candidate in candidates:
        if time.time() > deadline:
            break
        size = _evaluate_set(candidate)
        if size < best_size:
            best_size = size
            best_set = candidate
    
    # Local search: try to improve by swapping elements
    if time.time() < deadline - 10:
        current = list(best_set)
        current_size = best_size
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                for val in range(129):
                    if val in current:
                        continue
                    trial = current[:]
                    trial[i] = val
                    trial.sort()
                    if len(set(trial)) != 12:
                        continue
                    size = _evaluate_set(trial)
                    if size < current_size:
                        current = trial
                        current_size = size
                        best_set = trial
                        best_size = size
                        improved = True
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
    
    return best_set
# EVOLVE-BLOCK-END
