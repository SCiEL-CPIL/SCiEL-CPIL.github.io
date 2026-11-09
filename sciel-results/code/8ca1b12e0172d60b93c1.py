"""Rewritten program: search for an ambient 12-set in [0,128] minimizing the largest Sidon subset."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _max_sidon_size(ambient):
    """
    Compute the largest subset of 'ambient' such that all unordered pair sums
    (including diagonals, i.e., a+a) are unique.
    
    A Sidon set (B_2 set) requires that all sums a_i + a_j for i <= j are distinct.
    We find the maximum size k such that there exists a subset of size k with this property.
    """
    n = len(ambient)
    # Try largest k first
    for k in range(n, 0, -1):
        for subset in combinations(ambient, k):
            if _is_sidon(subset):
                return k
    return 0

def _is_sidon(subset):
    """Check if a subset is a Sidon set (all pair sums a_i+a_j for i<=j are unique)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _score(ambient):
    """Lower is better: the size of the largest Sidon subset."""
    return _max_sidon_size(ambient)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    # Initial candidate: the seed program
    best_ambient = list(range(12))
    best_score = _score(best_ambient)
    
    # Strategy: random search with local improvement
    # Key insight: to minimize the largest Sidon subset, we want the ambient set to have
    # many additive collisions. Dense clusters of numbers that are close together
    # create many duplicate pair sums, making it hard to find large Sidon subsets.
    
    random.seed(42)
    
    # Try several structural families
    candidates = []
    
    # Family 1: clustered sets - groups of close numbers
    for cluster_size in [3, 4, 6]:
        num_clusters = 12 // cluster_size
        base = random.randint(0, 128 - num_clusters * 5)
        ambient = []
        for c in range(num_clusters):
            start_val = base + c * 5
            for i in range(cluster_size):
                v = start_val + i
                if 0 <= v <= 128:
                    ambient.append(v)
        if len(ambient) == 12 and len(set(ambient)) == 12:
            candidates.append(ambient)
    
    # Family 2: arithmetic progressions (dense, many collisions)
    for start in range(0, 128 - 12 * 2, 1):
        for step in [1, 2, 3]:
            ambient = [start + i * step for i in range(12)]
            if max(ambient) <= 128:
                candidates.append(ambient)
    
    # Family 3: random sets biased toward small values
    for _ in range(200):
        ambient = sorted(random.sample(range(0, 64), 12))
        candidates.append(ambient)
    
    # Family 4: random sets with structure (pairs of close numbers)
    for _ in range(200):
        ambient = []
        while len(ambient) < 12:
            base = random.randint(0, 126)
            a = base
            b = base + random.randint(1, 3)
            if a not in ambient and b not in ambient and 0 <= a <= 128 and 0 <= b <= 128:
                ambient.append(a)
                ambient.append(b)
        ambient = sorted(ambient)
        if len(set(ambient)) == 12:
            candidates.append(ambient)
    
    # Family 5: very dense - all in a small range
    for start in range(0, 117):
        ambient = list(range(start, start + 12))
        candidates.append(ambient)
    
    # Family 6: random with small spread
    for _ in range(300):
        spread = random.randint(12, 40)
        start = random.randint(0, 128 - spread)
        ambient = sorted(random.sample(range(start, start + spread + 1), 12))
        candidates.append(ambient)
    
    # Evaluate all candidates, keep track of best
    for ambient in candidates:
        if time.time() > deadline:
            break
        s = _score(ambient)
        if s < best_score:
            best_score = s
            best_ambient = ambient[:]
    
    # Local search: try to improve by swapping elements
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            best_ambient.sort()
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, 12):
                    if time.time() > deadline - 5:
                        break
                    # Try swapping best_ambient[i] with a random new value
                    for new_val in range(0, 129):
                        if time.time() > deadline - 5:
                            break
                        if new_val in best_ambient:
                            continue
                        trial = best_ambient[:]
                        trial[j] = new_val
                        trial.sort()
                        s = _score(trial)
                        if s < best_score:
                            best_score = s
                            best_ambient = trial
                            improved = True
                            break
    
    # Final check: ensure valid output
    best_ambient = sorted(set(best_ambient))
    if len(best_ambient) != 12:
        # Fallback to seed
        best_ambient = list(range(12))
    
    return best_ambient

# EVOLVE-BLOCK-END
