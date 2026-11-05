import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if a subset has all unordered pair sums unique (including diagonals)."""
    n = len(subset)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(S, target_size=None):
    """Find the largest Sidon subset size in S. If target_size given, check if size target_size exists."""
    n = len(S)
    if target_size is not None:
        # Check if there exists a Sidon subset of size target_size
        for combo in combinations(range(n), target_size):
            if _is_sidon([S[i] for i in combo]):
                return True
        return False
    else:
        # Find max
        for k in range(n, 0, -1):
            for combo in combinations(range(n), k):
                if _is_sidon([S[i] for i in combo]):
                    return k
        return 1

def _has_sidon_subset_of_size(S, k):
    """Check if S has a Sidon subset of size k."""
    n = len(S)
    for combo in combinations(range(n), k):
        if _is_sidon([S[i] for i in combo]):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # First, verify seed
    seed_set = list(range(12))
    
    # Try to find a set with max Sidon size 3
    # Strategy: use structured sets with many additive collisions
    
    best_set = seed_set
    best_max_sidon = 4  # seed's value
    
    # Try various structured families
    random.seed(42)
    
    # Family 1: Arithmetic progressions
    for d in range(1, 11):  # 11*d <= 128 => d <= 11
        S = [i * d for i in range(12)]
        if max(S) > 128:
            continue
        # Check max Sidon size
        if not _has_sidon_subset_of_size(S, 4):
            best_set = S
            best_max_sidon = 3
            break
    
    if best_max_sidon == 3:
        return best_set
    
    # Family 2: Two clusters
    for gap in range(20, 60, 5):
        for spread in range(1, 5):
            S = list(range(spread)) + list(range(gap, gap + spread))
            # Need 12 elements, so adjust
            # Let's do: 6 elements in [0, spread1] and 6 in [gap, gap+spread2]
            for s1 in range(1, 8):
                for s2 in range(1, 8):
                    if s1 + s2 == 12:
                        S = list(range(s1)) + list(range(100, 100 + s2))
                        if max(S) > 128:
                            continue
                        if len(set(S)) != 12:
                            continue
                        if not _has_sidon_subset_of_size(S, 4):
                            best_set = S
                            best_max_sidon = 3
                            break
                if best_max_sidon == 3:
                    break
            if best_max_sidon == 3:
                break
        if best_max_sidon == 3:
            break
    
    if best_max_sidon == 3:
        return best_set
    
    # Family 3: Random search with local improvement
    # Generate random sets and try to minimize max Sidon size
    
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Generate random set
        S = sorted(random.sample(range(129), 12))
        
        # Check if max Sidon size is 3
        if not _has_sidon_subset_of_size(S, 4):
            best_set = S
            best_max_sidon = 3
            break
    
    if best_max_sidon == 3:
        return best_set
    
    # If we couldn't find size 3, try to at least confirm the seed or find something with size 4
    # but structured differently
    # Try: set with many duplicate sums
    # {0,1,2,3, 100,101,102,103, 200,...} won't work since max is 128
    
    # Try: two groups close together
    for g1 in range(0, 5):
        for g2 in range(10, 120):
            S = [g1 + i for i in range(6)] + [g2 + i for i in range(6)]
            if max(S) > 128 or len(set(S)) != 12:
                continue
            if not _has_sidon_subset_of_size(S, 4):
                best_set = S
                best_max_sidon = 3
                break
        if best_max_sidon == 3:
            break
    
    if best_max_sidon == 3:
        return best_set
    
    # Fallback: return the seed
    return seed_set
