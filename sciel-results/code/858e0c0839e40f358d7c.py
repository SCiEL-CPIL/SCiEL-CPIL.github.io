import time
import random
import itertools

def is_sidon(subset):
    """Check if a subset of integers has all unique pair sums (including diagonals)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_subset_size_k(A, k):
    """Check if A contains a Sidon subset of size k."""
    for combo in itertools.combinations(A, k):
        if is_sidon(combo):
            return True
    return False

def max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    """
    n = len(A)
    for k in range(n, 0, -1):
        if has_sidon_subset_size_k(A, k):
            return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Known baseline: [0,11] has max Sidon subset size 4
    best_candidate = list(range(12))
    best_score = 4  # max Sidon subset size
    
    # Try to find a set with max Sidon subset size 3
    # Strategy: random search with local improvement
    random.seed(42)
    
    # First, try structured constructions
    # Arithmetic progressions and their variants often create many sum collisions
    structured_sets = []
    
    # Dense clusters
    for base in range(0, 128, 2):
        # 12 elements in a small window
        if base + 11 <= 128:
            structured_sets.append(list(range(base, base + 12)))
    
    # Two-cluster structures
    for gap in range(2, 20):
        for cluster1_size in range(6, 13):
            cluster2_size = 12 - cluster1_size
            if cluster2_size < 1:
                continue
            base = 0
            s1 = list(range(base, base + cluster1_size))
            s2 = list(range(base + cluster1_size + gap, base + cluster1_size + gap + cluster2_size))
            if max(s2) <= 128:
                structured_sets.append(s1 + s2)
    
    # Check structured sets
    for s in structured_sets:
        if time.time() > deadline:
            break
        if max_sidon_size(s) < best_score:
            best_score = max_sidon_size(s)
            best_candidate = s[:]
            if best_score <= 3:
                break
    
    if best_score <= 3:
        return best_candidate
    
    # Random search for better candidates
    # Use greedy: start with a set, try to reduce max Sidon size by swapping elements
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Start with a random 12-element set, biased toward dense regions
        # Strategy: pick a center, spread elements with small gaps
        center = random.randint(10, 118)
        offsets = []
        for _ in range(12):
            # Biased toward small offsets
            off = random.randint(-15, 15)
            offsets.append(off)
        
        # Ensure distinctness
        candidates = set()
        for off in offsets:
            val = center + off
            if 0 <= val <= 128:
                candidates.add(val)
        
        # Fill in if we don't have 12
        if len(candidates) < 12:
            used = set(candidates)
            for v in range(0, 129):
                if v not in used:
                    candidates.add(v)
                    used.add(v)
                if len(candidates) == 12:
                    break
        
        candidate = sorted(candidates)
        if len(candidate) != 12:
            continue
        
        score = max_sidon_size(candidate)
        if score < best_score:
            best_score = score
            best_candidate = candidate[:]
            if best_score <= 3:
                break
    
    return best_candidate
