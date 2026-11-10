"""Search for a 12-element ambient set in [0,128] minimizing the largest
Sidon-with-diagonals subset size."""
import time
import random

def _largest_valid(a):
    """Largest subset of a with all pair sums (incl. diag) unique. a is sorted, len 12."""
    k = len(a)
    # Precompute pair sums
    # For each i, j (i<=j), sum = a[i]+a[j]
    # A subset is valid iff no two pairs in it share a sum.
    
    # Use bitset: for each element i, precompute which other elements j
    # would conflict if both are in the subset with i.
    # Actually, let's precompute for each pair (i,j) the sum, and for each sum
    # which pairs have that sum.
    
    # Simpler: for each i, build a bitmask conflict_i where bit j is set if
    # a[i]+a[j] equals a[i]+a[i] or a[j]+a[j] or a[i]+a[i'] for some... 
    # This is getting complex. Let's use a different approach.
    
    # For k=12, check all C(12,t) for t from 6 down to 2.
    # C(12,6)=924, C(12,5)=792, C(12,4)=495. Total ~2200 checks.
    # Each check: O(t^2) to verify all pair sums distinct.
    
    from itertools import combinations
    
    # Precompute all pair sums for the full set
    # sum_table[i][j] = a[i]+a[j] for i<=j
    sum_table = [[0]*k for _ in range(k)]
    for i in range(k):
        for j in range(i, k):
            sum_table[i][j] = a[i] + a[j]
            sum_table[j][i] = a[i] + a[j]
    
    def _is_valid(indices):
        """Check if subset with given indices is valid."""
        seen = set()
        n = len(indices)
        for i in range(n):
            for j in range(i, n):
                s = sum_table[indices[i]][indices[j]]
                if s in seen:
                    return False
                seen.add(s)
        return True
    
    # Check from largest to smallest
    for t in range(6, 1, -1):
        for combo in combinations(range(k), t):
            if _is_valid(combo):
                return t
    return 2

def _score_set(a):
    return _largest_valid(a)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    random.seed(42)
    
    best_set = list(range(12))
    best_score = _score_set(best_set)
    
    def random_set():
        return sorted(random.sample(range(129), 12))
    
    # Try several random starts
    for _ in range(5):
        s = random_set()
        sc = _score_set(s)
        if sc < best_score:
            best_score = sc
            best_set = s[:]
    
    current = best_set[:]
    current_score = best_score
    
    T = 3.0
    T_min = 0.05
    cooling = 0.998
    
    import math
    while time.time() < deadline:
        idx = random.randint(0, 11)
        old_val = current[idx]
        new_val = random.randint(0, 128)
        if new_val == old_val or new_val in current:
            continue
        current[idx] = new_val
        current.sort()
        new_score = _score_set(current)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current[:]
        else:
            current[idx] = old_val
            current.sort()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_set
