"""Improved program: greedy Sidon set construction to maximize |U-U|/|U+U| ratio."""
import time
import math

# EVOLVE-BLOCK-START

def _score(U):
    """Compute the lower bound score for set U."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute |U+U| and |U-U|
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    n_sum = len(sumset)
    n_diff = len(diffset)
    if n_sum == 0:
        return 0.0
    ratio = n_diff / n_sum
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _greedy_sidon_set(n, max_val):
    """Greedily construct a Sidon set of size n with elements in [0, max_val].
    
    A Sidon set has all pairwise sums distinct, which implies all positive
    differences are distinct, giving |U-U| = 2*sum_{i<j}(U[j]-U[i]) + 1
    and |U+U| = n(n+1)/2 (all sums distinct).
    
    We greedily pick each next element as the smallest value that maintains
    the Sidon property (all pairwise sums distinct).
    """
    if n < 1:
        return [0]
    U = [0]
    used_sums = {0}  # all pairwise sums (a+b) for a,b in U
    
    for _ in range(1, n):
        # Try values in increasing order
        found = False
        for candidate in range(1, max_val + 1):
            # Check if adding candidate maintains Sidon property
            # New sums: candidate + u for each u in U
            ok = True
            new_sums = set()
            for u in U:
                s = candidate + u
                if s in used_sums:
                    ok = False
                    break
                new_sums.add(s)
            if ok:
                U.append(candidate)
                used_sums.update(new_sums)
                found = True
                break
        if not found:
            break
    
    return U


def _greedy_diff_max_set(n, max_val):
    """Greedily construct a set maximizing |U-U| while keeping |U+U| small.
    
    Strategy: pick elements with large gaps so that all differences are distinct
    (giving |U-U| = 2*C(n,2)+1 = n^2-n+1) and all sums are also distinct
    (giving |U+U| = n(n+1)/2).
    
    This is achieved by a Sidon/B2 sequence construction.
    """
    if n < 1:
        return [0]
    
    U = [0]
    # Track all sums a+b for a,b in U to ensure uniqueness
    used_sums = set()
    for a in U:
        for b in U:
            used_sums.add(a + b)
    
    for _ in range(1, n):
        found = False
        # Try candidates starting from small values for compactness
        for candidate in range(1, max_val + 1):
            ok = True
            new_sums = []
            for u in U:
                s = candidate + u
                if s in used_sums:
                    ok = False
                    break
                new_sums.append(s)
            if ok:
                U.append(candidate)
                for s in new_sums:
                    used_sums.add(s)
                found = True
                break
        if not found:
            break
    
    return U


def _geometric_set(n, ratio):
    """Build a geometric progression set: {0, floor(r), floor(r^2), ..., floor(r^(n-1))}."""
    if n < 1:
        return [0]
    U = [0]
    val = 1
    for i in range(1, n):
        val = int(val * ratio + 0.5)
        if val <= U[-1]:
            val = U[-1] + 1
        U.append(val)
    return U


def _binary_sidon_set(n):
    """Build a Sidon set using binary representation with carefully chosen bits.
    
    Use elements that are sums of distinct powers of a large base.
    U = {sum of selected base^i}
    
    For base >= 4, if we use at most 2 bits per element, all sums are distinct.
    """
    base = 4
    # Generate all subsets of {0,1,...,k-1} with at most 2 elements
    # This gives a Sidon set
    elements = [0]
    # Single bits
    for i in range(n):
        val = base ** i
        if val not in elements:
            elements.append(val)
        if len(elements) >= n:
            break
    # If we need more, use pairs
    if len(elements) < n:
        for i in range(n):
            for j in range(i+1, n):
                val = base**i + base**j
                if val not in elements:
                    elements.append(val)
                if len(elements) >= n:
                    break
            if len(elements) >= n:
                break
    return sorted(elements[:n])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    def try_set(U):
        nonlocal best_score, best_set
        if not U or len(U) < 2:
            return
        U = sorted(set(U))
        if 0 not in U:
            U = sorted([0] + U)
        score = _score(U)
        if score > best_score:
            best_score = score
            best_set = list(U)
    
    # Strategy 1: Greedy Sidon sets with various sizes and max values
    for n in range(3, 50):
        if time.time() > deadline:
            break
        # The Sidon set will have max value roughly n^2 * log(n) for greedy
        # Try a few max_val options
        for max_val in [n * n, n * n * 2, n * n * 4, n * n * 8]:
            if time.time() > deadline:
                break
            U = _greedy_sidon_set(n, max_val)
            if len(U) >= n:
                try_set(U)
    
    # Strategy 2: Geometric progressions
    for n in range(3, 60):
        if time.time() > deadline:
            break
        for ratio_100 in range(102, 300):  # ratio from 1.02 to 3.00
            if time.time() > deadline:
                break
            ratio = ratio_100 / 100.0
            U = _geometric_set(n, ratio)
            try_set(U)
    
    # Strategy 3: Binary-based Sidon sets
    for n in range(3, 30):
        if time.time() > deadline:
            break
        U = _binary_sidon_set(n)
        try_set(U)
    
    # Strategy 4: Local refinement - try to improve best set
    if time.time() < deadline - 30 and best_set:
        best_set_copy = list(best_set)
        max_u = max(best_set_copy)
        n_best = len(best_set_copy)
        
        # Try adding one more element
        for candidate in range(1, max_u + n_best * 10 + 50):
            if time.time() > deadline:
                break
            if candidate in best_set_copy:
                continue
            U = sorted(best_set_copy + [candidate])
            try_set(U)
        
        # Try removing one element
        for i in range(len(best_set_copy)):
            if time.time() > deadline:
                break
            if best_set_copy[i] == 0:
                continue
            U = [x for j, x in enumerate(best_set_copy) if j != i]
            try_set(U)
    
    # Strategy 5: More aggressive greedy with larger max values for smaller n
    if time.time() < deadline - 10:
        for n in range(3, 20):
            if time.time() > deadline:
                break
            for max_val_mult in [10, 20, 50, 100]:
                if time.time() > deadline:
                    break
                max_val = n * n * max_val_mult
                U = _greedy_sidon_set(n, max_val)
                if len(U) >= n:
                    try_set(U)
    
    # Final validation
    if not best_set or 0 not in best_set:
        return [0, 1, 3]
    
    return sorted(set(best_set))

# EVOLVE-BLOCK-END
