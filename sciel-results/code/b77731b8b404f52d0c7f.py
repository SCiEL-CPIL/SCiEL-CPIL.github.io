import math
import time

def _squarefree_kernel(n):
    """Return frozenset of primes with odd exponent in factorization of n."""
    if n <= 1:
        return frozenset()
    result = set()
    f = n
    d = 2
    while d * d <= f:
        if f % d == 0:
            exp = 0
            while f % d == 0:
                f //= d
                exp += 1
            if exp % 2 == 1:
                result.add(d)
        d += 1 if d == 2 else 2
    if f > 1:
        result.add(f)
    return frozenset(result)

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    target_kernel = frozenset([2, 3, 5, 7])
    
    # Binary search on W = max_selected - 210
    # W must be at least 1 (since values > 210)
    # Upper bound: 10210 - 210 = 10000
    
    def can_solve(W):
        """Check if we can select distinct ints in (210, 210+W] with product*210 being perfect square."""
        if time.time() > deadline:
            return None  # timeout
        
        lo, hi = 211, 210 + W
        if hi < lo:
            return None
        
        # Collect all candidate numbers and their kernels
        candidates = []
        for x in range(lo, hi + 1):
            k = _squarefree_kernel(x)
            candidates.append((x, k))
        
        # Separate perfect squares (empty kernel) from others
        free_squares = [(x, k) for x, k in candidates if len(k) == 0]
        others = [(x, k) for x, k in candidates if len(k) > 0]
        
        # We need XOR of all selected kernels == target_kernel
        # Strategy: greedily pick numbers to build up the kernel
        # Start with target, try to cancel it out
        
        # Try: include all free squares (they don't affect kernel), then solve for others
        # The combined kernel of selected 'others' must equal target_kernel
        
        # Greedy approach: sort by value, try to include numbers that help reduce kernel distance
        # Better: use the fact that we can pick any subset of size <= 100
        
        # Simple greedy: track current_kernel, start empty, want to reach target_kernel
        # For each number in ascending order, decide to include or not
        # This is NP-hard in general, but with small kernels we can be smart
        
        # Alternative: since we just need ANY valid subset, try:
        # 1. Pick numbers whose kernels are subsets of target or that cancel
        # 2. Specifically, look for numbers with kernel {2}, {3}, {5}, {7} to pair up
        
        # Most direct: brute force over small subsets is too slow
        # Instead: note that if we pick a number with kernel K, it toggles those primes
        # We need final XOR = target_kernel
        
        # Greedy: sort others by value, for each, if including it gets us closer to target, include it
        # "Closer" = smaller symmetric difference with target
        
        current_kernel = frozenset()
        selected = list(free_squares)  # include all free squares
        
        # Sort others by value
        others_sorted = sorted(others, key=lambda t: t[0])
        
        for x, k in others_sorted:
            if time.time() > deadline:
                return None
            new_kernel = frozenset(current_kernel ^ k)
            # Check if new_kernel is "closer" to target
            curr_dist = len(current_kernel ^ target_kernel)
            new_dist = len(new_kernel ^ target_kernel)
            if new_dist < curr_dist:
                current_kernel = new_kernel
                selected.append((x, k))
            elif new_dist == curr_dist and len(selected) < 100:
                # Tie: prefer smaller value (already sorted)
                current_kernel = new_kernel
                selected.append((x, k))
        
        # Check if we achieved target
        if current_kernel == target_kernel and 1 <= len(selected) <= 100:
            return True
        
        # Try a second pass: maybe we need to be more selective
        # Try picking only numbers that are "useful" - have small kernels
        # Specifically, prioritize numbers with kernel size 1
        
        current_kernel = frozenset()
        selected2 = list(free_squares)
        
        # Prioritize: kernel size 1 first, then 2, etc.
        others_by_size = sorted(others, key=lambda t: (len(t[1]), t[0]))
        
        for x, k in others_by_size:
            if time.time() > deadline:
                return None
            new_kernel = frozenset(current_kernel ^ k)
            curr_dist = len(current_kernel ^ target_kernel)
            new_dist = len(new_kernel ^ target_kernel)
            if new_dist <= curr_dist:
                current_kernel = new_kernel
                selected2.append((x, k))
        
        if current_kernel == target_kernel and 1 <= len(selected2) <= 100:
            return True
        
        # Try: explicitly look for pairs/triples that cancel
        # For each prime in target, find a number with that prime in its kernel
        # This is getting complex; return False for now and let binary search try larger W
        return False
    
    # Binary search on W
    lo_W = 1
    hi_W = 10000
    best_W = None
    
    # First, try small W values
    while lo_W <= hi_W and time.time() < deadline:
        mid_W = (lo_W + hi_W) // 2
        result = can_solve(mid_W)
        if result is None:  # timeout
            break
        if result:
            best_W = mid_W
            hi_W = mid_W - 1
        else:
            lo_W = mid_W + 1
    
    if best_W is None:
        # Fallback: return a known valid answer
        # 4*210 = 840, product = 840*210 = 176400 = 420^2. Valid.
        # max - 210 = 840 - 210 = 630
        return [840]
    
    # Reconstruct the actual set for best_W
    W = best_W
    lo, hi = 211, 210 + W
    
    candidates = []
    for x in range(lo, hi + 1):
        k = _squarefree_kernel(x)
        candidates.append((x, k))
    
    free_squares = [(x, k) for x, k in candidates if len(k) == 0]
    others = [(x, k) for x, k in candidates if len(k) > 0]
    
    # Use the same greedy that worked
    current_kernel = frozenset()
    selected = list(free_squares)
    
    others_sorted = sorted(others, key=lambda t: t[0])
    for x, k in others_sorted:
        new_kernel = frozenset(current_kernel ^ k)
        curr_dist = len(current_kernel ^ target_kernel)
        new_dist = len(new_kernel ^ target_kernel)
        if new_dist < curr_dist:
            current_kernel = new_kernel
            selected.append((x, k))
        elif new_dist == curr_dist and len(selected) < 100:
            current_kernel = new_kernel
            selected.append((x, k))
    
    # If that didn't work, try the other strategy
    if current_kernel != target_kernel or not (1 <= len(selected) <= 100):
        current_kernel = frozenset()
        selected = list(free_squares)
        others_by_size = sorted(others, key=lambda t: (len(t[1]), t[0]))
        for x, k in others_by_size:
            new_kernel = frozenset(current_kernel ^ k)
            curr_dist = len(current_kernel ^ target_kernel)
            new_dist = len(new_kernel ^ target_kernel)
            if new_dist <= curr_dist:
                current_kernel = new_kernel
                selected.append((x, k))
    
    # Final verification
    if current_kernel == target_kernel and 1 <= len(selected) <= 100:
        result_list = [x for x, k in selected]
        # Verify
        prod = 210
        for x in result_list:
            prod *= x
        if _is_perfect_square(prod):
            return result_list
    
    # Fallback
    return [840]
