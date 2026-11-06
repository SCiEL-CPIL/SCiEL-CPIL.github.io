"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _has_4ap_in_set(S, n=512):
    """Check if set S contains a nonconstant 4-term arithmetic progression."""
    S_set = set(S)
    for a in S:
        for d in range(1, (n - a) // 3 + 1):
            if (a + d) in S_set and (a + 2*d) in S_set and (a + 3*d) in S_set:
                return True
    return False

def _can_add(x, S, n=512):
    """Check if adding x to S creates a 4-term AP. S is a set."""
    # x as 4th term: a, a+d, a+2d, x => d=(x-a)//3
    for a in S:
        if x > a:
            diff = x - a
            if diff % 3 == 0:
                d = diff // 3
                if d > 0 and (a + d) in S and (a + 2*d) in S:
                    return False
        # x as 3rd term: a, a+d, x, x+d => d=(x-a)//2
        if x > a:
            diff = x - a
            if diff % 2 == 0:
                d = diff // 2
                if d > 0 and (a + d) in S and (x + d) <= n and (x + d) in S:
                    return False
        # x as 2nd term: a, x, x+d, x+2d => d=x-a
        d = x - a
        if d > 0 and (x + d) <= n and (x + 2*d) <= n:
            if (x + d) in S and (x + 2*d) in S:
                return False
        # x as 1st term: x, x+d, x+2d, x+3d => need x+d, x+2d, x+3d in S
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                return False
    return True

def _ternary_01_set(n):
    """Generate all numbers in [1, n] whose base-3 representation contains only 0s and 1s."""
    result = set()
    # 3^6 = 729 > 512, so 6 digits suffice
    for mask in range(1, 1 << 6):
        val = 0
        for bit in range(6):
            if mask & (1 << bit):
                val += 3 ** bit
        if 1 <= val <= n:
            result.add(val)
    return result

def _greedy_construct(n=512):
    """Greedy construction: add x if it doesn't create a 4-AP."""
    selected = set()
    for x in range(1, n + 1):
        if _can_add(x, selected, n):
            selected.add(x)
    return selected

def _greedy_add_improvement(S, n=512, deadline=None):
    """Try to add more elements to S greedily."""
    S = set(S)
    for x in range(1, n + 1):
        if deadline is not None and time.time() > deadline:
            break
        if x in S:
            continue
        if _can_add(x, S, n):
            S.add(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline to stay under 60s limit
    n = 512
    
    # Start with ternary 0-1 set (known 3-AP free, hence 4-AP free)
    best_set = _ternary_01_set(n)
    best_size = len(best_set)
    
    # Try greedy from scratch
    greedy_set = _greedy_construct(n)
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(greedy_set)
    
    # Try to improve ternary set by adding more elements
    improved = _greedy_add_improvement(best_set, n, deadline)
    if len(improved) > best_size:
        best_set = improved
        best_size = len(improved)
    
    # Try randomized restarts with different orderings
    random.seed(42)
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Create a random ordering
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        # Start with a subset of ternary set or empty
        if attempt % 2 == 0:
            current = set(_ternary_01_set(n))
        else:
            current = set()
        
        # Add elements in random order
        for x in order:
            if x in current:
                continue
            if _can_add(x, current, n):
                current.add(x)
        
        # Try to fill in remaining elements
        current = _greedy_add_improvement(current, n, deadline)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Final verification
    if _has_4ap_in_set(best_set, n):
        # Fallback to pure greedy
        best_set = _greedy_construct(n)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
