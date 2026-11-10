# EVOLVE-BLOCK-START
import random
import time

def _can_add(existing, candidate, target):
    """Check if target is NOT the mean of any subset of existing with size >= 2."""
    n = len(existing)
    if n < 2:
        return True
    # subset sum DP: can we make sum = k*target with k in [2, n]?
    need_max = n * target
    dp = 1
    for v in existing:
        if v > need_max:
            continue
        dp |= dp << v
    for k in range(2, n + 1):
        s = k * target
        if s > need_max:
            break
        if (dp >> s) & 1:
            return False
    return True

def _verify(arr):
    """Full verification: no element is mean of any subset of >=2 others."""
    n = len(arr)
    for i in range(n):
        target = arr[i]
        others = [arr[j] for j in range(n) if j != i]
        if len(others) < 2:
            continue
        need_max = len(others) * target
        dp = 1
        for v in others:
            if v > need_max:
                continue
            dp |= dp << v
        for k in range(2, len(others) + 1):
            s = k * target
            if s > need_max:
                break
            if (dp >> s) & 1:
                return False
    return True

def _greedy_construct(order, deadline):
    """Greedy construction following the given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(selected, x, x):
            selected.append(x)
        if len(selected) >= 48:
            break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin before 90s kill
    rng = random.Random(42)
    
    best = []
    
    # Strategy 1: Try multiple random orderings
    # Larger numbers first tend to work better (less likely to be means)
    # Smaller numbers first fill the space
    
    # Start with a known-good greedy: try 1..512 in natural order
    natural = _greedy_construct(range(1, 513), deadline)
    if _verify(natural):
        best = natural
    
    # Try descending order
    if time.time() < deadline:
        desc = _greedy_construct(range(512, 0, -1), deadline)
        if _verify(desc):
            best = desc if len(desc) > len(best) else best
    
    # Try random shuffles biased toward certain ranges
    attempts = 0
    while time.time() < deadline and attempts < 200:
        attempts += 1
        # Generate a random ordering with bias
        # Mix: some large numbers, some small
        order = list(range(1, 513))
        rng.shuffle(order)
        result = _greedy_construct(order, deadline)
        if len(result) > len(best) and _verify(result):
            best = result
    
    # Final verification
    if not _verify(best):
        # Fallback: use the natural greedy
        best = _greedy_construct(range(1, 513), deadline)
        if not _verify(best):
            best = list(range(1, 49))
    
    return best[:48]
# EVOLVE-BLOCK-END
