"""Fresh design: multi-start greedy + local search for max P-subset in [1,256]."""
import time
import random

N = 256
DEADLINE_MARGIN = 5.0  # seconds to keep before 90s kill

def _valid_addition(a, present):
    """Check if a can be added to `present` (all elements > a).
    For every b,c in present with b>=c>a, need a not divide b+c."""
    # present contains only elements > a (we add in decreasing order)
    # Need: for all b,c in present, a does not divide b+c
    # b+c ranges from (min+min) to (max+max)
    # a divides b+c iff b+c is a multiple of a
    # Since b,c > a, b+c > 2a, so b+c can be 2a,3a,... but b,c>a means b+c>2a
    # Actually b,c >= a+1, so b+c >= 2a+2
    # a divides b+c means b+c = k*a for some k >= 3 (since b+c > 2a)
    # Equivalent: (b+c) % a == 0
    # We can check: does there exist b,c in present such that (b+c) % a == 0?
    # Faster: for each b in present, check if there exists c in present with (b+c) % a == 0
    # i.e., c % a == (-b) % a
    # So: compute residues of all elements in present mod a, check if any two sum to 0 mod a
    residues = {}
    for x in present:
        r = x % a
        residues[r] = residues.get(r, 0) + 1
    # Check: exists r1, r2 such that (r1+r2) % a == 0 and both appear in present
    # (r1 == r2) needs count >= 2, else just need both present
    for r, cnt in residues.items():
        need = (-r) % a
        if need == r:
            if cnt >= 2:
                return False
        else:
            if need in residues:
                return False
    return True

def _greedy_from_tail(tail_start, rng=None):
    """Start with [tail_start, 256], try to add elements from tail_start-1 down to 1."""
    present = set(range(tail_start, N + 1))
    for a in range(tail_start - 1, 0, -1):
        if _valid_addition(a, present):
            present.add(a)
    return present

def _local_search(present, rng, deadline):
    """Try to improve by swapping: remove one element, add another (smaller)."""
    present = set(present)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element and adding a smaller one
        for rem in sorted(present, reverse=True):
            if time.time() >= deadline:
                break
            present_no_rem = present - {rem}
            # Try adding elements not in present_no_rem that are < rem
            for add in range(1, rem):
                if add in present_no_rem:
                    continue
                if _valid_addition(add, present_no_rem):
                    # Check the full set is valid (rem was removed, add added)
                    # Since add < rem and all other elements > add, just need _valid_addition
                    present = present_no_rem | {add}
                    improved = True
                    break
            if improved:
                break
    return present

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    rng = random.Random(42)
    
    best = set(range(192, N + 1))  # baseline: upper tail of size 65
    
    # Multi-start: try various tail starts
    tail_starts = list(range(128, N + 1))
    rng.shuffle(tail_starts)
    
    for ts in tail_starts:
        if time.time() >= deadline:
            break
        # Quick greedy
        candidate = _greedy_from_tail(ts, rng)
        if len(candidate) > len(best):
            best = candidate
    
    # Local search on best
    if time.time() < deadline:
        best = _local_search(best, rng, deadline)
    
    # Additional multi-start with random perturbations
    for _ in range(200):
        if time.time() >= deadline:
            break
        ts = rng.randint(100, N)
        candidate = _greedy_from_tail(ts, rng)
        # Small random perturbation: remove a random element, try to add others
        if len(candidate) >= len(best):
            candidate = _local_search(candidate, rng, deadline)
        if len(candidate) > len(best):
            best = candidate
    
    return list(best)

# EVOLVE-BLOCK-START
# (construct defined above; keeping block markers for compatibility)
# EVOLVE-BLOCK-END
