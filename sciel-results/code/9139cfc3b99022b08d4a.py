"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 128}

def _is_valid_set(S):
    """Check if set S has no three distinct members a,b,c where a+b, a+c, b+c are all in S."""
    S_set = set(S)
    for a, b, c in combinations(S, 3):
        if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
            return False
    return True

def _can_add_fast(x, S, S_set, upper_block):
    """
    Fast check if adding x to S is valid.
    S contains only elements < x (we process in increasing order).
    upper_block is a set of elements > 64.
    If a,b in S and a+b in upper_block, then a+b is in S.
    We need to ensure NOT (a+b in S AND a+x in S AND b+x in S).
    """
    for i in range(len(S)):
        a = S[i]
        ax = a + x
        if ax not in S_set:
            continue
        for j in range(i+1, len(S)):
            b = S[j]
            bx = b + x
            if bx not in S_set:
                continue
            ab = a + b
            # Check if ab is in S
            if ab in S_set:
                return False
            # Check if ab is in upper_block (which is in S)
            if ab in upper_block:
                return False
            # ab might be x? No, a,b < x so ab can be x. But x is not in S yet.
            # The condition is: a+b, a+c, b+c are in S ∪ {c}. Here c=x.
            # So a+b must be in S or a+b == x.
            if ab == x:
                return False
    return True

def _greedy_lower(upper_block, deadline):
    """
    Greedily construct the lower part of the set (1..64)
    given a fixed upper block (65..128).
    """
    S = []
    S_set = set()
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if _can_add_fast(x, S, S_set, upper_block):
            S.append(x)
            S_set.add(x)
    return S

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    n = P['n']
    
    # Strategy: Fix the upper block as 65..128 (64 elements).
    # This block is "sum-free" internally (min sum 65+66=131 > 128).
    # So no triple entirely in the upper block is forbidden.
    # We just need to pick the largest subset of 1..64 such that
    # no triple (a,b,c) from the combined set is forbidden.
    
    upper_block = set(range(65, 129))
    
    # We can try different orders for the lower block to see if we get more elements.
    # But since the constraint is monotonic (adding more elements only adds constraints),
    # the greedy approach in increasing order is usually optimal for this type of "independent set" problem on a matroid-like structure, 
    # or at least very close. Let's try a few random permutations of 1..64 to see if we can beat the natural order.
    
    best_lower = []
    best_size = 0
    
    # Natural order
    lower = _greedy_lower(upper_block, deadline)
    if len(lower) > best_size:
        best_lower = lower
        best_size = len(lower)
    
    # Try a few random orders
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, 65))
        random.shuffle(order)
        # We must still respect the "increasing x" assumption in _can_add_fast?
        # No, _can_add_fast assumes S contains elements < x.
        # If we shuffle, we can't use the fast check directly unless we sort S or adapt the check.
        # Let's just use the natural order. It's likely optimal or near-optimal.
        # Actually, let's just stick to natural order for speed and reliability.
        pass
    
    # Combine
    S = best_lower + list(upper_block)
    S.sort()
    
    # Verify
    if not _is_valid_set(S):
        # Fallback to just the upper block
        return list(upper_block)
        
    return [int(x) for x in S]

# EVOLVE-BLOCK-END
