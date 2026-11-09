"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128
DEADLINE = 85.0  # Safety margin before 90s kill

# Precompute conflict structure using bitmasks
# For each element x (1..128), we want to know which pairs (a,b) in S would
# create a forbidden triple if x were added.

# The forbidden condition: distinct a,b,c in S with a+b, a+c, b+c all in S.
# When adding x to S, violations can occur if:
# 1. x is one of {a,b,c}: say x=c. Need a,b in S, a+b in S, a+x in S, b+x in S.
# 2. x is one of the sums: say x=a+b. Need a,b,c in S, a+c in S, b+c in S.

# For efficiency, precompute for each x:
# - sum_partners[x] = set of (a,b) pairs where a+b = x (for case 2)
# - For case 1, we need to check if there exist a,b in S such that a+b in S AND a+x in S AND b+x in S

# Let's use a different approach: for each x, precompute a bitmask of elements y
# such that if both x and y are in S, they could be part of a forbidden triple.
# This is still complex. Let's use direct checking with bitmasks for the set.

# Represent S as a bitmask (128 bits). Use Python integers for bit operations.

def _popcount(x):
    return bin(x).count('1')

def _can_add_bitmask(S_mask, x):
    """Check if x can be added to S (represented as bitmask) without creating forbidden triple."""
    # x is 1-indexed, so bit position is x-1
    bit = 1 << (x - 1)
    if S_mask & bit:
        return False
    
    # Case 1: x is the third element c
    # Need: exist a,b in S, a!=b, a+b in S, a+x in S, b+x in S
    # For each a in S (a < x or a > x, but a != x):
    #   For each b in S (b != a, b != x):
    #     Check a+b in S, a+x in S, b+x in S
    # This is O(n^2) where n = |S|. Let's optimize.
    
    # For case 1: fix a in S. Then we need b in S such that:
    #   b != a, b != x
    #   a+b in S
    #   a+x in S (this is a constant check for fixed a)
    #   b+x in S (this depends on b)
    
    # So: for each a in S where a+x in S:
    #   We need b in S such that b != a, b != x, a+b in S, b+x in S
    #   i.e., b in (S & sum_mask[a] & sum_mask_x) where sum_mask[a] = {b : a+b in S}
    #   and sum_mask_x = {b : b+x in S}
    
    # Let's precompute for each a: sum_mask[a] = bitmask of b such that a+b <= 128 and (a+b) in S
    # But this depends on S, so we can't precompute globally.
    
    # Alternative: iterate over a in S
    S_list = [i+1 for i in range(N) if S_mask & (1 << i)]
    n = len(S_list)
    
    # Case 1: x as c
    for i in range(n):
        a = S_list[i]
        if a == x:
            continue
        ax = a + x
        if ax > N:
            continue
        if not (S_mask & (1 << (ax - 1))):
            continue
        # Now check if there's b in S, b != a, b != x, with a+b in S and b+x in S
        for j in range(n):
            b = S_list[j]
            if b == a or b == x:
                continue
            ab = a + b
            if ab > N:
                continue
            if not (S_mask & (1 << (ab - 1))):
                continue
            bx = b + x
            if bx > N:
                continue
            if S_mask & (1 << (bx - 1)):
                return False
    
    # Case 2: x as a+b
    # Need: exist a,b,c in S, a!=b, a!=c, b!=c, a+b=x, a+c in S, b+c in S
    for i in range(n):
        a = S_list[i]
        if a >= x:  # a+b = x, b >= 1, so a <= x-1
            continue
        b = x - a
        if b <= 0 or b >= x:  # b must be positive and distinct from a
            continue
        if b == a:
            continue
        # Check b in S
        if not (S_mask & (1 << (b - 1))):
            continue
        # Now check if there's c in S, c != a, c != b, with a+c in S and b+c in S
        for k in range(n):
            c = S_list[k]
            if c == a or c == b:
                continue
            ac = a + c
            if ac > N:
                continue
            if not (S_mask & (1 << (ac - 1))):
                continue
            bc = b + c
            if bc > N:
                continue
            if S_mask & (1 << (bc - 1)):
                return False
    
    return True

def _greedy(seed):
    """Greedy construction with random ordering."""
    rng = random.Random(seed)
    S_mask = 0
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    for x in candidates:
        if _can_add_bitmask(S_mask, x):
            S_mask |= (1 << (x - 1))
    return S_mask

def _local_search(S_mask, time_limit):
    """Local search: try swaps and additions."""
    start_time = time.time()
    best_mask = S_mask
    best_size = _popcount(best_mask)
    current_mask = S_mask
    current_size = best_size
    
    while time.time() - start_time < time_limit:
        # Try to add a random element
        x = random.randint(1, N)
        if not (current_mask & (1 << (x - 1))):
            if _can_add_bitmask(current_mask, x):
                new_mask = current_mask | (1 << (x - 1))
                new_size = current_size + 1
                if new_size > best_size:
                    best_mask = new_mask
                    best_size = new_size
                current_mask = new_mask
                current_size = new_size
            continue
        
        # Try a swap: remove one element, try to add another
        S_list = [i+1 for i in range(N) if current_mask & (1 << i)]
        if not S_list:
            break
        to_remove = random.choice(S_list)
        new_mask = current_mask & ~(1 << (to_remove - 1))
        new_size = current_size - 1
        
        # Try to add some elements
        added = False
        for x in random.sample(range(1, N + 1), min(15, N)):
            if not (new_mask & (1 << (x - 1))):
                if _can_add_bitmask(new_mask, x):
                    new_mask |= (1 << (x - 1))
                    new_size += 1
                    added = True
                    break
        
        if new_size > best_size:
            best_mask = new_mask
            best_size = new_size
        
        # Accept with some probability
        if new_size >= current_size or random.random() < 0.2:
            current_mask = new_mask
            current_size = new_size
    
    return best_mask, best_size

def _verify(result):
    """Verify no forbidden triple."""
    S_set = set(result)
    n = len(result)
    for i in range(n):
        a = result[i]
        for j in range(i + 1, n):
            b = result[j]
            if a + b in S_set:
                for k in range(j + 1, n):
                    c = result[k]
                    if a + c in S_set and b + c in S_set:
                        return False
    return True

def construct():
    start_time = time.time()
    time_limit = DEADLINE

    best_mask = 0
    best_size = 0

    # Phase 1: Multiple greedy restarts
    for seed in range(500):
        if time.time() > time_limit - 30:
            break
        m = _greedy(seed)
        sz = _popcount(m)
        if sz > best_size:
            best_size = sz
            best_mask = m

    # Phase 2: Local search from best
    if time.time() < time_limit - 10:
        best_mask, best_size = _local_search(best_mask, time_limit - 5)

    # Phase 3: More local search from different starting points
    while time.time() < time_limit - 5:
        m = _greedy(random.randint(0, 99999))
        sz = _popcount(m)
        if sz + 2 > best_size:
            m, sz = _local_search(m, time_limit - 3)
            if sz > best_size:
                best_size = sz
                best_mask = m

    # Convert to list
    result = [x for x in range(1, N + 1) if (best_mask & (1 << (x - 1)))]

    # Verify
    if not _verify(result):
        result = list(range(N // 2 + 1, N + 1))

    return result
# EVOLVE-BLOCK-END
