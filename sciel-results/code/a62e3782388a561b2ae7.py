import time
import itertools

# EVOLVE-BLOCK-START

def _can_add(S_set, x):
    """Check if x can be added to S_set without creating a triple violation."""
    # For any two distinct a, b in S_set, triple (x,a,b) violates if
    # x+a, x+b, a+b all in S_set
    for a, b in itertools.combinations(S_set, 2):
        if (x + a) in S_set and (x + b) in S_set and (a + b) in S_set:
            return False
    return True

def _greedy_add(S_set, all_nums, deadline):
    """Repeatedly add any element that can be added without violation."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in sorted(all_nums - S_set):
            if time.time() > deadline:
                return S_set
            if _can_add(S_set, x):
                S_set.add(x)
                improved = True
    return S_set

def construct():
    deadline = time.time() + 80  # safety margin
    
    all_nums = set(range(1, 129))
    
    # Start with {63..128} which is valid (b+c >= 64+65=129 > 128 for any distinct a<b<c)
    S = set(range(63, 129))
    S = _greedy_add(S, all_nums, deadline)
    best = set(S)
    
    # Local search: try removing one element, then greedy add
    for remove in list(best):
        if time.time() > deadline:
            break
        S2 = set(best)
        S2.discard(remove)
        S2 = _greedy_add(S2, all_nums, deadline)
        if len(S2) > len(best):
            best = set(S2)
    
    # Another round of removals on the improved best
    for remove in list(best):
        if time.time() > deadline:
            break
        S2 = set(best)
        S2.discard(remove)
        S2 = _greedy_add(S2, all_nums, deadline)
        if len(S2) > len(best):
            best = set(S2)
    
    result = sorted(best)
    return list(result)

# EVOLVE-BLOCK-END
