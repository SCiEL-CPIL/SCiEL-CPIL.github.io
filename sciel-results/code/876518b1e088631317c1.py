"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128
# Precompute conflict masks: for each pair (i, j), which k would form a forbidden triple
# Forbidden: three distinct a, b, c in S such that a+b, a+c, b+c are all in S.
# For a candidate element x to be added to set S, we need to check that no existing 
# pair (a, b) in S creates a forbidden triple with x.
# If x is added, and a, b are in S, then we need a+b, a+c=x, b+c=x NOT all in S.
# Actually the condition is: no three DISTINCT members a,b,c for which a+b, a+c, b+c are also selected.
# So if we have set S and want to add x, we check: for any two distinct a,b in S\{x},
# if a+b in S and a+x in S and b+x in S, then {a,b,x} would be a forbidden triple.
# But wait - the sums must also be in S. So we need a+b ∈ S, a+x ∈ S, b+x ∈ S.
# Since x is being added, a+x and b+x are new (or x itself if a or b is 0, but we're in 1..128).
# Actually a+x and b+x could already be in S. Let me reconsider.
# 
# The forbidden condition: exist distinct a,b,c in S such that a+b, a+c, b+c all in S.
# When adding x to S, new forbidden triples could involve x as one of {a,b,c} or as one of the sums.
# 
# Case 1: x is one of the three elements, say x=c. Then we need a,b in S, a≠b, a≠x, b≠x,
#         and a+b, a+x, b+x all in S.
# Case 2: x is one of the sums, say x=a+b. Then we need a,b,c in S with a+b=x, a+c in S, b+c in S.
#         But x is not in S yet (we're adding it), so x=a+b means x was already the sum.
#         Wait, x is being added to S. If x=a+b for some a,b in S, then after adding x, 
#         we need to check if there's a c in S such that a+c and b+c are also in S.
#         But that's a pre-existing condition on a,b,c - if a+c and b+c were already in S,
#         then {a,b,c} would already be forbidden without x. So this case doesn't create NEW violations.
# 
# Actually, let me re-examine. The violation is: three distinct a,b,c in S, and a+b,a+c,b+c all in S.
# When we add x, the only new way to create a violation is if x participates in the triple.
# - x as one of {a,b,c}: WLOG x=c. Need a,b in S\{x}, a≠b, and a+b∈S, a+x∈S, b+x∈S.
# - x as one of the sums: WLOG x=a+b. Need a,b,c in S, a≠b, a≠c, b≠c, a+c∈S, b+c∈S.
#   But this requires a,b,c already in S with a+c and b+c in S. If a+b was not in S before,
#   adding x=a+b doesn't create this violation because the violation requires a+b to be in S,
#   which it wasn't before. After adding x, a+b=x is in S, so we DO create a violation if 
#   a+c and b+c are in S. So this IS a new violation.
#
# So both cases matter. Let me use bitmasks for efficiency.

# conflict_pair_mask[i][j] = bitmask of all k such that {i,j,k} would be forbidden if all in S
# This is complex. Let me use a different approach: for each element x, precompute which 
# pairs (a,b) would create a violation with x.

# For element x, the conflicts are:
# 1. x as the third element: pairs (a,b) in S where a+b∈S, a+x∈S, b+x∈S
# 2. x as a sum: pairs (a,b) in S where a+b=x, and there exists c in S with a+c∈S and b+c∈S
#    Actually, for case 2, we need to check if adding x=a+b creates a violation with some c.
#    The violation would be {a,b,c} with a+b=x∈S, a+c∈S, b+c∈S.

# Let me precompute for each x, the set of (a,b) pairs that would conflict.
# But this is still O(N^2) per check. Let me think differently.

# Alternative: For each element x, define conflict_mask[x] as a bitmask over all elements y
# such that having both x and y in S could potentially lead to a violation.
# This is getting complex. Let me just use a direct check but optimize with numpy.

def _is_valid(S):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            s_ab = a + b
            if s_ab not in S_set:
                continue
            for k in range(j+1, n):
                c = S_list[k]
                if a + c in S_set and b + c in S_set:
                    return False
    return True

def _can_add(S_set, S_list, x):
    """Check if x can be added to S without creating a forbidden triple."""
    if x in S_set:
        return False
    n = len(S_list)
    # Check case 1: x is the third element c
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            if a + b in S_set and a + x in S_set and b + x in S_set:
                return False
    # Check case 2: x is a sum a+b
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            if a + b == x:
                # Check if there's a c in S such that a+c and b+c are in S
                for k in range(n):
                    c = S_list[k]
                    if c != a and c != b and a + c in S_set and b + c in S_set:
                        return False
    return True

def _greedy_construct():
    """Greedy construction: add elements in a random order if valid."""
    S = set()
    S_list = []
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(S, S_list, x):
            S.add(x)
            S_list.append(x)
    return S, S_list

def _local_search(S, S_list, time_limit=55.0):
    """Local search with swaps and additions."""
    start_time = time.time()
    best_S = set(S)
    best_S_list = list(S_list)
    best_size = len(best_S)
    
    current_S = set(S)
    current_S_list = list(S_list)
    
    while time.time() - start_time < time_limit:
        # Try random swaps: remove one element, try to add another
        if len(current_S_list) == 0:
            break
        idx = random.randint(0, len(current_S_list) - 1)
        to_remove = current_S_list[idx]
        new_S = set(current_S)
        new_S.remove(to_remove)
        new_S_list = [x for x in current_S_list if x != to_remove]
        
        # Try to add random elements
        added = False
        for x in random.sample(range(1, N + 1), min(20, N)):
            if x not in new_S and _can_add(new_S, new_S_list, x):
                new_S.add(x)
                new_S_list.append(x)
                added = True
                break
        
        if len(new_S) > best_size:
            best_S = set(new_S)
            best_S_list = list(new_S_list)
            best_size = len(best_S)
        
        # Accept with some probability (simulated annealing style)
        if len(new_S) >= len(current_S) or random.random() < 0.3:
            current_S = new_S
            current_S_list = new_S_list
    
    return best_S, best_S_list

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    best_S = set()
    best_S_list = []
    best_size = 0
    
    # Multiple restarts
    while time.time() - start_time < time_limit:
        S, S_list = _greedy_construct()
        if len(S) > best_size:
            best_S = set(S)
            best_S_list = list(S_list)
            best_size = len(S)
        
        # Local search
        if time.time() - start_time < time_limit - 10.0:
            S, S_list = _local_search(S, S_list, time_limit=time_limit - (time.time() - start_time) - 5.0)
            if len(S) > best_size:
                best_S = set(S)
                best_S_list = list(S_list)
                best_size = len(S)
        
        if best_size >= 80:  # Good enough, stop early
            break
    
    # Verify validity
    if not _is_valid(best_S_list):
        # Fallback to simple valid set
        best_S_list = list(range(N//2 + 1, N + 1))
    
    return list(best_S_list)
# EVOLVE-BLOCK-END
