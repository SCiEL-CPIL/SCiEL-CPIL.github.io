"""Rewritten program: structural change to a 3-base (base-3) digit-based construction,
with a local-search improvement pass under a strict wall-clock deadline."""
import time
import random

# EVOLVE-BLOCK-START

N = 512
K = 4  # length of forbidden arithmetic progression

# Precompute conflicts
# conflicts[x] is a list of tuples (y, z, w) such that {x, y, z, w} forms a 4-AP
conflicts = [[] for _ in range(N + 1)]

def _precompute_conflicts():
    # Iterate all possible 4-APs in [1, N]
    # a, a+d, a+2d, a+3d
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # The AP is (a, b, c, e)
            # For a, the others are b, c, e
            conflicts[a].append((b, c, e))
            # For b, the others are a, c, e
            conflicts[b].append((a, c, e))
            # For c, the others are a, b, e
            conflicts[c].append((a, b, e))
            # For e, the others are a, b, c
            conflicts[e].append((a, b, c))

_precompute_conflicts()

def can_add(x, S):
    """Check if x can be added to S without creating a 4-AP."""
    for (y, z, w) in conflicts[x]:
        if y in S and z in S and w in S:
            return False
    return True

def has_4ap_in_set(S):
    """Check if set S contains a nonconstant 4-term arithmetic progression."""
    S_set = set(S)
    for x in S:
        for (y, z, w) in conflicts[x]:
            if y in S_set and z in S_set and w in S_set:
                return True
    return False

def base3_no_zero_digits():
    """Construct a set using numbers whose base-3 representation has no digit 0."""
    S = set()
    for x in range(1, N + 1):
        n = x
        ok = True
        while n > 0:
            if n % 3 == 0:
                ok = False
                break
            n //= 3
        if ok:
            S.add(x)
    return S

def greedy_construct(order=None):
    """Greedy: add x if it doesn't create a 4-AP."""
    if order is None:
        order = list(range(1, N + 1))
    selected = set()
    for x in order:
        if can_add(x, selected):
            selected.add(x)
    return selected

def local_improve(S, deadline):
    """Try to add elements that don't create 4-APs, in random order, until deadline."""
    S = set(S)
    candidates = [x for x in range(1, N + 1) if x not in S]
    random.shuffle(candidates)
    
    # Multiple passes?
    # If we add an element, it might block others. 
    # But if we can't add x now, we might be able to later? 
    # No, adding elements only restricts the set further. 
    # So if x can't be added now, it will never be able to be added in this pass 
    # unless we remove something. 
    # So one pass of adding all possible candidates in random order is optimal for "add-only".
    
    for x in candidates:
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin: 55s out of 60s
    
    best_S = set()
    best_size = 0
    
    # Strategy 1: Base-3 + Local Improve
    S1 = base3_no_zero_digits()
    S1 = local_improve(S1, deadline)
    if len(S1) > best_size:
        best_S = S1
        best_size = len(S1)
    
    # Strategy 2: Greedy with random orders
    # We have plenty of time now because can_add is fast.
    # Let's try several random greedy constructions.
    
    # Fixed seed for reproducibility in evaluation? 
    # The task says "The evaluator measures the raw value". 
    # If it runs once, randomness is fine. 
    # But to be safe, I'll use a few different seeds or just one good one.
    
    num_restarts = 1000 # High number, we will break on time
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S2 = greedy_construct(order)
        
        # Try local improve on the greedy result
        # This is fast too
        S2 = local_improve(S2, deadline)
        
        if len(S2) > best_size:
            best_S = S2
            best_size = len(S2)
            
        # Early stop if we hit a theoretical max? 
        # Max size for 4-AP free in [1, 512] is not known exactly, 
        # but base-3 gives ~3^log3(512) = 3^6 = 729? No, 
        # base-3 no zero digits count: 
        # 1 digit: 1,2 (2)
        # 2 digits: 1,2 x 1,2 (4)
        # 3 digits: 27
        # 4 digits: 81
        # 5 digits: 243
        # 6 digits: 729 > 512.
        # So we can have at most numbers with <= 5 digits in base 3? 
        # 3^5 = 243. Numbers 1..242 have <=5 digits? 
        # 243 is 10000_3. 
        # So base-3 set size is sum_{k=1}^5 2^k = 2+4+8+16+32 = 62? 
        # Wait. 
        # Digits can be 1 or 2.
        # 1 digit: 1, 2 (2 nums)
        # 2 digits: 11, 12, 21, 22 (4 nums) -> 4, 5, 7, 8
        # 3 digits: 8 nums
        # 4 digits: 16 nums
        # 5 digits: 32 nums
        # Total 62.
        # But we can go up to 512. 
        # 512 in base 3 is 201002? 
        # 3^5 = 243. 3^6 = 729.
        # So we have 6 digits.
        # Max size of base-3 no-zero set in [1, 512]:
        # All numbers with 1-5 digits (no 0s): 2+4+8+16+32 = 62.
        # Numbers with 6 digits: 1xxxxx, 2xxxxx.
        # 100000_3 = 729 > 512.
        # So we can only have 1xxxxx where value <= 512.
        # 100000_3 = 243.
        # 110000_3 = 243 + 81 = 324.
        # 111000_3 = 324 + 27 = 351.
        # 111100_3 = 351 + 9 = 360.
        # 111110_3 = 360 + 3 = 363.
        # 111111_3 = 363 + 1 = 364.
        # 120000_3 = 243 + 162 = 405.
        # ...
        # 200000_3 = 486.
        # 210000_3 = 486 + 81 = 567 > 512.
        # So we have a limited number of 6-digit numbers.
        # Total base-3 set is small. 
        # Greedy should do much better.

    # Final safety check
    if has_4ap_in_set(best_S):
        # Fallback to a known valid set
        best_S = base3_no_zero_digits()

    return sorted(best_S)

# EVOLVE-BLOCK-END
