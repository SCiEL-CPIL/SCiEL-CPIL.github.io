import time
import random
import numpy as np

P = {'n': 128}

def construct():
    n = P['n']
    deadline = time.time() + 88.0
    
    # Strategy:
    # 1. Split the range [1, 128] into two parts: L (low) and U (high).
    # 2. Ensure U is "sum-free" relative to itself and L is "weakly sum-free" relative to itself.
    #    Specifically, we want to avoid any triple (a,b,c) in S such that a+b, a+c, b+c are in S.
    #    If we choose L and U such that:
    #    - All sums of two elements in U are > max(U) (so not in U) and > max(S) (so not in S).
    #      This requires min(U) + min(U) > max(S). Since max(S) is likely in U, this is hard.
    #      Alternative: Ensure that no sum of two elements in U is in U.
    #      If U is a Sidon set or sum-free set, this helps.
    #    - Actually, the known improvement was "L union U where L is weakly sum-free and U is sum-free,
    #      exploiting the gap between 2*max(L) and min(U)".
    #      If 2*max(L) < min(U), then sums of elements in L are < min(U).
    #      If sums of elements in L are in L, that's fine as long as no triple forms.
    #      If sums of elements in L are in U, that's bad if combined with other sums.
    #      
    #    Let's use a simpler structural guarantee:
    #    Let S = L ∪ U.
    #    Condition: For any a,b,c in S, NOT (a+b in S AND a+c in S AND b+c in S).
    #    
    #    If we make U such that for any x,y in U, x+y is NOT in U (U is sum-free w.r.t U),
    #    and for any x in L, y in U, x+y is NOT in U (cross sums not in U),
    #    and for any x,y in L, x+y is NOT in L (L is sum-free w.r.t L),
    #    then NO sum of any two elements in S is in S.
    #    If no sum is in S, then no triple can exist (since a+b must be in S for the triple condition).
    #    So, if S is "sum-free" (no x,y,z in S with x+y=z), it is valid.
    #    
    #    Max size of sum-free subset of [1, N] is ceil(N/2).
    #    For N=128, max size is 64.
    #    But we can do better if we allow some sums to be in S, as long as they don't form a triangle.
    #    
    #    The "two-part" idea:
    #    Let L be a set of small numbers, U be a set of large numbers.
    #    If we ensure that all sums of pairs from L are in L (or outside S),
    #    and all sums of pairs from U are outside S,
    #    and cross sums are outside S,
    #    we avoid triangles.
    #    
    #    Actually, the most robust simple strategy for high score is often:
    #    Take the top half: {65, ..., 128}. Size 64.
    #    Can we add any numbers from 1..64?
    #    If we add x < 65, we need to ensure x+y is not in S for any y in S such that x+z and y+z are also in S.
    #    
    #    Let's try a Greedy Construction with Bitmask Validation.
    #    Represent S as a bitmask of 128 bits.
    #    Precompute for each pair (i, j) the index k = i+j.
    #    A triple (a,b,c) is bad if:
    #    bit(a) & bit(b) & bit(c) & bit(a+b) & bit(a+c) & bit(b+c) are all 1.
    #    
    #    We can iterate through all 128 numbers and decide to include or exclude.
    #    To maximize size, we prefer to include.
    #    But checking validity of the full set is O(N^3).
    #    
    #    Alternative: Backtracking with pruning? Too slow for 128.
    #    
    #    Let's stick to the "Two-Part" structural insight from the cell finding.
    #    Finding: "L union U where L is weakly sum-free and U is sum-free, exploiting the gap... increased size from 66 to 81."
    #    This suggests we can get ~81.
    #    
    #    How to construct such a set?
    #    Let U be a sum-free set in the upper range. E.g., {65..128} is sum-free?
    #    Min sum in U: 65+66=131 > 128. So all sums of two elements in U are > 128, thus not in [1,128].
    #    So U = {65..128} is sum-free w.r.t [1,128].
    #    Now, can we add elements from L = {1..64}?
    #    If we add x in L, we must ensure that x+y is not in S for any y in S such that the triple condition holds.
    #    Actually, if U is sum-free w.r.t [1,128], then for any y,z in U, y+z not in S.
    #    So any triple involving two elements from U is safe (since one of the sums will be y+z which is not in S).
    #    The only dangerous triples are those with 0 or 1 elements from U.
    #    Case 1: All 3 in L. Then we need L to be such that no triple in L has all 3 sums in S.
    #             Since sums of L elements are <= 64+63=127, they could be in L or U.
    #    Case 2: 2 in L, 1 in U. Let a,b in L, c in U.
    #             Sums: a+b (in [2,127]), a+c (in [66,192]), b+c (in [66,192]).
    #             For the triple to be bad, a+b, a+c, b+c must be in S.
    #             Since c >= 65, a+c >= 66. If a+c is in S, it must be in U (since L is 1..64).
    #             So we need a+c in U and b+c in U.
    #             Also a+b in S (could be L or U).
    #    Case 3: 1 in L, 2 in U. Safe as per above (sum of two U elements not in S).
    #    Case 4: 3 in U. Safe.
    #    
    #    So we only need to worry about triples with >= 2 elements in L.
    #    If we choose L to be very small, we minimize risk.
    #    But we want to maximize |S| = |L| + |U|.
    #    |U| is fixed at 64 if we take all of {65..128}.
    #    Can we add more to U? No, max is 128.
    #    Can we make U larger? No, U is defined by the gap.
    #    What if we don't take all of {65..128}?
    #    If we remove some from U, we can potentially add more to L.
    #    
    #    Let's try a Greedy approach:
    #    Start with U = {65..128}.
    #    Try to add elements from 1..64 one by one.
    #    For each candidate x in 1..64, check if adding x creates a bad triple.
    #    A bad triple involving x can be:
    #    1. (x, y, z) where y,z in L (already added).
    #    2. (x, y, u) where y in L, u in U.
    #    3. (x, u1, u2) where u1,u2 in U. -> Safe, because u1+u2 not in S.
    #    
    #    So we only check triples with x and at least one other from L, or x and two from L.
    #    Actually, if we add x, we need to check:
    #    - For any y, z in current L: if (x,y,z) is a bad triple.
    #    - For any y in current L, u in U: if (x,y,u) is a bad triple.
    #    
    #    This check is fast if L is small.
    #    
    #    Algorithm:
    #    1. U = set(range(65, 129))
    #    2. L = set()
    #    3. S = U ∪ L
    #    4. For x in 1..64:
    #       Tentatively add x to L.
    #       Check if S is valid.
    #       If valid, keep x in L.
    #       Else, remove x.
    #    5. Return S.
    
    U = set(range(65, 129))
    L = set()
    
    # Helper to check if adding x to L keeps S valid
    # S = U ∪ L ∪ {x}
    # We only need to check triples involving x.
    # Triples (a,b,c) with a<b<c.
    # If x is the largest, check (a,b,x).
    # If x is the smallest, check (x,b,c).
    # If x is in middle, check (a,x,c).
    
    def is_valid_with_x(x, L, U):
        # L and U are sets. x is not in L or U.
        # New L' = L ∪ {x}
        # New S = U ∪ L'
        # Check all triples in S that involve x.
        # Triples are distinct a,b,c.
        
        # Case 1: x is the largest element in the triple.
        # Triple (a, b, x) with a, b in L.
        # Need a+b in S, a+x in S, b+x in S.
        # a+x and b+x are > 64 (since x>=1, a>=1, but if x is small, a+x might be in L or U).
        # Actually, if x < 65, a+x < 128.
        
        for a in L:
            for b in L:
                if a == b: continue
                if a > b: continue # assume a < b
                # Triple (a, b, x)
                # Sums: a+b, a+x, b+x
                if (a+b) in U or (a+b) in L:
                    if (a+x) in U or (a+x) in L:
                        if (b+x) in U or (b+x) in L:
                            return False
        
        # Case 2: x is the smallest element in the triple.
        # Triple (x, b, c) with b, c in L.
        for b in L:
            for c in L:
                if b == c: continue
                if b > c: continue
                # Triple (x, b, c)
                # Sums: x+b, x+c, b+c
                if (x+b) in U or (x+b) in L:
                    if (x+c) in U or (x+c) in L:
                        if (b+c) in U or (b+c) in L:
                            return False
        
        # Case 3: x is in the middle.
        # Triple (a, x, c) with a in L, c in L.
        for a in L:
            for c in L:
                if a == c: continue
                if a > c: continue
                # Triple (a, x, c)
                # Sums: a+x, a+c, x+c
                if (a+x) in U or (a+x) in L:
                    if (a+c) in U or (a+c) in L:
                        if (x+c) in U or (x+c) in L:
                            return False
        
        # Case 4: x and one other in L, one in U.
        # Triple (x, y, u) with y in L, u in U.
        # Sums: x+y, x+u, y+u.
        # x+u > 64 (since u>=65, x>=1). So x+u is in [66, 192].
        # For x+u to be in S, it must be in U (since L is 1..64).
        # y+u is also in [66, 192], so must be in U.
        # x+y is in [2, 127], so could be in L or U.
        
        for y in L:
            for u in U:
                # Triple (x, y, u)
                # Note: x, y, u are distinct.
                # Check sums
                s1 = x + y
                s2 = x + u
                s3 = y + u
                
                # s2 and s3 must be in U
                if s2 in U and s3 in U:
                    if s1 in U or s1 in L:
                        return False
        
        # Case 5: x and two in U.
        # Triple (x, u1, u2) with u1, u2 in U.
        # Sums: x+u1, x+u2, u1+u2.
        # u1+u2 > 128, so not in S.
        # So this case is always safe.
        
        return True

    # Greedy addition
    # Order of adding x matters. Smaller x might block larger x?
    # Let's try adding in increasing order.
    for x in range(1, 65):
        if is_valid_with_x(x, L, U):
            L.add(x)
            
    S = U | L
    best_set = set(S)
    best_size = len(S)
    
    # Try to improve by swapping?
    # Maybe remove a small x from L to allow a larger x?
    # Or remove an element from U to allow more in L?
    
    # Let's try a second pass:
    # Randomly remove an element from L and see if we can add two? Unlikely.
    # Try removing an element from U (larger ones first?) to allow more in L?
    # If we remove u from U, we might be able to add more to L.
    # But we lose 1 from U. We need to gain >1 from L to improve.
    
    # Let's try a simple local search on the final set.
    # But checking validity of the whole set is expensive.
    # We can use the incremental check if we only change one element.
    
    current_set = set(S)
    
    # Try to add any remaining element from 1..64 that was rejected?
    # No, they were rejected for a reason.
    
    # Try swaps: Remove x from L, Add y from U? No, U is fixed.
    # Remove x from U, Add y to L?
    
    # Let's try to optimize U.
    # What if U is not {65..128}?
    # What if U = {70..128}? Size 59.
    # Then L can be larger?
    # Gap: 2*max(L) < min(U).
    # If min(U)=70, max(L) < 35.
    # L subset of {1..34}. Max size 34.
    # Total 59+34 = 93?
    # Wait, the condition "2*max(L) < min(U)" is sufficient but not necessary.
    # The previous finding said size 81 with gap.
    # Let's try different splits.
    
    for min_u in range(60, 80):
        if time.time() > deadline:
            break
        U_test = set(range(min_u, 129))
        # L must be such that no bad triples.
        # We can greedily add to L from 1..min_u-1.
        L_test = set()
        for x in range(1, min_u):
            if is_valid_with_x(x, L_test, U_test):
                L_test.add(x)
        
        size = len(U_test) + len(L_test)
        if size > best_size:
            best_size = size
            best_set = U_test | L_test
            # Update current_set for potential further search
            current_set = set(best_set)

    # Final check and return
    return list(best_set)

