import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 88.0
    n = 128
    candidates = list(range(1, n + 1))
    
    # Precompute all forbidden triples (a, b, c) with a < b < c
    # such that a+b, a+c, b+c are all in 1..128.
    # Note: the condition is that a+b, a+c, b+c are in the SET S.
    # So for a candidate set S, a triple (a,b,c) is forbidden if a,b,c in S AND a+b, a+c, b+c in S.
    # This is a 6-tuple constraint, not just a 3-uniform hypergraph on the triple alone.
    # Wait, the constraint is: no THREE DISTINCT members a,b,c for which a+b, a+c and b+c are also selected.
    # So the "forbidden configuration" is a 6-element set {a,b,c,a+b,a+c,b+c} (with possible overlaps, but a,b,c distinct).
    # This is not a simple hypergraph on triples. It's a constraint on 6-tuples.
    
    # Let's re-think. The constraint is monotone: if S is valid, any subset is valid.
    # So we want the largest subset.
    
    # A different approach: Use the structure that {65..128} is sum-free.
    # Let U = {65..128}. Let L be a subset of {1..64}.
    # Forbiddance conditions involving elements from U:
    # 1. Three elements in U: impossible (sums > 128).
    # 2. Two in U, one in L: Let a,b in U, c in L.
    #    Need a+b, a+c, b+c in S.
    #    a+b > 128, so a+b not in S. Thus this triple is never forbidden.
    # 3. One in U, two in L: Let a in U, b,c in L.
    #    Need a+b, a+c, b+c in S.
    #    b+c <= 64+63 = 127. So b+c could be in L or U.
    #    a+b >= 65+1 = 66. So a+b is in U (if <=128) or >128.
    #    If a+b > 128, condition fails.
    #    If a+b <= 128, then a+b in U.
    #    So we need: a+b in U (always true if <=128 since U is all of 65..128), 
    #    a+c in U (if <=128), and b+c in S.
    #    Since U is fully included, a+b and a+c are in S if they are <= 128.
    #    So the condition reduces to: b+c in S AND a+b <= 128 AND a+c <= 128.
    #    Wait, if a+b > 128, then a+b is not in S, so the triple is not forbidden.
    #    So for a in U, b,c in L, the triple is forbidden iff:
    #    (a+b <= 128) AND (a+c <= 128) AND (b+c in S).
    #    Since a >= 65, b <= 64, a+b <= 128 implies b <= 128-a.
    #    Similarly c <= 128-a.
    #    So if we pick L to be "small" enough such that for all a in U, b,c in L, 
    #    either a+b > 128 or a+c > 128 or b+c not in L, we are safe.
    
    # 4. Three in L: Let a,b,c in L.
    #    Need a+b, a+c, b+c in S.
    #    a+b, a+c, b+c are in 1..127.
    #    They could be in L or in U.
    #    If any of them is in U, it's in S.
    #    If any of them is in L, it must be in L.
    
    # So the constraint is complex. Let's use a different structural insight.
    # What if we take L = {1..k} for some k?
    # Or L = {m..n}?
    
    # Let's try a randomized local search with a very fast validity check.
    # Represent S as a bitset of 128 bits.
    # Validity check: for each triple (a,b,c) in S, check if a+b, a+c, b+c in S.
    # We can precompute the list of all potential "bad" 6-tuples? No, too many.
    
    # Alternative: Use the fact that the problem is a maximum independent set in a graph?
    # No, it's a hypergraph.
    
    # Let's go back to the known good solution: size 80.
    # U = {65..128} (64 elements).
    # L = {17..32} (16 elements).
    # Total 80.
    # Can we do better?
    
    # Try to add more elements to L or change L.
    # What if L = {1..16}? 
    # b+c <= 31. a+b <= 65+16 = 81.
    # If a=65, b=16, a+b=81 in U.
    # We need b+c in S.
    # If L={1..16}, b+c is in 2..31.
    # If b+c is in {1..16}, it's in L.
    # If b+c is in {17..31}, it's not in L (if L={1..16}) and not in U (since U starts at 65).
    # So b+c is NOT in S.
    # Thus, if L={1..16}, for any a in U, b,c in L, b+c is not in S (since b+c <= 31 < 65).
    # So condition 3 is never triggered.
    # Condition 4: a,b,c in L={1..16}.
    # a+b, a+c, b+c are in 2..31.
    # If any of these is in {17..31}, it is not in S.
    # If all are in {1..16}, they are in S.
    # So we need to ensure that for any a,b,c in L, not all of a+b, a+c, b+c are in L.
    # This is a "weakly sum-free" condition on L? No, it's a 3-wise condition.
    # Let's check L={1..16}.
    # Take a=1, b=2, c=3. a+b=3, a+c=4, b+c=5. All in {1..16}. So forbidden!
    # So L={1..16} is not valid.
    
    # The previous solution used L={17..32}.
    # b+c >= 34. So b+c is not in L (since L ends at 32) and not in U (since U starts at 65).
    # So b+c is never in S.
    # Thus condition 3 is never triggered for L={17..32}.
    # Condition 4: a,b,c in {17..32}.
    # a+b >= 34. So a+b, a+c, b+c are all >= 34.
    # They are not in L (since L ends at 32).
    # They are not in U (since U starts at 65, and max sum is 32+31=63 < 65).
    # So a+b, a+c, b+c are never in S.
    # Thus L={17..32} is valid and independent of U.
    # So S = U ∪ L is always valid!
    # Size = 64 + 16 = 80.
    
    # Can we find a larger L?
    # We need L ⊆ {1..64} such that:
    # 1. For any a,b,c in L, a+b, a+c, b+c are not in L ∪ U.
    #    Since U={65..128}, this means a+b, a+c, b+c must be in {33..64} (the gap) or >128.
    #    Since max sum is 64+63=127, they can't be >128.
    #    So we need a+b, a+c, b+c ∈ {33..64} for all a,b,c in L? No, we need them NOT in S.
    #    S = L ∪ U.
    #    So we need a+b, a+c, b+c ∉ L and ∉ U.
    #    Since U={65..128}, and a+b <= 127, a+b ∉ U iff a+b <= 64.
    #    So we need a+b, a+c, b+c ∈ {1..64} \ L.
    #    This is a strong condition.
    
    # Let's try to maximize |L| subject to: for all distinct a,b,c in L, 
    # a+b, a+c, b+c ∉ L.
    # (Note: a+b could be in U? No, a+b <= 127. If a+b >= 65, it's in U, which is in S. 
    # So we must also ensure a+b < 65 for all pairs in L? 
    # Wait, if a+b >= 65, then a+b in U ⊆ S. 
    # Then for the triple to be forbidden, we need a+c in S and b+c in S.
    # If we can ensure that for any a,b,c in L, at least one of a+b, a+c, b+c is in {33..64}\L, 
    # or if two are in U, the third is not in S.
    
    # Let's use a heuristic: pick L such that the sums of pairs in L are mostly in the gap {33..64}.
    # The gap has 32 numbers.
    # If L is large, many pairs will have sums in the gap.
    # Let's try L = {33..64}.
    # Min sum = 33+34 = 67. This is in U!
    # So a+b in U.
    # Then we need a+c, b+c in S.
    # a+c >= 33+35 = 68 in U.
    # b+c >= 34+35 = 69 in U.
    # So a+b, a+c, b+c are all in U ⊆ S.
    # So L={33..64} is INVALID.
    
    # We need L such that pair sums are NOT all in S.
    # If pair sums are in {33..64}, they are not in S (if not in L).
    # So we want L to be a set where pair sums avoid L and U.
    # This is like a "sum-free" set with respect to the gap.
    
    # Let's try a random search for L.
    # Start with L = {17..32} (size 16).
    # Try to add elements from {1..16} ∪ {33..64}.
    
    def is_valid_L(L):
        L_set = set(L)
        for a, b, c in combinations(L, 3):
            s1, s2, s3 = a+b, a+c, b+c
            if s1 in L_set or s2 in L_set or s3 in L_set:
                return False
            if s1 >= 65 and s2 >= 65 and s3 >= 65:
                return False
            if s1 >= 65 and s2 >= 65:
                if s3 in L_set: return False
            if s1 >= 65 and s3 >= 65:
                if s2 in L_set: return False
            if s2 >= 65 and s3 >= 65:
                if s1 in L_set: return False
            # If one sum is in gap (33..64) and not in L, it's safe.
            # If one sum is in L, it's bad.
            # If one sum is in U (>=65), it's in S.
            # So the condition is: NOT (s1 in S AND s2 in S AND s3 in S)
            # S = L ∪ U.
            # s in S iff s in L OR s >= 65.
            in_S = lambda x: (x in L_set) or (x >= 65)
            if in_S(s1) and in_S(s2) and in_S(s3):
                return False
        return True

    # Check L={17..32}
    L_base = list(range(17, 33))
    assert is_valid_L(L_base)
    
    # Try to add elements
    candidates = [x for x in range(1, 65) if x not in L_base]
    random.seed(42)
    
    best_L = L_base[:]
    best_size = len(L_base)
    
    # Local search: try to add one element at a time
    changed = True
    while changed and time.time() < deadline:
        changed = False
        random.shuffle(candidates)
        for x in candidates:
            if x in best_L:
                continue
            new_L = best_L + [x]
            if is_valid_L(new_L):
                best_L = new_L
                best_size = len(best_L)
                changed = True
                candidates.remove(x)
    
    # Also try swapping
    for _ in range(1000):
        if time.time() > deadline:
            break
        if not best_L:
            break
        i = random.randint(0, len(best_L)-1)
        j = random.choice([x for x in candidates if x not in best_L]) if candidates else None
        if j is None:
            continue
        new_L = best_L[:]
        new_L[i] = j
        if is_valid_L(new_L):
            best_L = new_L
            if best_L[i] in candidates:
                candidates.remove(best_L[i])
            if j not in candidates:
                candidates.append(j)
    
    S = list(set(range(65, 129)) | set(best_L))
    S.sort()
    
    # Final verification
    S_set = set(S)
    for a, b, c in combinations(S, 3):
        if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
            # Fallback to base solution
            return [int(x) for x in list(range(65, 129)) + list(range(17, 33))]
            
    return [int(x) for x in S]

# EVOLVE-BLOCK-END
