"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    
    n = 128
    
    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a forbidden triple.
        A forbidden triple (a,b,c) distinct with a+b, a+c, b+c all in S.
        When adding x, new forbidden triples must include x.
        So check: for pairs (a,b) in S_set, if a+b in S_set, and x+a in S_set, and x+b in S_set,
        then (x,a,b) would be forbidden. Also need to check if x itself forms pairs with sums in S_set.
        """
        # For each pair (a, b) in S_set with a < b:
        #   If a+b is in S_set, then we need to ensure NOT (x+a in S_set AND x+b in S_set)
        #   because then (x, a, b) would be a forbidden triple (all distinct if x not in {a,b})
        # Also need: x+a in S_set and a+b in S_set means pair (x,a) has sum in S, so for any c in S,
        #   check x+c in S and a+c in S -> but that's already covered by the pair (a,b) check with b=c
        # Actually let's be more careful. A forbidden triple is three DISTINCT elements p,q,r in S
        # such that p+q, p+r, q+r are all in S.
        # When we add x, new forbidden triples are those that include x.
        # So for distinct a,b in S_set (a != b, a != x, b != x):
        #   Check if x+a in S_set, x+b in S_set, and a+b in S_set.
        #   If all three are in S_set, then (x,a,b) is a forbidden triple.
        # Note: x+a, x+b, a+b could equal x or other elements, but since we're checking membership in S_set
        # (which doesn't include x yet), and the sums are > max(x,a,b) typically, they won't equal x unless
        # some element is 0 (not in our range). So we just check membership.
        
        elems = list(S_set)
        k = len(elems)
        for i in range(k):
            a = elems[i]
            if a == x:
                continue
            for j in range(i+1, k):
                b = elems[j]
                if b == x:
                    continue
                ab = a + b
                if ab in S_set:
                    if (x + a) in S_set and (x + b) in S_set:
                        return False
        return True
    
    def is_valid(S_set):
        """Full validity check - verify no forbidden triple exists."""
        elems = sorted(S_set)
        k = len(elems)
        for i in range(k):
            a = elems[i]
            for j in range(i+1, k):
                b = elems[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for kk in range(j+1, k):
                    c = elems[kk]
                    if (a + c) in S_set and (b + c) in S_set:
                        return False
        return True
    
    # Strategy: Use a structured approach
    # Start with upper half, then greedily add from lower half
    # Then do systematic local search with fast validity checks
    
    # Phase 1: Greedy construction starting from top
    S = set(range(64, n + 1))
    
    # Try adding elements from 1 to 63 in various orders
    orders = [
        list(range(63, 0, -1)),
        list(range(1, 64)),
        sorted(range(1, 64), key=lambda x: x % 7),
        sorted(range(1, 64), key=lambda x: x % 5),
    ]
    
    best_S = set(S)
    best_size = len(S)
    
    for order in orders:
        if time.time() > deadline:
            break
        S_temp = set(S)
        for x in order:
            if time.time() > deadline:
                break
            if x in S_temp:
                continue
            if can_add(x, S_temp):
                S_temp = S_temp | {x}
        if len(S_temp) > best_size:
            best_S = set(S_temp)
            best_size = len(S_temp)
    
    # Phase 2: Systematic local search - try all single swaps
    cur_S = set(best_S)
    all_elements = list(range(1, n + 1))
    in_S = set(cur_S)
    not_in_S = [x for x in all_elements if x not in in_S]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each element not in S
        for x in not_in_S:
            if time.time() > deadline:
                break
            if can_add(x, cur_S):
                cur_S = cur_S | {x}
                in_S.add(x)
                not_in_S.remove(x)
                improved = True
        # Try swapping: remove one, add one
        for y in list(in_S):
            if time.time() > deadline:
                break
            for x in not_in_S:
                if time.time() > deadline:
                    break
                new_S = (cur_S - {y}) | {x}
                if is_valid(new_S):
                    cur_S = new_S
                    in_S.remove(y)
                    in_S.add(x)
                    not_in_S.remove(x)
                    not_in_S.append(y)
                    improved = True
                    break
    
    # Phase 3: Final greedy pass
    S_final = set(cur_S)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x in S_final:
            continue
        if can_add(x, S_final):
            S_final = S_final | {x}
    
    # Verify final result
    if not is_valid(S_final):
        # Fallback to known good set
        S_final = set(range(64, n + 1))
    
    result = sorted(S_final)
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
