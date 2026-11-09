"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s
    
    n = 128
    
    # Precompute conflict masks for fast validity checking
    # For each element x, conflict_mask[x] is a bitmask of elements y such that
    # there exists some z with {x, y, z} forming a forbidden configuration
    # More precisely: if we have selected set S, adding x is invalid if there exist
    # distinct a, b in S such that a+b in S, a+c in S, b+c in S where c=x
    # This is complex, so let's use a different approach.
    
    # Better approach: Precompute for each pair (a,b) with a+b <= 128, 
    # the set of c values that would create a forbidden triple with a,b
    # A triple (a,b,c) is forbidden if a+b, a+c, b+c are all in S
    # So for fixed a,b in S with a+b in S, any c in S with a+c in S and b+c in S creates a violation
    
    # Let's use bit-parallel approach:
    # Represent S as a 128-bit integer (bit i set means i+1 is in S)
    # For each element x, precompute:
    #   partners[x] = bitmask of elements y such that x+y <= 128
    #   For validity: x can be added to S iff for all pairs (a,b) in S with a+b in S,
    #   x is not in the "forbidden set" for that pair
    
    # Actually, let me think more carefully.
    # A forbidden triple is (a,b,c) distinct such that a+b, a+c, b+c all in S.
    # When we add x to S, the only new forbidden triples are those involving x.
    # So we need: no distinct a,b in S such that x+a in S, x+b in S, and a+b in S.
    # (where a,b are distinct from x and from each other)
    
    # Precompute for each x:
    #   pair_sums[x] = set of (a,b) pairs where a<b, a,b != x, x+a<=128, x+b<=128
    #   But we need a+b in S too.
    
    # Let's just do efficient checking with precomputed data structures.
    
    # Precompute: for each x in 1..128, the list of pairs (a,b) with a<b, a!=x, b!=x,
    # such that x+a <= 128 and x+b <= 128 and a+b <= 128
    # When checking if x can be added to S:
    #   For each (a,b) in pairs_for_x:
    #     if a in S and b in S and (x+a) in S and (x+b) in S and (a+b) in S:
    #       return False (cannot add x)
    
    # This is still O(k^2) per check where k = len(S), but with small constants.
    
    # Let me use a different strategy: start with a large valid set and try to grow it.
    # Known: {65, 66, ..., 128} has size 64. Let's see if we can do better.
    
    # Actually, let's try a smarter construction.
    # If we take all numbers > 64, their pair sums are > 128, so they can't be in S.
    # So {65,...,128} is valid with size 64.
    
    # Can we add some smaller numbers?
    # If we add x <= 64, then for any a,b in {65,...,128}, x+a and x+b are in {65+x,...,128+x}.
    # For x+a to be in S, we need x+a <= 128, i.e., a <= 128-x.
    # And a+b must be in S, but a,b >= 65 so a+b >= 130 > 128, so a+b is never in S.
    # Wait, that means {65,...,128} is valid because no pair sum of two elements is in the set.
    
    # So we can potentially add many small elements!
    # Let's try: S = {1, 2, ..., k} union {65, 66, ..., 128}
    # For this to be valid, we need no forbidden triple.
    # A forbidden triple (a,b,c) with a+b, a+c, b+c in S.
    # Case 1: a,b,c all in {1,...,k}. Then a+b, a+c, b+c could be in {1,...,k} or in {65,...,128}.
    #   If a+b <= k, a+c <= k, b+c <= k, then we need to avoid this.
    #   If a+b > k but a+b in {65,...,128}, same for a+c, b+c.
    # Case 2: Two in {1,...,k}, one in {65,...,128}. Say a,b in {1,...,k}, c in {65,...,128}.
    #   a+b could be in {1,...,k} or {65,...,128}.
    #   a+c in {66,...,k+128}, b+c in {66,...,k+128}.
    #   For a+c and b+c to be in S, they need to be in {65,...,128} (since > k for reasonable k).
    #   So a+c >= 65 and a+c <= 128, similarly for b+c.
    #   And a+b must be in S.
    # Case 3: One in {1,...,k}, two in {65,...,128}. Say a in {1,...,k}, b,c in {65,...,128}.
    #   b+c > 128, so b+c not in S. No forbidden triple of this type.
    # Case 4: All in {65,...,128}. Pair sums > 128, not in S. No forbidden triple.
    
    # So the main concern is Case 1 and Case 2.
    # Let's just do a greedy construction with efficient checking.
    
    def check_can_add(x, S_set):
        """Check if x can be added to S_set without creating a forbidden triple."""
        if x in S_set:
            return True
        # Check all pairs (a,b) in S_set with a < b, a != x, b != x
        # such that x+a, x+b, a+b are all in S_set
        elems = sorted(S_set)
        k = len(elems)
        for i in range(k):
            a = elems[i]
            if a == x:
                continue
            xa = x + a
            if xa > n or xa not in S_set:
                continue
            for j in range(i+1, k):
                b = elems[j]
                if b == x:
                    continue
                xb = x + b
                if xb > n or xb not in S_set:
                    continue
                ab = a + b
                if ab <= n and ab in S_set:
                    return False
        return True
    
    # Phase 1: Start with upper half, greedily add from lower half
    S = set(range(65, n + 1))
    
    # Try adding elements from 1 to 64 in increasing order
    for x in range(1, 65):
        if time.time() > deadline:
            break
        if check_can_add(x, S):
            S.add(x)
    
    best_S = set(S)
    best_size = len(S)
    
    # Phase 2: Local search - try to add more elements
    remaining = [x for x in range(1, n + 1) if x not in S]
    
    # Try multiple times to add elements
    rng = random.Random(42)
    
    for iteration in range(100):
        if time.time() > deadline:
            break
        if not remaining:
            break
        
        # Try each remaining element
        for x in remaining[:]:
            if time.time() > deadline:
                break
            if check_can_add(x, S):
                S.add(x)
                remaining.remove(x)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
    
    # Phase 3: Try swaps to improve
    cur_S = set(best_S)
    cur_size = len(cur_S)
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        improved = False
        in_S = list(cur_S)
        not_in_S = [x for x in range(1, n + 1) if x not in cur_S]
        
        for x in in_S:
            if time.time() > deadline:
                break
            for y in not_in_S:
                if time.time() > deadline:
                    break
                new_S = (cur_S - {x}) | {y}
                if check_can_add(y, new_S - {y}) and len(new_S) == cur_size:
                    # Check if this swap helps us add more later
                    # For now, just accept if it doesn't reduce size
                    pass
        
        # Try adding after removing one element
        for x in in_S:
            if time.time() > deadline:
                break
            temp_S = cur_S - {x}
            for y in not_in_S:
                if time.time() > deadline:
                    break
                if check_can_add(y, temp_S):
                    new_S = temp_S | {y}
                    if len(new_S) > cur_size:
                        cur_S = new_S
                        cur_size = len(cur_S)
                        if cur_size > best_size:
                            best_S = set(cur_S)
                            best_size = cur_size
                        improved = True
                        break
            if improved:
                break
    
    # Final greedy pass
    S_final = set(best_S)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x in S_final:
            continue
        if check_can_add(x, S_final):
            S_final.add(x)
    
    result = sorted(S_final)
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
