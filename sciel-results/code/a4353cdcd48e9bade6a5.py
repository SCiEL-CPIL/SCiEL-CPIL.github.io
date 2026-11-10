import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 180
    domain = set(range(1, N + 1))

    # Precompute all forbidden triples (a, b, c) with a < b < c, a, b, c distinct,
    # satisfying 1/a = 1/b + 1/c
    # This means b*c = a*(b+c), i.e., b*c - a*b - a*c = 0, i.e., (b-a)(c-a) = a^2
    # For efficiency, precompute for each pair (b,c) with b<c, whether there exists a
    # such that 1/a = 1/b + 1/c, i.e., a = bc/(b+c).
    # We need a to be an integer in [1,180] and distinct from b and c.

    # Build a map: for each element a, which pairs (b,c) are forbidden with a?
    # Actually, for checking if adding x to a set S creates a violation:
    # x can be the "a" role: need 1/x = 1/b + 1/c for b,c in S
    # x can be the "b" or "c" role: need 1/a = 1/x + 1/c for a,c in S (with x as b or c)
    
    # Precompute all valid triples (a, b, c) with a < b < c, a,b,c distinct, 1/a = 1/b + 1/c
    # Use the relation: (b-a)(c-a) = a^2
    # For each a from 1 to 180, find all factor pairs of a^2 to get b-a and c-a
    # where b-a > 0, c-a > 0, and b != c, and b,c in [1,180]
    
    forbidden_triples = []
    # For each a, (b-a)(c-a) = a^2
    # Let d1 = b-a, d2 = c-a, then d1*d2 = a^2, d1 < d2 (since b < c means d1 < d2)
    # b = a + d1, c = a + d2
    # Need b != c (d1 != d2, guaranteed since d1 < d2), b,c in [1,180]
    # Also need a, b, c all distinct - since d1,d2 > 0, b>a and c>a, so a is distinct from b,c
    # b and c are distinct since d1 < d2
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find all divisors d1 of a2 such that d1 < sqrt(a2), i.e., d1*d2 = a2 with d1 < d2
        # d1 from 1 to sqrt(a2)
        limit = int(a2 ** 0.5)
        for d1 in range(1, limit + 1):
            if a2 % d1 != 0:
                continue
            d2 = a2 // d1
            if d1 == d2:
                continue
            b = a + d1
            c = a + d2
            if 1 <= b <= N and 1 <= c <= N and b < c:
                forbidden_triples.append((a, b, c))
            elif 1 <= b <= N and 1 <= c <= N and c < b:
                # This shouldn't happen since d1 < d2 means b < c
                pass

    # Also need to consider cases where the "a" in 1/a = 1/b + 1/c might not be the smallest
    # Wait, in the equation 1/a = 1/b + 1/c, we have 1/a < 1/b and 1/a < 1/c, so a > b and a > c.
    # So a is always the largest element! Let me re-derive.
    # 1/a = 1/b + 1/c => a = bc/(b+c). Since b,c > 0, a < b and a < c.
    # So a is the SMALLEST element. My derivation above was correct: a < b and a < c.
    
    # Now build a data structure for fast violation checking
    # For a given set S, to check if adding x creates a violation:
    # Case 1: x is the smallest (a role): need b,c in S with 1/x = 1/b + 1/c
    # Case 2: x is the middle or largest: need a in S and another element y in S 
    #          such that 1/a = 1/x + 1/y or 1/a = 1/y + 1/x
    
    # Precompute for each x, the list of pairs (b,c) such that 1/x = 1/b + 1/c
    # i.e., b*c = x*(b+c), (b-x)(c-x) = x^2
    pairs_for_a = {}
    for a in range(1, N + 1):
        a2 = a * a
        limit = int(a2 ** 0.5)
        result = []
        for d1 in range(1, limit + 1):
            if a2 % d1 != 0:
                continue
            d2 = a2 // d1
            if d1 == d2:
                continue
            b = a + d1
            c = a + d2
            if 1 <= b <= N and 1 <= c <= N:
                result.append((b, c))
        pairs_for_a[a] = result
    
    # For each x (as b or c), precompute the set of (a, other) pairs
    # where 1/a = 1/x + 1/other, i.e., a = x*other/(x+other)
    # For a given x, iterate over possible 'a' values... this is less efficient.
    # Alternative: for each triple (a,b,c), store that {b,c} conflicts with a, and {a,c} conflicts with b, and {a,b} conflicts with c
    
    # Let's use a different approach: for each element x, precompute all y such that 
    # x and y together force a third element.
    # If x and y are in S, and x < y, then:
    # - If 1/z = 1/x + 1/y has integer z in [1,180], z != x,y, then z is forbidden
    # - If x is the "a" role: 1/x = 1/y + 1/w => w = x*y/(y-x) (need y > x)
    # - If y is the "a" role: 1/y = 1/x + 1/w => w = x*y/(x-y) (negative, skip)
    
    # Let me just precompute for each unordered pair {x,y}, the set of z that would form a violation
    conflict_map = {}
    for a in range(1, N + 1):
        for b, c in pairs_for_a[a]:
            # Triple (a, b, c) with a < b, a < c
            # Conflict pairs: (a,b)->c, (a,c)->b, (b,c)->a
            key = (min(a,b), max(a,b))
            if key not in conflict_map:
                conflict_map[key] = set()
            conflict_map[key].add(c)
            
            key = (min(a,c), max(a,c))
            if key not in conflict_map:
                conflict_map[key] = set()
            conflict_map[key].add(b)
            
            key = (min(b,c), max(b,c))
            if key not in conflict_map:
                conflict_map[key] = set()
            conflict_map[key].add(a)
    
    def is_valid(S):
        S = set(S)
        for x in S:
            for y in S:
                if x == y:
                    continue
                key = (min(x,y), max(x,y))
                if key in conflict_map:
                    for z in conflict_map[key]:
                        if z in S and z != x and z != y:
                            return False
        return True
    
    # Greedy approach: start with all elements, remove those causing conflicts
    # Or: greedy addition
    
    def greedy_random(seed_val):
        random.seed(seed_val)
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        S = []
        S_set = set()
        for x in elements:
            # Check if adding x to S_set creates a violation
            valid = True
            for y in S_set:
                key = (min(x,y), max(x,y))
                if key in conflict_map:
                    for z in conflict_map[key]:
                        if z in S_set and z != x and z != y:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                S.append(x)
                S_set.add(x)
        return S
    
    def local_search(S, seed_val, time_limit):
        """Try to improve S by removing and re-adding elements"""
        random.seed(seed_val)
        S = list(S)
        S_set = set(S)
        best = len(S)
        
        end_time = start_time + time_limit
        while time.time() < end_time:
            # Try removing a random element and see if we can add a different one
            if not S:
                break
            idx = random.randint(0, len(S) - 1)
            removed = S[idx]
            S.remove(removed)
            S_set.discard(removed)
            
            # Try to add elements not in S
            candidates = [x for x in range(1, N+1) if x not in S_set]
            random.shuffle(candidates)
            added = False
            for x in candidates:
                valid = True
                for y in S_set:
                    key = (min(x,y), max(x,y))
                    if key in conflict_map:
                        for z in conflict_map[key]:
                            if z in S_set and z != x and z != y:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    S.append(x)
                    S_set.add(x)
                    added = True
                    break
            
            if not added:
                # Put removed back
                S.append(removed)
                S_set.add(removed)
            
            if len(S) > best:
                best = len(S)
        
        return S
    
    # Run multiple greedy + local search
    best_result = []
    
    for trial in range(50):
        if time.time() > deadline - 5.0:
            break
        
        seed = trial * 1000 + 42
        S = greedy_random(seed)
        
        # Local search with remaining time
        remaining = deadline - time.time() - 2.0
        if remaining > 0.5:
            S = local_search(S, seed, remaining)
        
        if len(S) > len(best_result):
            best_result = S
    
    # Final validity check
    if not is_valid(best_result):
        # Fallback to simple valid set
        best_result = list(range(91, 181))
    
    return best_result
