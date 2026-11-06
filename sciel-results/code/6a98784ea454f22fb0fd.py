import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (90s limit)
    
    N = 180
    
    # Precompute forbidden triples: (a, b, c) where a < b < c and 1/a = 1/b + 1/c
    # (b-a)(c-a) = a^2
    # For each a, find factor pairs (d, e) of a^2 with d*e = a^2, d <= e
    # Then b = a + d, c = a + e, need b > a, c > a, b < c (so d < e, i.e., d != e unless we handle)
    # Actually b and c just need to be distinct and > a, and we want a < b < c for the triple
    # b = a + d, c = a + e where d * e = a^2, d < e ensures b < c
    # If d == e, then b == c, not three distinct, so skip
    
    # Build conflict structure: for each element x, which pairs (y,z) with y<z, y,z != x, x != y, x != z
    # such that {x,y,z} forms a forbidden triple
    # Better: for each a, the set of (b,c) pairs. For greedy addition, when considering adding element x,
    # check if x conflicts with any two existing elements.
    
    # Let's precompute: for each a in [1,N], list of (b,c) with a<b<c, (b-a)(c-a)=a^2
    forbidden_pairs_for_a = {}  # a -> list of (b, c)
    for a in range(1, N + 1):
        a2 = a * a
        pairs = []
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d < e:  # ensures b < c and b != c
                    b = a + d
                    c = a + e
                    if b <= N and c <= N:
                        pairs.append((b, c))
            d += 1
        forbidden_pairs_for_a[a] = pairs
    
    # Also need: for each element x, which (a, other) pairs involve x
    # When we have a set S and want to add x, we need to check:
    # 1. Is there a in S such that x is b or c in some forbidden triple (a, x, y) or (a, y, x) with y in S?
    # 2. Is there a in S and y in S such that x is the 'a' in (x, a, y) or (x, y, a)?
    
    # For efficiency, let's build:
    # conflicts_as_a[x] = list of (y, z) such that (x, y, z) is forbidden (x < y < z)
    # conflicts_as_bc[x] = list of (a, other) such that (a, x, other) or (a, other, x) is forbidden with a < min(x,other)
    
    # Actually, simpler: for each x, precompute all pairs (y, z) with y != z, y != x, z != x
    # such that {x, y, z} is a forbidden triple. Then when adding x to set S, check if any such pair is subset of S.
    
    # But that's expensive. Let's use the structure:
    # A forbidden triple has a unique smallest element a. So:
    # - If x is the smallest (a=x): check if any (b,c) in forbidden_pairs_for_a[x] has both b,c in S
    # - If x is the middle (b=x): for each a < x, check if there's c > x with (a,x,c) forbidden and c in S
    # - If x is the largest (c=x): for each a < x, check if there's b with a < b < x, (a,b,x) forbidden and b in S
    
    # Let's precompute for each x:
    # x_as_smallest: forbidden_pairs_for_a[x]
    # x_as_middle: for each a < x, find c such that (x-a)(c-a) = a^2, i.e., c = a + a^2/(x-a), need (x-a) | a^2
    # x_as_largest: for each a < x, find b such that (b-a)(x-a) = a^2, i.e., b = a + a^2/(x-a), need (x-a) | a^2
    
    # Let me just precompute all forbidden triples and build adjacency
    
    # All forbidden triples (a, b, c) with a < b < c
    all_triples = []
    for a in range(1, N + 1):
        for (b, c) in forbidden_pairs_for_a[a]:
            all_triples.append((a, b, c))
    
    # For each element x, list of triples it participates in
    triples_for_element = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in all_triples:
        triples_for_element[a].append((b, c))  # x=a, need b and c in S
        triples_for_element[b].append((a, c))  # x=b, need a and c in S
        triples_for_element[c].append((a, b))  # x=c, need a and b in S
    
    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a forbidden triple."""
        for (y, z) in triples_for_element[x]:
            if y in S_set and z in S_set:
                return False
        return True
    
    # Strategy: try different initial ranges and greedy orders
    best_set = set(range(91, 181))  # seed
    best_size = len(best_set)
    
    # Try greedy from high to low with various starting points
    for start_hi in range(100, 181, 5):
        if time.time() > deadline - 10:
            break
        S = set(range(start_hi, 181))
        S_set = S
        # Add elements from start_hi-1 down to 1
        for x in range(start_hi - 1, 0, -1):
            if time.time() > deadline - 5:
                break
            if can_add(x, S_set):
                S_set.add(x)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = S_set.copy()
    
    # Try greedy from low to high with various starting points
    for start_lo in range(1, 50, 3):
        if time.time() > deadline - 10:
            break
        S_set = set(range(start_lo, 181))
        # Try adding elements below start_lo
        for x in range(start_lo - 1, 0, -1):
            if time.time() > deadline - 5:
                break
            if can_add(x, S_set):
                S_set.add(x)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = S_set.copy()
    
    # Try full greedy: start empty, add from 180 down to 1
    if time.time() < deadline - 15:
        S_set = set()
        for x in range(180, 0, -1):
            if can_add(x, S_set):
                S_set.add(x)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = S_set.copy()
    
    # Try full greedy: start empty, add from 1 to 180
    if time.time() < deadline - 15:
        S_set = set()
        for x in range(1, 181):
            if can_add(x, S_set):
                S_set.add(x)
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = S_set.copy()
    
    # Local improvement: try to add any remaining element
    if time.time() < deadline - 5:
        for x in range(1, 181):
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
    
    # One more pass of local improvement
    if time.time() < deadline - 3:
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for x in range(1, 181):
                if x not in best_set and can_add(x, best_set):
                    best_set.add(x)
                    changed = True
    
    return list(best_set)

