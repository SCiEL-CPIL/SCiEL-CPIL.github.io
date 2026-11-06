"""Strategy: Greedy + local search for maximum set avoiding reciprocal Schur triples."""
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 180
    numbers = list(range(1, N + 1))
    
    # Precompute all forbidden triples {a, b, c} with a, b, c distinct
    # 1/a = 1/b + 1/c  <=>  a = bc / (b+c)
    # For each pair (b, c) with b < c, compute a = bc/(b+c).
    # If a is integer and distinct from b, c, and in [1, N], form triple.
    
    triples = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    triple = tuple(sorted((a, b, c)))
                    triples.add(triple)
    
    # Also need to check: for a given 'a', which pairs (b, c) satisfy 1/a = 1/b + 1/c?
    # 1/b + 1/c = 1/a => (b+c)/(bc) = 1/a => a(b+c) = bc => bc - ab - ac = 0 => (b-a)(c-a) = a^2
    # So for each a, the pairs (b, c) with b, c > a (wlog b < c) correspond to factor pairs of a^2.
    # Let b = a + d1, c = a + d2 where d1 * d2 = a^2, d1 < d2.
    
    # Build conflict graph: for each number, list of numbers it conflicts with (in the sense that they form a triple with some third element)
    # Actually, for greedy: maintain a set S. When considering adding x, check if there exist y, z in S such that {x, y, z} is a triple.
    
    # Precompute for each x, the set of pairs (y, z) such that {x, y, z} is a triple.
    # Or better: for each x, precompute the set of "conflict pairs" - but that's expensive.
    
    # Alternative: For each x, precompute the set of y such that there exists z with {x, y, z} triple.
    # Then when checking if x can be added to S, check if any y in S has a conflict with x (i.e., exists z in S such that {x, y, z} is triple).
    
    # Even better: Precompute for each pair (x, y), the set of z such that {x, y, z} is a triple.
    # Then when adding x to S, for each y in S, check if there's a z in S such that {x, y, z} is triple.
    
    # Let's precompute: triple_lookup[(x, y)] = set of z such that {x, y, z} is a triple (with x != y != z != x)
    triple_lookup = {}
    for (a, b, c) in triples:
        for pair in [(a, b), (a, c), (b, c)]:
            x, y = pair
            if x > y:
                x, y = y, x
            if (x, y) not in triple_lookup:
                triple_lookup[(x, y)] = set()
            triple_lookup[(x, y)].add(c if pair == (a, b) else (b if pair == (a, c) else a))
    
    # Greedy: start from large numbers (they can only be 'a' in a triple if they are the smallest, but actually a = bc/(b+c) < min(b,c), so a is always smaller than b and c.
    # So numbers > 90 can never be 'a' in a triple with both b,c in [1,180]... wait, a < b and a < c, so if a > 90, then b > 90 and c > 90, so b+c > 180, bc/(b+c) < max(b,c). 
    # Actually a = bc/(b+c). If b, c >= 91, then a = bc/(b+c). For b=c=91, a = 91^2/182 = 91/2 = 45.5, not integer. 
    # But a could be in [1, 90]. So large numbers can be b or c in a triple where a is small.
    
    # So the greedy approach: add numbers in descending order. When considering x, check if adding it creates a triple with two existing elements.
    
    S = set()
    for x in range(N, 0, -1):
        if time.time() > deadline - 5:
            break
        # Check if x can be added: for each y in S, check if there exists z in S such that {x, y, z} is a triple.
        can_add = True
        for y in S:
            if y == x:
                continue
            key = (min(x, y), max(x, y))
            if key in triple_lookup:
                for z in triple_lookup[key]:
                    if z in S and z != x and z != y:
                        can_add = False
                        break
            if not can_add:
                break
        if can_add:
            S.add(x)
    
    # Local search: try to swap elements to improve
    # Try to remove an element and add a different one
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        current_size = len(S)
        for x in sorted(S):
            if time.time() > deadline - 2:
                break
            # Try removing x
            S_test = S - {x}
            # Try adding each y not in S
            for y in range(1, N + 1):
                if y in S_test:
                    continue
                if time.time() > deadline - 2:
                    break
                can_add = True
                for z in S_test:
                    if z == y:
                        continue
                    key = (min(y, z), max(y, z))
                    if key in triple_lookup:
                        for w in triple_lookup[key]:
                            if w in S_test and w != y and w != z:
                                can_add = False
                                break
                    if not can_add:
                        break
                if can_add and len(S_test) + 1 > current_size:
                    S = S_test | {y}
                    improved = True
                    break
    
    return list(S)
