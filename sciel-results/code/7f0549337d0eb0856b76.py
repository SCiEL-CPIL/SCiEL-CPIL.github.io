import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 180
    
    # Precompute all forbidden triples: 1/a = 1/b + 1/c => a = bc/(b+c)
    # with a < b < c (distinct, and a is the smallest since 1/a > 1/b + 1/c)
    # Actually 1/a = 1/b + 1/c means a < b and a < c.
    # Let's enumerate all triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    triples = []
    # For each pair (b, c) with b < c, check if a = bc/(b+c) is integer and a < b
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a < b:
                    triples.append((a, b, c))
    
    # Build adjacency: for each element, which triples it's in
    # We want max subset with no complete triple (independent set in 3-uniform hypergraph)
    
    # Strategy: 
    # 1. Start with all numbers 100-180 (81 elements, likely no triples among them)
    # 2. Try to add smaller numbers
    # 3. Use local search to improve
    
    # Check: are there any triples entirely within [100, 180]?
    # a >= 100, b > a, c > b, 1/a = 1/b + 1/c
    # Since b, c >= 101, 1/b + 1/c <= 2/101 < 1/100, so no such triple exists.
    # Great! So [100, 180] is a valid independent set of size 81.
    
    # What about [90, 180]? Check if any triple is entirely within [90, 180]
    # a >= 90, b >= 91, c >= 92
    # 1/90 = 1/b + 1/c, so b,c > 90
    # a = bc/(b+c). If a=90, bc = 90(b+c), bc - 90b - 90c = 0, (b-90)(c-90) = 8100
    # b-90 and c-90 are factors of 8100 with b > 90, c > b
    # 8100 = 2^2 * 3^4 * 5^2
    # Factors: 1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 25, 30, 36, 45, 50, 54, 60, 75, 90, ...
    # b = 90 + f, c = 90 + 8100/f, need b < c so f < sqrt(8100) = 90
    # f=1: b=91, c=90+8100=8190 > 180
    # f=2: b=92, c=90+4050=4140 > 180
    # ... all c will be > 180 since 8100/f >= 8100/90 = 90, so c >= 180
    # f=90: b=180, c=180, but b < c required
    # So no triple entirely in [90, 180] with a=90.
    # For a=91: bc/(b+c)=91, bc=91b+91c, (b-91)(c-91)=91^2=8281=7*11*103... 
    # This is getting complex. Let me just check computationally.
    
    # Build conflict graph: for each pair of elements, how many triples do they share?
    # Actually, let's use a different approach.
    
    # Precompute for each element x, the set of triples containing x
    triples_by_elem = {i: [] for i in range(1, N+1)}
    for idx, (a, b, c) in enumerate(triples):
        triples_by_elem[a].append(idx)
        triples_by_elem[b].append(idx)
        triples_by_elem[c].append(idx)
    
    # Greedy construction: start empty, add elements in some order
    # or start with [100,180] and try to add more
    
    def is_valid_set(s):
        """Check if set s has no forbidden triple."""
        s_set = set(s)
        for (a, b, c) in triples:
            if a in s_set and b in s_set and c in s_set:
                return False
        return True
    
    def can_add(s_set, candidate, triples, triples_by_elem):
        """Check if adding candidate to s_set creates a forbidden triple."""
        for tidx in triples_by_elem[candidate]:
            (a, b, c) = triples[tidx]
            others = [a, b, c]
            others.remove(candidate)
            if all(o in s_set for o in others):
                return False
        return True
    
    # Strategy 1: Start with [100, 180], greedily add from 99 down to 1
    best_set = set(range(100, N+1))
    
    # Try multiple orders
    for attempt in range(500):
        if time.time() > deadline - 5.0:
            break
        
        if attempt == 0:
            order = list(range(99, 0, -1))
        else:
            order = list(range(99, 0, -1))
            random.shuffle(order)
        
        cur = set(range(100, N+1))
        
        for cand in order:
            if time.time() > deadline - 3.0:
                break
            if can_add(cur, cand, triples, triples_by_elem):
                cur.add(cand)
        
        if len(cur) > len(best_set):
            best_set = set(cur)
    
    # Local search: try to swap elements to improve
    # For each element not in best_set, try adding it and removing one element that causes conflict
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        cur = set(best_set)
        for cand in range(1, 100):
            if time.time() > deadline - 3.0:
                break
            if cand in cur:
                continue
            # Find all triples that would be completed by adding cand
            blocking_triples = []
            for tidx in triples_by_elem[cand]:
                (a, b, c) = triples[tidx]
                others = [a, b, c]
                others.remove(cand)
                if all(o in cur for o in others):
                    blocking_triples.append((a, b, c))
            
            if not blocking_triples:
                # Can add without removing
                cur.add(cand)
                improved = True
                continue
            
            # Try removing one element from the union of blocking triples
            removable = set()
            for (a, b, c) in blocking_triples:
                for x in (a, b, c):
                    if x != cand:
                        removable.add(x)
            
            for rem in removable:
                # Check if removing rem and adding cand results in valid set
                test_set = cur - {rem} | {cand}
                if can_add(set(test_set - {cand}), cand, triples, triples_by_elem):
                    # Also need to check no other triples are violated
                    valid = True
                    test_set_set = set(test_set)
                    for (a, b, c) in triples:
                        if a in test_set_set and b in test_set_set and c in test_set_set:
                            valid = False
                            break
                    if valid:
                        best_set = set(test_set)
                        cur = set(test_set)
                        improved = True
                        break
    
    return list(best_set)
