"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 180
    elements = list(range(1, N+1))
    
    # Precompute all conflict triples: (a, b, c) where 1/a = 1/b + 1/c
    # a = k*m*n, b = k*m*(m+n), c = k*n*(m+n), gcd(m,n)=1
    # Build conflict graph: edge between x,y if they appear together in some triple
    # Actually, we need hyperedges of size 3. For max independent set, 
    # we need: no three elements form a valid triple.
    
    # Let's build the set of all valid triples
    triples = set()
    for k in range(1, 181):
        for m in range(1, 181):
            for n in range(1, 181):
                if m >= n:  # avoid duplicates, assume m < n
                    continue
                # Check gcd(m,n) = 1
                if _gcd(m, n) != 1:
                    continue
                a = k * m * n
                b = k * m * (m + n)
                c = k * n * (m + n)
                if a <= N and b <= N and c <= N:
                    # Check all distinct
                    if len({a, b, c}) == 3:
                        triples.add((min(a,b,c), min(max(a,b), max(b,c)), max(a,b,c)))
    
    # Build conflict adjacency: for each pair (x,y), check if there exists z such that (x,y,z) is a triple
    # For efficiency, build a dict: for each pair (x,y), list of z that complete a triple
    pair_conflicts = {}  # (x,y) -> set of z
    for (a, b, c) in triples:
        for (x, y, z) in [(a,b,c), (a,c,b), (b,a,c), (b,c,a), (c,a,b), (c,b,a)]:
            key = (x, y)
            if key not in pair_conflicts:
                pair_conflicts[key] = set()
            pair_conflicts[key].add(z)
    
    # Also build for each element, the set of other elements it conflicts with (appears in a triple with)
    element_conflicts = {i: set() for i in range(1, N+1)}
    for (a, b, c) in triples:
        element_conflicts[a].update([b, c])
        element_conflicts[b].update([a, c])
        element_conflicts[c].update([a, b])
    
    # Greedy construction with multiple orderings + local search
    best_set = set(range(91, 181))  # initial seed
    best_size = len(best_set)
    
    def is_valid(s):
        """Check if set s has no valid triple"""
        for (a, b, c) in triples:
            if a in s and b in s and c in s:
                return False
        return True
    
    # Verify initial
    if not is_valid(best_set):
        best_set = set()
        best_size = 0
    
    # Try many greedy orderings
    random.seed(42)
    
    def greedy_build(order):
        """Build a valid set by greedily adding elements in given order"""
        result = set()
        for x in order:
            # Check if adding x creates a conflict
            can_add = True
            for y in element_conflicts[x]:
                if y in result:
                    # Check if there exists z in result such that (x,y,z) is a triple
                    # Actually, we need to check all triples containing x
                    # For each pair (y, z) where z is in result and (x,y,z) is a triple
                    pass
            # More careful check: for each triple containing x, check if the other two are in result
            for (a, b, c) in triples:
                if x == a:
                    if b in result and c in result:
                        can_add = False
                        break
                elif x == b:
                    if a in result and c in result:
                        can_add = False
                        break
                elif x == c:
                    if a in result and b in result:
                        can_add = False
                        break
            if can_add:
                result.add(x)
        return result
    
    # Precompute for each element x, the list of triples it participates in
    element_triples = {i: [] for i in range(1, N+1)}
    for (a, b, c) in triples:
        element_triples[a].append((a, b, c))
        element_triples[b].append((a, b, c))
        element_triples[c].append((a, b, c))
    
    def greedy_build_fast(order):
        """Faster greedy using precomputed triples"""
        result = set()
        for x in order:
            can_add = True
            for (a, b, c) in element_triples[x]:
                if x == a:
                    if b in result and c in result:
                        can_add = False
                        break
                elif x == b:
                    if a in result and c in result:
                        can_add = False
                        break
                elif x == c:
                    if a in result and b in result:
                        can_add = False
                        break
            if can_add:
                result.add(x)
        return result
    
    # Try different orderings
    orderings = [
        list(range(1, N+1)),           # ascending
        list(range(N, 0, -1)),         # descending
        list(range(91, N+1)) + list(range(1, 91)),  # large first
    ]
    
    # Add random orderings
    for _ in range(50):
        if time.time() - start_time > DEADLINE:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() - start_time > DEADLINE:
            break
        s = greedy_build_fast(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search: try to add elements, or swap
    def try_add(s, x):
        """Try to add x to set s, return True if valid"""
        for (a, b, c) in element_triples[x]:
            if x == a:
                if b in s and c in s:
                    return False
            elif x == b:
                if a in s and c in s:
                    return False
            elif x == c:
                if a in s and b in s:
                    return False
        return True
    
    # Local search: repeatedly try to add random elements
    for iteration in range(200):
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for x in range(1, N+1):
            if x not in best_set and try_add(best_set, x):
                best_set.add(x)
                best_size = len(best_set)
                improved = True
        if not improved:
            break
    
    # Try swaps: remove one, add one or more
    for iteration in range(100):
        if time.time() - start_time > DEADLINE:
            break
        improved = False
        for x in list(best_set):
            test_set = best_set - {x}
            # Try to add elements not in test_set
            added = 0
            for y in range(1, N+1):
                if y not in test_set and try_add(test_set, y):
                    test_set.add(y)
                    added += 1
            if len(test_set) > len(best_set):
                best_set = test_set
                best_size = len(best_set)
                improved = True
                break
        if not improved:
            break
    
    # Final: try all single additions again
    for x in range(1, N+1):
        if x not in best_set and try_add(best_set, x):
            best_set.add(x)
            best_size = len(best_set)
    
    if time.time() - start_time > DEADLINE:
        pass  # just return what we have
    
    return list(best_set)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# EVOLVE-BLOCK-END
