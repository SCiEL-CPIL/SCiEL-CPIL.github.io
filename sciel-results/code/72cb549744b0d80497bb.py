import time
from math import gcd
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 75  # safety margin before 90s kill

    N = 180
    # Precompute all reciprocal Schur triples (a, b, c) with 1/a = 1/b + 1/c, a,b,c distinct
    # 1/a = 1/b + 1/c  =>  c = a*b / (b + a - a) ... let me derive:
    # 1/a = 1/b + 1/c = (b+c)/(bc)
    # => bc = a(b+c)
    # => bc - ab - ac = 0
    # => bc - ab - ac + a^2 = a^2
    # => (b-a)(c-a) = a^2
    # So for each a, we need factor pairs (d, e) of a^2 with d*e = a^2, d != e (for distinct), 
    # then b = a + d, c = a + e (or b = a + e, c = a + d), with b,c in range.
    
    triples = []  # list of frozensets or tuples (a, b, c)
    for a in range(1, N + 1):
        a2 = a * a
        # find all divisors d of a^2
        for d in range(1, a2 + 1):
            if a2 % d == 0:
                e = a2 // d
                if d == e:
                    continue  # would give b = c = a + d, not distinct b,c
                b = a + d
                c = a + e
                if b <= N and c <= N and b != c and b != a and c != a:
                    # ensure distinct a, b, c
                    if a != b and a != c and b != c:
                        triples.append((a, b, c))
                # also consider swapping (b, c) but same triple essentially
                # b = a + e, c = a + d would be the same set
    # Deduplicate triples (as frozensets)
    seen = set()
    unique_triples = []
    for t in triples:
        fs = frozenset(t)
        if len(fs) == 3:  # all distinct
            if fs not in seen:
                seen.add(fs)
                unique_triples.append(tuple(sorted(fs)))
    
    # Now we have all forbidden triples. We want max subset of [1,N] avoiding all.
    # This is maximum independent set in a 3-uniform hypergraph - NP-hard.
    # Use greedy + local search / simulated annealing.
    
    # Greedy: start with all, remove elements that are in most triples (worst-first),
    # but that removes the "hub" elements. Actually we want to keep as many as possible.
    # Better greedy: iteratively remove the element that participates in the most
    # currently-violated triples.
    
    # Start with full set
    S = set(range(1, N + 1))
    
    # Build incidence: element -> list of triple indices containing it
    elem_triples = {i: [] for i in range(1, N + 1)}
    for idx, t in enumerate(unique_triples):
        for x in t:
            if x in elem_triples:
                elem_triples[x].append(idx)
    
    # Violation count per element
    def count_violations(S):
        """For each element in S, count how many triples have all 3 elements in S."""
        in_set = set(S)
        viol = {i: 0 for i in in_set}
        for t in unique_triples:
            if all(x in in_set for x in t):
                for x in t:
                    viol[x] += 1
        return viol
    
    # Greedy removal
    changed = True
    while changed:
        changed = False
        if time.time() > DEADLINE:
            break
        viol = count_violations(S)
        # Find element with max violations
        max_v = 0
        worst = None
        for x in S:
            v = viol.get(x, 0)
            if v > max_v:
                max_v = v
                worst = x
        if worst is not None and max_v > 0:
            S.discard(worst)
            changed = True
    
    base_size = len(S)
    
    # Now try to add back elements (local search / hill climbing)
    # Try adding each excluded element, then greedily fix violations.
    
    excluded = [i for i in range(1, N + 1) if i not in S]
    best_S = set(S)
    best_size = len(S)
    
    # Hill climbing: try adding an excluded element, then remove one element from the 
    # newly violated triples to restore validity.
    def is_valid(S):
        in_set = set(S)
        for t in unique_triples:
            if all(x in in_set for x in t):
                return False
        return True
    
    def fix_by_removal(S):
        """Remove minimal elements to make S valid. Greedy: remove element in most violated triples."""
        changed = True
        while changed:
            changed = False
            in_set = set(S)
            viol = {i: 0 for i in in_set}
            for t in unique_triples:
                if all(x in in_set for x in t):
                    for x in t:
                        viol[x] += 1
            max_v = 0
            worst = None
            for x in in_set:
                v = viol.get(x, 0)
                if v > max_v:
                    max_v = v
                    worst = x
            if worst is not None and max_v > 0:
                S.discard(worst)
                changed = True
        return S
    
    # Try adding back each excluded element
    for x in excluded:
        if time.time() > DEADLINE:
            break
        trial = set(best_S)
        trial.add(x)
        fixed = fix_by_removal(trial)
        if len(fixed) > best_size:
            best_S = fixed
            best_size = len(fixed)
    
    # Second pass of hill climbing
    for _ in range(20):
        if time.time() > DEADLINE:
            break
        excluded = [i for i in range(1, N + 1) if i not in best_S]
        improved = False
        for x in excluded:
            if time.time() > DEADLINE:
                break
            trial = set(best_S)
            trial.add(x)
            fixed = fix_by_removal(trial)
            if len(fixed) > best_size:
                best_S = fixed
                best_size = len(fixed)
                improved = True
        if not improved:
            break
    
    # Simulated annealing / random restarts if time permits
    import random
    rng = random.Random(42)
    for iteration in range(100):
        if time.time() > DEADLINE:
            break
        # Random local search: pick random element to add, fix
        if best_size >= N:
            break
        excluded = [i for i in range(1, N + 1) if i not in best_S]
        if not excluded:
            break
        x = rng.choice(excluded)
        trial = set(best_S)
        trial.add(x)
        fixed = fix_by_removal(trial)
        if len(fixed) > best_size:
            best_S = fixed
            best_size = len(fixed)
    
    # Verify final result
    final = sorted(best_S)
    if not final:
        # Fallback: return a known valid set
        final = list(range(73, 180))
    
    # Double-check validity
    in_final = set(final)
    for t in unique_triples:
        if all(x in in_final for x in t):
            # Remove one element to fix
            final.remove(t[0])
            in_final.discard(t[0])
    
    if not final:
        final = [1]
    
    return final
