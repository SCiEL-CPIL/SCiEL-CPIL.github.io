"""
Triple systems on 16 points avoiding K(2,2,2) in transversals.
Goal: maximize number of triples (3-subsets of [16]) such that there do not exist
three disjoint pairs (a1,b1), (a2,b2), (a3,b3) for which all 8 transversals
(a_i or b_i from each pair) are present as triples.

We use a greedy randomized local search with a deadline guard.
"""
import time
import random
import itertools
import sys

def _is_valid(triples_set, triples_list):
    """Check that no K(2,2,2) transversal is fully present."""
    # Precompute: for each pair of points, which triples contain both?
    # But we need to check for 3 disjoint pairs: all 8 combos present.
    # Efficient check: iterate over all ways to pick 3 disjoint pairs.
    # Number of ways: choose 6 points out of 16, then partition into 3 pairs.
    # C(16,6) = 8008, and for each 6-set, number of perfect matchings = 15.
    # So ~120k checks, each checking 8 lookups. That's ~1M lookups total.
    # But we can do better: only check when adding a triple.
    
    # For incremental checking: when adding triple T, check if it completes any K(2,2,2).
    # A K(2,2,2) is completed by T if there exist two other disjoint pairs such that
    # all 8 transversals are in the set. T is one of the 8.
    # T = {x, y, z}. For T to be part of a K(2,2,2), there must be a partition of
    # {x,y,z} into... actually T uses one element from each pair.
    # So the three pairs are: (x, x'), (y, y'), (z, z') where x', y', z' are other points.
    # All 8 triples: {x or x', y or y', z or z'} must be in the set.
    # T = {x, y, z} is one. The other 7 must also be present.
    # So for each choice of x', y', z' (distinct from x,y,z and from each other),
    # check if all 8 triples are present.
    
    # But this is expensive to do naively. Let's use a different approach.
    # Precompute for each triple which "opposite" triples would complete a K(2,2,2).
    
    # Actually, let's just do the full check on the final set for verification,
    # and use a heuristic during construction.
    
    return True  # Placeholder; actual check done separately

def _check_no_k222(triples_set):
    """Full check: no three disjoint pairs have all 8 transversals present."""
    points = set(range(16))
    # Enumerate all 3-disjoint-pair configurations
    for six in itertools.combinations(range(16), 6):
        # All perfect matchings of 6 points
        a, b, c, d, e, f = six
        matchings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in matchings:
            # Check all 8 transversals
            all_present = True
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        t = frozenset([p1[i], p2[j], p3[k]])
                        if t not in triples_set:
                            all_present = False
                            break
                    if not all_present:
                        break
                if not all_present:
                    break
            if all_present:
                return False
    return True

def _try_add(triples_set, triple):
    """Try to add triple; return True if valid."""
    t = frozenset(triple)
    if t in triples_set:
        return True  # Already present
    # Check if adding t would create a K(2,2,2)
    # t = {x, y, z}. For t to complete a K(2,2,2), we need x', y', z' such that
    # all 8 transversals of pairs (x,x'), (y,y'), (z,z') are present.
    # t is one of them (the one with x,y,z). The other 7 must be in triples_set.
    x, y, z = tuple(triple)
    others = set(range(16)) - {x, y, z}
    for xp in others:
        for yp in others - {xp}:
            for zp in others - {xp, yp}:
                # Check all 8 transversals
                # (x or xp, y or yp, z or zp)
                # We already have (x,y,z). Check the other 7:
                if (frozenset([xp, y, z]) in triples_set and
                    frozenset([x, yp, z]) in triples_set and
                    frozenset([x, y, zp]) in triples_set and
                    frozenset([xp, yp, z]) in triples_set and
                    frozenset([xp, y, zp]) in triples_set and
                    frozenset([x, yp, zp]) in triples_set and
                    frozenset([xp, yp, zp]) in triples_set):
                    return False
    triples_set.add(t)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_set = set()
    best_count = 0
    
    # Strategy: greedy random addition with occasional local search
    all_triples = [frozenset(t) for t in itertools.combinations(range(16), 3)]
    
    # Multiple restarts with different random orders
    rng = random.Random(42)
    
    for attempt in range(50):
        if time.time() > deadline - 20:
            break
        
        triples_set = set()
        remaining = list(all_triples)
        rng.shuffle(remaining)
        
        for t in remaining:
            if time.time() > deadline - 20:
                break
            if t not in triples_set:
                _try_add(triples_set, t)
        
        # Local search: try to swap out triples to add more
        if time.time() > deadline - 15:
            break
        
        # Simple local search: try removing one and adding two
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            current_list = list(triples_set)
            rng.shuffle(current_list)
            for t_remove in current_list:
                if time.time() > deadline - 15:
                    break
                # Try removing t_remove and see if we can add more
                triples_set.discard(t_remove)
                added = 0
                # Try to add some random triples
                candidates = [t for t in all_triples if t not in triples_set]
                rng.shuffle(candidates)
                for t_add in candidates[:100]:
                    if time.time() > deadline - 15:
                        break
                    if _try_add(triples_set, t_add):
                        added += 1
                if added > 1:
                    improved = True
                else:
                    triples_set.add(t_remove)
        
        if len(triples_set) > best_count:
            best_count = len(triples_set)
            best_set = set(triples_set)
    
    return [list(t) for t in best_set]
