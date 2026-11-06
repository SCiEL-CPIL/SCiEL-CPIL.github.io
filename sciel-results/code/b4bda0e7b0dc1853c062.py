"""Improved program for triple systems on 16 points without K(2,2,2)."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    deadline = time.time() + 115  # safety margin before 120s limit
    best_triples = set()
    
    # Precompute for each triple, the set of "conflict configurations"
    # A K(2,2,2) on 6 distinct points {a,b,c,d,e,f} has 8 transversals:
    # We need to check: for any 6 distinct points, if all 8 transversals are present, it's invalid.
    # The 8 transversals of K(2,2,2) with parts {a,b}, {c,d}, {e,f} are:
    # {a,c,e}, {a,c,f}, {a,d,e}, {a,d,f}, {b,c,e}, {b,c,f}, {b,d,e}, {b,d,f}
    
    # For efficiency, we'll track for each pair of points, which triples contain both.
    # Actually, let's use a different approach: 
    # For each triple t, when we add it, we need to check if it completes any K(2,2,2).
    # A new triple {a,b,c} completes a K(2,2,2) if there exist d,e,f (distinct from a,b,c)
    # such that all 8 transversals of some K(2,2,2) containing {a,b,c} are present.
    
    # Let's precompute: for each triple t and each "role" it can play in a K(2,2,2),
    # what other triples are needed?
    
    # Actually, let's use a simpler but effective approach:
    # Maintain for each 6-element subset, the count of how many of its 8 transversals are present.
    # But 6-subsets of 16 is C(16,6) = 8008, and each has 8 transversals. That's manageable.
    
    # Let's precompute all 6-subsets and their 8 transversals.
    # For a 6-subset S = {v0,v1,v2,v3,v4,v5}, the 8 transversals depend on how we partition into 3 pairs.
    # There are 15 ways to partition 6 elements into 3 pairs. For each partition, there are 8 transversals.
    # But wait, K(2,2,2) is specifically about a fixed partition into 3 parts of size 2.
    # So for each 6-subset and each of the 15 pair partitions, we have 8 transversals.
    # Total: 8008 * 15 * 8 = 960,960 transversal checks. That's a lot to track.
    
    # Alternative: For each triple, check all possible K(2,2,2) it could complete.
    # A triple {a,b,c} can be in a K(2,2,2) in different ways. Let's think about it differently.
    
    # Let's use the WalkSAT approach with a smart conflict check.
    
    def check_conflict(triples_set, t):
        """Check if adding triple t to triples_set creates a K(2,2,2)."""
        a, b, c = t
        # For each pair of other points (d,e) and (f,g), check if all 8 transversals are present.
        # But that's O(n^4) per check. Too slow.
        
        # Better: for each choice of 3 other points d,e,f, check if {a,b,c} is one of the 8 transversals
        # of a K(2,2,2) on {a,b,c,d,e,f}.
        # The 8 transversals of K(2,2,2) with parts {a,d},{b,e},{c,f} are:
        # {a,b,c}, {a,b,f}, {a,e,c}, {a,e,f}, {d,b,c}, {d,b,f}, {d,e,c}, {d,e,f}
        # So if we have {a,b,c}, the other 7 needed are: {a,b,f}, {a,e,c}, {a,e,f}, {d,b,c}, {d,b,f}, {d,e,c}, {d,e,f}
        
        # But the partition matters. {a,b,c} could be the transversal in different partitions.
        # Let's enumerate: for points d,e,f distinct from a,b,c,
        # {a,b,c} is a transversal of K(2,2,2) on {a,b,c,d,e,f} if and only if
        # it picks one from each part. The parts are a partition of {a,b,c,d,e,f} into 3 pairs.
        # For {a,b,c} to be a transversal, each of a,b,c must be in a different part.
        # So d,e,f must be paired with a,b,c respectively (in some order).
        # There are 3! = 6 ways to assign d,e,f to pairs with a,b,c.
        
        # For each permutation (d1,d2,d3) of (d,e,f), the parts are {a,d1},{b,d2},{c,d3}.
        # The 8 transversals are all combinations of picking one from each part.
        # {a,b,c} is one of them. The other 7 are:
        # {a,b,d3}, {a,d2,c}, {a,d2,d3}, {d1,b,c}, {d1,b,d3}, {d1,d2,c}, {d1,d2,d3}
        
        others = [x for x in range(n) if x not in (a, b, c)]
        for d in others:
            for e in others:
                if e == d:
                    continue
                for f in others:
                    if f == d or f == e:
                        continue
                    # Check all 6 permutations of assigning d,e,f to a,b,c
                    for perm in [(d,e,f), (d,f,e), (e,d,f), (e,f,d), (f,d,e), (f,e,d)]:
                        d1, d2, d3 = perm
                        # The 7 other transversals needed:
                        needed = [
                            (a, b, d3),
                            (a, d2, c),
                            (a, d2, d3),
                            (d1, b, c),
                            (d1, b, d3),
                            (d1, d2, c),
                            (d1, d2, d3),
                        ]
                        # Check if all 7 are in triples_set
                        all_present = True
                        for nt in needed:
                            if nt not in triples_set:
                                all_present = False
                                break
                        if all_present:
                            return True
        return False
    
    # This is still O(n^3 * 6) per check = O(13^3 * 6) ≈ 13000 per check. Too slow.
    
    # Let's use a much faster approach: precompute for each triple, the list of "witness" sets.
    # For each triple t and each potential K(2,2,2) it could complete, store the set of 7 other triples needed.
    # Then when adding t, just check if any of those 7-sets are fully present.
    
    # Actually, let's use a different strategy: local search with a fast conflict check.
    # We'll maintain a data structure that allows fast conflict detection.
    
    # For each pair (u,v), maintain the set of points w such that {u,v,w} is in the current set.
    pair_to_thirds = {}  # (u,v) -> set of w
    
    def can_add(triples_set, pair_to_thirds, t):
        """Fast check if adding t creates a K(2,2,2)."""
        a, b, c = t
        # For a K(2,2,2) to be completed by t={a,b,c}, we need 3 other points d,e,f
        # such that all 8 transversals are present.
        # Key insight: if {a,b,c} is added, it can only create a new K(2,2,2) if there exist
        # d,e,f such that the other 7 transversals are already present.
        # For fixed a,b,c, the other 7 transversals involve pairs from {a,b}x{d,e}, etc.
        
        # Let's enumerate possible d (partner of a in some part), e (partner of b), f (partner of c)
        # For each permutation, check if the 7 other triples are present.
        
        others = [x for x in range(n) if x not in (a, b, c)]
        for d in others:
            for e in others:
                if e == d:
                    continue
                for f in others:
                    if f == d or f == e:
                        continue
                    # Check permutation (a,d), (b,e), (c,f)
                    # Other 7: {a,b,f}, {a,e,c}, {a,e,f}, {d,b,c}, {d,b,f}, {d,e,c}, {d,e,f}
                    if (all(x != y for x, y in [(a,b,f),(a,e,c),(a,e,f),(d,b,c),(d,b,f),(d,e,c),(d,e,f)])) and \
                       (a,b,f) in triples_set and (a,e,c) in triples_set and (a,e,f) in triples_set and \
                       (d,b,c) in triples_set and (d,b,f) in triples_set and (d,e,c) in triples_set and (d,e,f) in triples_set:
                        return False
                    # Check permutation (a,d), (b,f), (c,e)
                    if (all(x != y for x, y in [(a,b,e),(a,f,c),(a,f,e),(d,b,c),(d,b,e),(d,f,c),(d,f,e)])) and \
                       (a,b,e) in triples_set and (a,f,c) in triples_set and (a,f,e) in triples_set and \
                       (d,b,c) in triples_set and (d,b,e) in triples_set and (d,f,c) in triples_set and (d,f,e) in triples_set:
                        return False
                    # Check permutation (a,e), (b,d), (c,f)
                    if (all(x != y for x, y in [(a,b,f),(a,d,c),(a,d,f),(e,b,c),(e,b,f),(e,d,c),(e,d,f)])) and \
                       (a,b,f) in triples_set and (a,d,c) in triples_set and (a,d,f) in triples_set and \
                       (e,b,c) in triples_set and (e,b,f) in triples_set and (e,d,c) in triples_set and (e,d,f) in triples_set:
                        return False
                    # Check permutation (a,e), (b,f), (c,d)
                    if (all(x != y for x, y in [(a,b,d),(a,f,c),(a,f,d),(e,b,c),(e,b,d),(e,f,c),(e,f,d)])) and \
                       (a,b,d) in triples_set and (a,f,c) in triples_set and (a,f,d) in triples_set and \
                       (e,b,c) in triples_set and (e,b,d) in triples_set and (e,f,c) in triples_set and (e,f,d) in triples_set:
                        return False
                    # Check permutation (a,f), (b,d), (c,e)
                    if (all(x != y for x, y in [(a,b,e),(a,d,c),(a,d,e),(f,b,c),(f,b,e),(f,d,c),(f,d,e)])) and \
                       (a,b,e) in triples_set and (a,d,c) in triples_set and (a,d,e) in triples_set and \
                       (f,b,c) in triples_set and (f,b,e) in triples_set and (f,d,c) in triples_set and (f,d,e) in triples_set:
                        return False
                    # Check permutation (a,f), (b,e), (c,d)
                    if (all(x != y for x, y in [(a,b,d),(a,e,c),(a,e,d),(f,b,c),(f,b,d),(f,e,c),(f,e,d)])) and \
                       (a,b,d) in triples_set and (a,e,c) in triples_set and (a,e,d) in triples_set and \
                       (f,b,c) in triples_set and (f,b,d) in triples_set and (f,e,c) in triples_set and (f,e,d) in triples_set:
                        return False
        return True
    
    # This is still too slow. Let's use a completely different approach.
    # Use the fact that we can precompute all K(2,2,2) configurations.
    
    # Total number of K(2,2,2) configurations: choose 6 points, then choose a partition into 3 pairs.
    # C(16,6) * 15 = 8008 * 15 = 120,120 configurations.
    # For each configuration, store the set of 8 transversals.
    
    # Precompute all configurations
    configs = []
    for six in combinations(range(n), 6):
        # All 15 ways to partition 6 elements into 3 pairs
        v0, v1, v2, v3, v4, v5 = six
        partitions = [
            ((v0,v1),(v2,v3),(v4,v5)),
            ((v0,v1),(v2,v4),(v3,v5)),
            ((v0,v1),(v2,v5),(v3,v4)),
            ((v0,v2),(v1,v3),(v4,v5)),
            ((v0,v2),(v1,v4),(v3,v5)),
            ((v0,v2),(v1,v5),(v3,v4)),
            ((v0,v3),(v1,v2),(v4,v5)),
            ((v0,v3),(v1,v4),(v2,v5)),
            ((v0,v3),(v1,v5),(v2,v4)),
            ((v0,v4),(v1,v2),(v3,v5)),
            ((v0,v4),(v1,v3),(v2,v5)),
            ((v0,v4),(v1,v5),(v2,v3)),
            ((v0,v5),(v1,v2),(v3,v4)),
            ((v0,v5),(v1,v3),(v2,v4)),
            ((v0,v5),(v1,v4),(v2,v3)),
        ]
        for parts in partitions:
            p1, p2, p3 = parts
            transversals = set()
            for x in p1:
                for y in p2:
                    for z in p3:
                        transversals.add(tuple(sorted((x, y, z))))
            configs.append(frozenset(transversals))
    
    # For each triple, store the list of configurations that contain it
    triple_to_configs = {t: [] for t in all_triples}
    for i, config in enumerate(configs):
        for t in config:
            triple_to_configs[t].append(i)
    
    # Now for local search:
    # Maintain a count for each config: how many of its 8 transversals are present.
    config_counts = [0] * len(configs)
    
    def add_triple(triples_set, config_counts, t):
        """Add triple t, updating config counts. Returns True if no conflict."""
        for ci in triple_to_configs[t]:
            config_counts[ci] += 1
            if config_counts[ci] == 8:
                # Conflict!
                config_counts[ci] -= 1
                return False
        triples_set.add(t)
        return True
    
    def remove_triple(triples_set, config_counts, t):
        """Remove triple t, updating config counts."""
        for ci in triple_to_configs[t]:
            config_counts[ci] -= 1
        triples_set.discard(t)
    
    # WalkSAT-style local search
    random.seed(42)
    shuffled = all_triples[:]
    random.shuffle(shuffled)
    
    triples_set = set()
    config_counts = [0] * len(configs)
    
    # Phase 1: Greedy add
    for t in shuffled:
        if add_triple(triples_set, config_counts, t):
            pass
    
    best_size = len(triples_set)
    best_triples_copy = set(triples_set)
    
    # Phase 2: Local search with add/swap/fluctuation
    step = 0
    fluctuation_interval = 5000
    
    while time.time() < deadline:
        step += 1
        
        # Fluctuation: randomly remove some triples
        if step % fluctuation_interval == 0:
            for _ in range(random.randint(3, 8)):
                if triples_set:
                    t = random.choice(list(triples_set))
                    remove_triple(triples_set, config_counts, t)
        
        # Try to add a random triple
        if random.random() < 0.7:
            # Try adding a random triple
            candidates = [t for t in shuffled if t not in triples_set]
            if candidates:
                t = random.choice(candidates)
                if add_triple(triples_set, config_counts, t):
                    if len(triples_set) > best_size:
                        best_size = len(triples_set)
                        best_triples_copy = set(triples_set)
                else:
                    # Try swap: remove a random existing triple, then add t
                    if triples_set:
                        t_remove = random.choice(list(triples_set))
                        remove_triple(triples_set, config_counts, t_remove)
                        if add_triple(triples_set, config_counts, t):
                            if len(triples_set) > best_size:
                                best_size = len(triples_set)
                                best_triples_copy = set(triples_set)
                        else:
                            # Revert
                            if t_remove in triples_set:
                                remove_triple(triples_set, config_counts, t_remove)
                            add_triple(triples_set, config_counts, t_remove)
        else:
            # Try swap: remove one, add another
            if triples_set:
                t_remove = random.choice(list(triples_set))
                remove_triple(triples_set, config_counts, t_remove)
                candidates = [t for t in shuffled if t not in triples_set]
                if candidates:
                    t_add = random.choice(candidates)
                    if add_triple(triples_set, config_counts, t_add):
                        if len(triples_set) > best_size:
                            best_size = len(triples_set)
                            best_triples_copy = set(triples_set)
                    else:
                        # Revert
                        add_triple(triples_set, config_counts, t_remove)
    
    return [list(t) for t in best_triples_copy]
# EVOLVE-BLOCK-END
