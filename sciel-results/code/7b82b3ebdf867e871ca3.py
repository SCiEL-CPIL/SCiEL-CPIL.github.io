"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute: for each triple, the bitmask of triples that conflict with it
    # Two triples conflict if they share a pair of points (i.e., they are the same triple, 
    # or they share exactly 2 points)
    # Actually, the constraint is: no three disjoint pairs have all eight transversal triples present.
    # Let's re-read the task carefully.
    # "no three disjoint pairs have all eight transversal triples present"
    # This means: we cannot have a set of triples that contains all 8 transversals of a K(2,2,2).
    # A K(2,2,2) has 3 parts of size 2. Let the parts be {a1,a2}, {b1,b2}, {c1,c2}.
    # The 8 transversals are the triples that pick one from each part: 
    # (a1,b1,c1), (a1,b1,c2), (a1,b2,c1), (a1,b2,c2), 
    # (a2,b1,c1), (a2,b1,c2), (a2,b2,c1), (a2,b2,c2).
    # So a "bad" configuration is when our set of triples contains all 8 of these.
    # We want to maximize the number of triples such that for every choice of 3 disjoint pairs, 
    # we do NOT contain all 8 transversals.
    
    # This is equivalent to: the set of triples we select must not contain the entire set of 8 transversals
    # for any 3 disjoint pairs.
    
    # How many such "forbidden 8-sets" are there?
    # A K(2,2,2) is defined by 3 disjoint pairs. The number of ways to choose 3 disjoint pairs from 16 points:
    # Choose 6 points: C(16,6) = 8008
    # Partition 6 points into 3 pairs: 6! / (2!^3 * 3!) = 720 / (8*6) = 15
    # Total K(2,2,2) structures: 8008 * 15 = 120120
    # Each structure forbids one specific set of 8 triples.
    
    # So we have 120120 constraints, each saying "not all 8 of these triples can be selected".
    # This is a 0-1 integer programming problem, or we can use a greedy/heuristic approach.
    
    # Strategy: Start with all triples, then iteratively remove triples to break all "bad" configurations.
    # Or: Greedily add triples, checking if adding a triple completes any bad configuration.
    
    # Better strategy: 
    # 1. Enumerate all K(2,2,2) structures and their 8 transversals.
    # 2. Use a greedy algorithm: repeatedly pick the triple that is in the fewest "active" bad configurations,
    #    or use a maximum independent set heuristic on a conflict graph.
    
    # Actually, let's think of it as a hypergraph problem. We have a hypergraph where hyperedges are the 8-transversal sets.
    # We want the maximum subset of triples that hits no hyperedge completely.
    # This is the "maximum subset avoiding a set of forbidden configurations" problem.
    
    # Greedy approach:
    # Start with all triples.
    # While there exists a bad configuration (all 8 present), remove one triple from it.
    # Which one to remove? The one that appears in the most other bad configurations (to break more at once).
    
    # Let's precompute for each triple, how many bad configurations it participates in.
    # And for each bad configuration, which triples it contains.
    
    # Step 1: Generate all bad configurations.
    bad_configs = []  # list of frozensets of 8 triples
    triple_to_configs = [set() for _ in range(num_triples)]
    
    # Map from triple to index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Generate all 3 disjoint pairs from 16 points
    # For each set of 6 points, generate all 15 pairings
    for six_points in combinations(range(n), 6):
        # Generate all perfect matchings of 6 points
        # Number of perfect matchings of 6 points is 15
        # We can generate them by fixing the first point and pairing it with each of the other 5,
        # then recursively matching the remaining 4.
        def gen_matchings(points):
            if not points:
                yield []
                return
            p0 = points[0]
            for i in range(1, len(points)):
                p1 = points[i]
                rest = points[1:i] + points[i+1:]
                for m in gen_matchings(rest):
                    yield [(p0, p1)] + m
        
        for matching in gen_matchings(six_points):
            # matching is a list of 3 pairs
            # The 8 transversals are all combinations of one from each pair
            pairs = matching
            transversals = []
            for c1 in range(2):
                for c2 in range(2):
                    for c3 in range(2):
                        triple = tuple(sorted((pairs[0][c1], pairs[1][c2], pairs[2][c3])))
                        transversals.append(triple)
            # Convert to indices
            config = frozenset(triple_to_idx[t] for t in transversals)
            bad_configs.append(config)
            for idx in config:
                triple_to_configs[idx].add(len(bad_configs) - 1)
    
    # Step 2: Greedy removal
    active = [True] * num_triples
    config_active = [True] * len(bad_configs)
    
    # Count for each config how many of its triples are still active
    config_count = [8] * len(bad_configs)
    
    # Count for each triple how many active configs it's in
    triple_config_count = [len(triple_to_configs[i]) for i in range(num_triples)]
    
    while True:
        if time.time() > deadline:
            break
        
        # Find a bad config that is fully active
        found = -1
        for ci in range(len(bad_configs)):
            if config_active[ci] and config_count[ci] == 8:
                found = ci
                break
        
        if found == -1:
            break
        
        # All 8 triples in this config are active. Remove one.
        # Choose the triple that appears in the most other active configs.
        config = bad_configs[found]
        best_triple = -1
        best_count = -1
        for t in config:
            if active[t]:
                # Count how many active configs this triple is in
                count = 0
                for ci2 in triple_to_configs[t]:
                    if config_active[ci2]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_triple = t
        
        # Remove best_triple
        active[best_triple] = False
        
        # Update config counts
        for ci2 in triple_to_configs[best_triple]:
            if config_active[ci2]:
                config_count[ci2] -= 1
                if config_count[ci2] < 8:
                    config_active[ci2] = False
        
        # Update triple_config_count for other triples? Not strictly needed if we recompute on the fly.
    
    # Build result
    result = [list(all_triples[i]) for i in range(num_triples) if active[i]]
    
    # Try to improve with local search: try to add back removed triples if possible
    removed = [i for i in range(num_triples) if not active[i]]
    random.shuffle(removed)
    
    for t in removed:
        if time.time() > deadline:
            break
        # Check if adding t would complete any bad config
        can_add = True
        for ci in triple_to_configs[t]:
            if config_count[ci] == 7:  # 7 of the other 8 are active
                can_add = False
                break
        if can_add:
            active[t] = True
            for ci in triple_to_configs[t]:
                if config_count[ci] < 8:
                    config_count[ci] += 1
                    if config_count[ci] == 8:
                        config_active[ci] = True
    
    result = [list(all_triples[i]) for i in range(num_triples) if active[i]]
    return result
# EVOLVE-BLOCK-END
