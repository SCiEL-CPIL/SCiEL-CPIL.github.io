"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    
    # Generate all C(16,3) = 560 triples
    all_triples = list(combinations(range(n), 3))
    all_triples_set = set(all_triples)
    
    # For each triple, find all K(2,2,2)s it belongs to
    # A K(2,2,2) is defined by 3 disjoint pairs: (a1,a2), (b1,b2), (c1,c2)
    # The 8 transversal triples are {ai, bj, ck} for all i,j,k in {0,1}
    
    # Precompute: for each triple, list of K(2,2,2) IDs it belongs to
    # First, enumerate all K(2,2,2)s
    
    # A K(2,2,2) consists of 6 points forming 3 disjoint pairs
    # The 8 transversal triples are all combinations choosing one from each pair
    
    # Let's enumerate all possible K(2,2,2) configurations
    # A K(2,2,2) is determined by choosing 6 points and partitioning them into 3 pairs
    # But we need to be careful: the same set of 8 triples can arise from different pairings
    
    # Actually, let's think differently. A K(2,2,2) configuration is a set of 8 triples
    # that form the complete 3-partite 3-uniform hypergraph K(2,2,2).
    # These 8 triples are exactly all triples {a_i, b_j, c_k} for i,j,k in {0,1}
    # where {a1,a2}, {b1,b2}, {c1,c2} are three disjoint pairs.
    
    # So we need to:
    # 1. Enumerate all ways to choose 3 disjoint pairs from 16 points
    # 2. For each such choice, compute the 8 transversal triples
    # 3. Check if all 8 are present in our triple set
    
    # Let's precompute all K(2,2,2) configurations
    # Each is identified by the set of 8 triples it contains
    
    k222_configs = []
    # Choose 6 points, then partition into 3 pairs
    # Number of ways: C(16,6) * (6!/(2^3 * 3!)) = 8008 * 15 = 120120
    # This is manageable
    
    for six_points in combinations(range(n), 6):
        # Partition 6 points into 3 unordered pairs
        # Number of perfect matchings of 6 points: 15
        a, b, c, d, e, f = six_points
        pairings = [
            ((a,b), (c,d), (e,f)),
            ((a,b), (c,e), (d,f)),
            ((a,b), (c,f), (d,e)),
            ((a,c), (b,d), (e,f)),
            ((a,c), (b,e), (d,f)),
            ((a,c), (b,f), (d,e)),
            ((a,d), (b,c), (e,f)),
            ((a,d), (b,e), (c,f)),
            ((a,d), (b,f), (c,e)),
            ((a,e), (b,c), (d,f)),
            ((a,e), (b,d), (c,f)),
            ((a,e), (b,f), (c,d)),
            ((a,f), (b,c), (d,e)),
            ((a,f), (b,d), (c,e)),
            ((a,f), (b,e), (c,d)),
        ]
        
        for p1, p2, p3 in pairings:
            # Generate the 8 transversal triples
            triples_in_config = set()
            for i in [0, 1]:
                for j in [0, 1]:
                    for k in [0, 1]:
                        triple = tuple(sorted((p1[i], p2[j], p3[k])))
                        triples_in_config.add(triple)
            
            k222_configs.append(frozenset(triples_in_config))
    
    # Deduplicate configs (same set of 8 triples might arise from different pairings)
    k222_configs = list(set(k222_configs))
    
    # For each triple, find which K(2,2,2) configs it belongs to
    triple_to_configs = {t: [] for t in all_triples}
    for idx, config in enumerate(k222_configs):
        for t in config:
            if t in triple_to_configs:
                triple_to_configs[t].append(idx)
    
    # Now use a greedy + local search approach
    # Start with a random subset of triples, then try to add more or swap
    
    best_triples = set()
    best_count = 0
    
    def is_valid(triple_set):
        """Check if no K(2,2,2) is fully contained."""
        for config in k222_configs:
            if config.issubset(triple_set):
                return False
        return True
    
    # Greedy construction: start empty, add triples in random order if valid
    random.seed(42)
    
    # Try multiple random orderings
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        order = list(all_triples)
        random.shuffle(order)
        
        current = set()
        for t in order:
            # Check if adding t creates a K(2,2,2)
            # Only need to check configs that contain t
            conflict = False
            for config_idx in triple_to_configs[t]:
                config = k222_configs[config_idx]
                if config.issubset(current | {t}):
                    conflict = True
                    break
            if not conflict:
                current.add(t)
        
        if len(current) > best_count:
            best_count = len(current)
            best_triples = set(current)
    
    # Local search: try to improve by removing one triple and adding others
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try removing each triple and see if we can add more
        for t_to_remove in list(best_triples):
            if time.time() > deadline - 5:
                break
            
            candidate = best_triples - {t_to_remove}
            added = 0
            
            # Try to add triples not in candidate
            for t_add in all_triples:
                if t_add in candidate:
                    continue
                conflict = False
                for config_idx in triple_to_configs[t_add]:
                    config = k222_configs[config_idx]
                    if config.issubset(candidate | {t_add}):
                        conflict = True
                        break
                if not conflict:
                    candidate.add(t_add)
                    added += 1
            
            if len(candidate) > len(best_triples):
                best_triples = candidate
                improved = True
        
        if not improved:
            break
    
    # Convert to list of lists
    result = [list(t) for t in best_triples]
    
    # Verify validity
    triple_set = set(tuple(sorted(t)) for t in result)
    for config in k222_configs:
        if config.issubset(triple_set):
            # Should not happen, but if it does, remove problematic triples
            for t in config:
                if t in triple_set:
                    triple_set.remove(t)
                    break
            break
    
    return [list(t) for t in triple_set]
# EVOLVE-BLOCK-END
