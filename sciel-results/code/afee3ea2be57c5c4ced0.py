import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations
    configs = []
    config_id_map = {}
    triples_in_config = []
    
    points = range(N)
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
                
    num_configs = len(configs)
    
    # Build conflict graph
    # adj[i] is a set of indices j such that i and j cannot be in the same independent set
    # Two triples conflict if they are part of the same K(2,2,2) configuration
    # Actually, the constraint is: no configuration has ALL 8 triples present.
    # This is a "packing" problem, not a simple graph independent set on triples.
    # Wait, the conflict is not pairwise.
    # If we select a set of triples S, S is valid iff for every config C, |S ∩ C| < 8.
    # This is a 3-uniform hypergraph packing problem, specifically a (1, 7)-packing or similar?
    # No, it's a "7-packing" in the sense that we can have at most 7 from any specific set of 8.
    
    # This is NOT a standard Maximum Independent Set problem on a graph.
    # It is a maximum size subset of a 3-uniform hypergraph such that no specific 8-set is fully contained.
    # This is a "Turán-type" problem or a hypergraph matching/packing problem.
    
    # The previous program used Simulated Annealing because it's a general combinatorial optimization problem.
    # Can we model this as an Integer Linear Program?
    # Maximize sum(x_i)
    # Subject to: for each config C, sum_{i in C} x_i <= 7
    # x_i in {0, 1}
    
    # Since we don't have a generic ILP solver that is guaranteed to be fast enough for 560 variables and ~1000 constraints in 120s without scipy.optimize.milp (which might be slow or unavailable/limited), we should use a custom branch and bound or a heuristic that exploits the structure.
    
    # However, the directive suggests "Maximum Independent Set in a 560-vertex conflict graph".
    # Is there a conflict graph formulation?
    # If we define the conflict graph G where vertices are triples, and edges represent "incompatibility", then an Independent Set in G would be a set of triples with no conflicts.
    # But the conflict here is "8 of these 8 cannot be present". This is a hypergraph constraint.
    # We can transform this into a graph problem by adding "dummy" vertices or using a different formulation, but it's not a direct IS problem.
    
    # Let's look at the structure again.
    # A K(2,2,2) configuration consists of 8 triples.
    # The condition is that we cannot have all 8.
    # This is equivalent to saying: For every config C, at least one triple in C must be EXCLUDED.
    # This is a "Hitting Set" problem if we think about excluded triples, but we want to MAXIMIZE included triples, which is equivalent to MINIMIZING excluded triples.
    # Let Y be the set of excluded triples.
    # We want to minimize |Y| such that for every config C, C ∩ Y != empty.
    # This is exactly the Minimum Hitting Set (or Minimum Vertex Cover in a hypergraph) problem!
    # The hypergraph H has vertices = all 560 triples.
    # The hyperedges are the 8-sets corresponding to each K(2,2,2) configuration.
    # We want to find a minimum size set of vertices (triples) that hits every hyperedge (config).
    # The size of the maximum valid triple system is 560 - min_hitting_set_size.
    
    # So the problem reduces to: Find the Minimum Hitting Set of the hypergraph defined by the K(2,2,2) configurations.
    # This is NP-hard, but the structure might help.
    # Number of configs:
    # 6 points can be chosen in C(16,6) ways.
    # For each 6-set, there are 15 pairings (perfect matchings of 6 points).
    # Total configs = C(16,6) * 15 = 8008 * 15 = 120120.
    # So we have a hypergraph with 560 vertices and ~120,000 hyperedges of size 8.
    
    # Minimum Hitting Set can be solved via Branch and Bound.
    # Or, we can use the duality: Max Independent Set in the "conflict" structure?
    # No, Min Hitting Set is the dual of Max Packing.
    
    # Strategy: Use a Branch and Bound solver for Minimum Hitting Set.
    # Or, since we want to maximize the number of triples, we can use a Greedy + Local Search approach specifically tailored for Hitting Set/Packing.
    
    # Given the time limit, a pure exact B&B might be too slow if the instance is hard.
    # However, the structure is highly symmetric (affine geometry F2^4?).
    # Let's check if there's a known algebraic construction.
    # The seed used affine lines.
    # A better approach for Hitting Set is often a Greedy algorithm followed by Local Search (minimizing the hitting set size).
    
    # Let's implement a Greedy Hitting Set solver with Local Search (1-swap, 2-swap).
    # Objective: Minimize |Y| (excluded triples).
    # Constraint: Every config C must have at least one triple in Y.
    
    # 1. Greedy Initialization:
    #    While there are unhit configs:
    #       Pick the triple t that hits the most unhit configs.
    #       Add t to Y.
    #       Mark all configs containing t as hit.
    
    # 2. Local Search:
    #    Try to remove a triple t from Y.
    #    If removing t makes some config C unhit (i.e., C had ONLY t in Y), then we must add another triple from C to Y.
    #    We want to swap: remove t, add t' from C, such that the net change in |Y| is 0 (swap) or -1 (if we can remove t and don't need to add? No, if C becomes unhit, we MUST add something).
    #    Actually, if we remove t, and C becomes unhit, we must add SOME triple from C.
    #    If we add t' (where t' != t), |Y| stays same.
    #    Can we reduce |Y|?
    #    Only if removing t doesn't unhit any config. Then |Y| decreases by 1.
    #    So the local search should first try to remove any t that is "redundant" (its removal doesn't unhit any config).
    #    If no redundant t exists, try swaps: remove t, add t' from a config that becomes unhit.
    #    This is essentially a local search on the Hitting Set.
    
    # This is a much more structured problem than the generic SA in the seed.
    
    # Let's implement this.
    
    # Data Structures:
    # triple_to_configs: list of lists of config indices for each triple.
    # config_to_triples: list of lists of triple indices for each config.
    # present_in_Y: boolean array for excluded triples.
    # config_hit_count: number of triples in Y for each config.
    
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    config_to_triples = triples_in_config # Already computed
    
    # Greedy Hitting Set
    def greedy_hitting_set():
        unhit_configs = set(range(num_configs))
        Y = set()
        
        # Count how many unhit configs each triple hits
        # To be efficient, we can maintain a count of unhit configs per triple.
        # Initially, count[t] = number of configs containing t.
        hit_counts = [0] * num_triples
        for t in range(num_triples):
            hit_counts[t] = len(triple_to_configs[t])
            
        while unhit_configs:
            if time.time() > deadline - 2:
                break
            
            # Find triple with max hit count among those that hit at least one unhit config
            # Optimization: Only consider triples that are in some unhit config.
            # We can just iterate all triples and check max hit count.
            # 560 triples is small.
            best_t = -1
            max_hits = 0
            for t in range(num_triples):
                if hit_counts[t] > max_hits:
                    max_hits = hit_counts[t]
                    best_t = t
                    
            if best_t == -1:
                break # Should not happen if unhit_configs is not empty
                
            # Add best_t to Y
            Y.add(best_t)
            
            # Update hit counts for all triples in configs that best_t hits
            # For each config C containing best_t:
            #   If C is in unhit_configs:
            #       Remove C from unhit_configs
            #       For each other triple t' in C:
            #           hit_counts[t'] -= 1
            
            for cid in triple_to_configs[best_t]:
                if cid in unhit_configs:
                    unhit_configs.remove(cid)
                    for t_prime in config_to_triples[cid]:
                        if t_prime != best_t:
                            hit_counts[t_prime] -= 1
                            
        return Y, unhit_configs
        
    # Local Search to minimize |Y|
    def local_search_hitting_set(Y):
        # Y is a set of triple indices
        # We want to minimize len(Y)
        
        # Precompute for each config, the list of triples in Y
        # config_Y_count[c] = number of triples in Y for config c
        config_Y_count = [0] * num_configs
        for t in Y:
            for cid in triple_to_configs[t]:
                config_Y_count[cid] += 1
                
        # Check validity: all config_Y_count should be >= 1
        # (Greedy ensures this)
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
                
            # 1. Try to remove redundant triples
            # A triple t is redundant if removing it does not make any config unhit.
            # i.e., for all cid in triple_to_configs[t], config_Y_count[cid] > 1.
            
            redundant_candidates = []
            for t in Y:
                is_redundant = True
                for cid in triple_to_configs[t]:
                    if config_Y_count[cid] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_candidates.append(t)
                    
            if redundant_candidates:
                # Remove all redundant triples? Or one by one?
                # Removing one might make another non-redundant?
                # No, if t is redundant, removing it decreases config_Y_count for its configs by 1.
                # This might make OTHER triples in those configs become critical (count becomes 1).
                # So we should remove them one by one, or carefully.
                # Let's just remove one and restart the loop to re-evaluate.
                t_to_remove = redundant_candidates[0]
                Y.remove(t_to_remove)
                for cid in triple_to_configs[t_to_remove]:
                    config_Y_count[cid] -= 1
                improved = True
                continue # Re-evaluate redundancy
                
            # 2. Try swaps
            # Find a triple t in Y such that removing it makes some config C unhit (count becomes 0).
            # Then we must add another triple t' from C to Y.
            # This is a swap: Y = Y \ {t} ∪ {t'}.
            # Does this help?
            # It keeps |Y| same.
            # But it might enable future removals.
            # We need a heuristic to choose t and t'.
            
            # For each t in Y:
            #   Find configs C where t is the ONLY triple in Y (config_Y_count[C] == 1).
            #   For each such C, find a triple t' in C \ Y.
            #   Evaluate the "quality" of the swap.
            #   A good swap might be one where t' hits many configs that are currently "tight" (count=1),
            #   potentially allowing other triples to become redundant?
            #   Or simply: try random swaps and accept if it doesn't increase |Y| (it never does)
            #   and maybe use a tie-breaker or randomization to escape local minima.
            
            # To avoid getting stuck, we can do a limited number of random swaps.
            
            # Identify "critical" triples: t in Y such that it is the unique hitter for some config.
            critical_triples = []
            for t in Y:
                for cid in triple_to_configs[t]:
                    if config_Y_count[cid] == 1:
                        critical_triples.append(t)
                        break # Just need to know it's critical
                        
            if not critical_triples:
                # No critical triples, but no redundant ones either?
                # This implies a local minimum where every triple is critical for at least one config,
                # and no triple is redundant.
                # We must swap.
                pass
            
            # Pick a random critical triple t to swap out
            if critical_triples:
                t_out = random.choice(critical_triples)
                
                # Find a config C where t_out is the unique hitter
                target_config = None
                for cid in triple_to_configs[t_out]:
                    if config_Y_count[cid] == 1:
                        target_config = cid
                        break
                        
                if target_config is None:
                    # Should not happen if t_out was identified as critical
                    continue
                    
                # Find a triple t_in in target_config that is NOT in Y
                candidates_in = [t for t in config_to_triples[target_config] if t not in Y]
                
                if candidates_in:
                    # Choose t_in randomly or by heuristic
                    # Heuristic: Choose t_in that hits the most "tight" configs (count=1) other than target_config?
                    # Or just random.
                    t_in = random.choice(candidates_in)
                    
                    # Perform swap
                    Y.remove(t_out)
                    Y.add(t_in)
                    
                    # Update counts
                    for cid in triple_to_configs[t_out]:
                        config_Y_count[cid] -= 1
                    for cid in triple_to_configs[t_in]:
                        config_Y_count[cid] += 1
                        
                    improved = True # Force restart of loop to check for new redundancies
                    continue
                    
        return Y
        
    # Run the algorithm
    Y, _ = greedy_hitting_set()
    Y = local_search_hitting_set(Y)
    
    # The valid triple system is all triples NOT in Y
    result = [all_triples[i] for i in range(num_triples) if i not in Y]
    
    return result

