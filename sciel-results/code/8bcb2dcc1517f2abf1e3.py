"""Rewritten program using maximum independent set formulation with efficient conflict detection."""
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large set of triples on 16 points with no K(2,2,2) using
    a greedy maximum independent set heuristic with dynamic weights.
    """
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Build conflict graph
    conflict_sets = [set() for _ in range(num_triples)]
    
    # For each triple {a,b,c}, find all conflicting triples
    for t_idx, (a, b, c) in enumerate(all_triples):
        seen_conflicts = set()
        for a_prime in range(n):
            if a_prime == a:
                continue
            for b_prime in range(n):
                if b_prime in (a, b, a_prime):
                    continue
                for c_prime in range(n):
                    if c_prime in (a, b, c, a_prime, b_prime):
                        continue
                    part1 = [a, a_prime]
                    part2 = [b, b_prime]
                    part3 = [c, c_prime]
                    
                    for x in part1:
                        for y in part2:
                            for z in part3:
                                t2 = tuple(sorted((x, y, z)))
                                idx2 = triple_to_idx[t2]
                                if idx2 != t_idx:
                                    seen_conflicts.add(idx2)
        
        conflict_sets[t_idx] = seen_conflicts
    
    # Greedy independent set with dynamic weight updates
    # Use multiple strategies and pick the best
    
    best_solution = []
    
    for seed in range(20):
        random.seed(seed)
        active = set(range(num_triples))
        solution = []
        weights = [1.0] * num_triples
        
        while active:
            # Compute effective weight: prefer vertices with lower degree in remaining graph
            # but also consider original weight
            best_v = None
            best_score = float('inf')
            
            for v in active:
                deg = sum(1 for nb in conflict_sets[v] if nb in active)
                # Score: lower is better (we want to pick low-degree vertices)
                score = deg / (weights[v] + 1e-9)
                if score < best_score:
                    best_score = score
                    best_v = v
            
            solution.append(all_triples[best_v])
            active.remove(best_v)
            
            # Remove neighbors
            removed_neighbors = set()
            for nb in conflict_sets[best_v]:
                if nb in active:
                    active.remove(nb)
                    removed_neighbors.add(nb)
            
            # Update weights: boost weights of vertices that had high conflict with removed set
            # This encourages picking vertices that are "clean" in the remaining graph
            for v in active:
                conflicts_with_removed = sum(1 for nb in conflict_sets[v] if nb in removed_neighbors)
                # Increase weight for vertices with fewer conflicts (they're more "available")
                weights[v] = weights[v] * (1.0 + 0.1 * (len(conflict_sets[v]) - conflicts_with_removed))
        
        if len(solution) > len(best_solution):
            best_solution = solution
    
    return [list(t) for t in best_solution]
# EVOLVE-BLOCK-END
