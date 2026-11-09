"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency set for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all 4-cliques and 4-independent sets for fast validity checking
    all_4cliques = []
    all_4indep = []
    for combo in combinations(range(n), 4):
        is_clique = True
        is_indep = True
        for i in range(4):
            for j in range(i+1, 4):
                if combo[j] in adj[combo[i]]:
                    is_indep = False
                else:
                    is_clique = False
        if is_clique:
            all_4cliques.append(combo)
        if is_indep:
            all_4indep.append(combo)
    
    def is_valid(vertices):
        vset = set(vertices)
        for clique in all_4cliques:
            if all(v in vset for v in clique):
                return False
        for indep in all_4indep:
            if all(v in vset for v in indep):
                return False
        return True
    
    def count_distinct_degrees(vertices):
        vset = set(vertices)
        degrees = set()
        for v in vertices:
            deg = sum(1 for u in vset if u != v and u in adj[v])
            degrees.add(deg)
        return len(degrees)
    
    best_count = 0
    best_vertices = None
    
    # Randomized Local Search
    num_restarts = 200
    for _ in range(num_restarts):
        # Initialize with a random valid set of size min_v
        # We need to find a valid starting set.
        # Simple strategy: pick random vertices until we have min_v and it's valid.
        # If we fail, retry.
        current_set = None
        for _ in range(50):
            cand = random.sample(range(n), min_v)
            if is_valid(cand):
                current_set = set(cand)
                break
        
        if current_set is None:
            continue
            
        current_score = count_distinct_degrees(current_set)
        
        # Local Search: try to improve by adding vertices or swapping
        improved = True
        while improved:
            improved = False
            # Try adding a vertex
            for v in range(n):
                if v not in current_set:
                    new_set = current_set | {v}
                    if is_valid(new_set):
                        new_score = count_distinct_degrees(new_set)
                        if new_score > current_score:
                            current_set = new_set
                            current_score = new_score
                            improved = True
                            break
            # Try swapping a vertex
            for v_in in list(current_set):
                for v_out in range(n):
                    if v_out not in current_set:
                        new_set = (current_set - {v_in}) | {v_out}
                        if is_valid(new_set):
                            new_score = count_distinct_degrees(new_set)
                            if new_score > current_score:
                                current_set = new_set
                                current_score = new_score
                                improved = True
                                break
                if improved:
                    break
        
        if current_score > best_count:
            best_count = current_score
            best_vertices = list(current_set)
            
    if best_vertices is None:
        # Fallback: find any valid set of size min_v
        for combo in combinations(range(n), min_v):
            if is_valid(combo):
                return list(combo)
        return list(range(min_v))
    
    return list(best_vertices)
# EVOLVE-BLOCK-END
