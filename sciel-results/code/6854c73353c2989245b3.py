"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency and edge sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    total_edges = len(edge_set)
    if total_edges == 0:
        return []
    
    # Helper: check if a biclique is valid (all cross pairs are edges)
    def is_valid_biclique(L, R):
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    return False
        return True
    
    # Helper: get the set of edges covered by a biclique
    def get_covered_edges(L, R):
        covered = set()
        for u in L:
            for v in R:
                covered.add((min(u, v), max(u, v)))
        return covered
    
    # Generate candidate bicliques using various strategies
    
    # Strategy 1: For each vertex, find maximal bicliques where it's in one side
    candidates = []
    seen_keys = set()
    
    # Strategy 1: Single vertex on left, try to extend right side
    for u in range(n):
        # Start with u on left, all neighbors on right
        L = [u]
        R = list(adj[u])
        if not R:
            continue
        
        # Try to maximize by greedily adding vertices
        # Actually, let's try a simpler approach: for each pair (u, v), 
        # find the largest common neighborhood and build a biclique
        
        pass
    
    # Strategy 2: For each edge (u, v), build a biclique
    # L = {u} + common neighbors that can form a complete bipartite with R
    # R = {v} + ...
    
    # Let's use a different approach: enumerate all maximal bicliques
    # using a recursive backtracking method, but limit depth
    
    # Simpler: use greedy construction
    # 1. Find large bicliques by looking at dense subgraphs
    
    # Strategy: For each subset of size 2-3 on left, find max right side
    def find_biclique_from_seed(L_seed, R_seed):
        """Expand a seed biclique to be maximal"""
        L = set(L_seed)
        R = set(R_seed)
        
        # Verify validity
        if not is_valid_biclique(list(L), list(R)):
            return None
        
        # Try to add vertices to L
        changed = True
        while changed:
            changed = False
            # A vertex can be added to L if it is adjacent to all in R
            for v in range(n):
                if v not in L:
                    # Check if v is adjacent to all in R
                    if all(w in adj[v] for w in R):
                        L.add(v)
                        changed = True
        
        # Try to add vertices to R
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if v not in R:
                    if all(w in adj[v] for w in L):
                        R.add(v)
                        changed = True
        
        if not L or not R:
            return None
        
        return (list(L), list(R))
    
    # Generate candidates from pairs of edges
    edge_list = list(edge_set)
    
    for i in range(len(edge_list)):
        if time.time() > deadline - 30:
            break
        
        u1, v1 = edge_list[i]
        
        # Try L={u1}, R={v1}
        result = find_biclique_from_seed([u1], [v1])
        if result:
            L, R = result
            # Deduplicate: sort and use frozenset as key
            key = (frozenset(L), frozenset(R))
            # Also try swapped
            key2 = (frozenset(R), frozenset(L))
            if key not in seen_keys and key2 not in seen_keys:
                seen_keys.add(key)
                candidates.append((L, R))
        
        # Try L={u1,v1}, R={} -> won't work since R must be non-empty
        # Try L={u1}, R={v1} with other vertices added
        
        # For each other vertex w, try L={u1}, R={v1,w} if u1-w and u1-v1 and v1-w... 
        # Actually, let's try: for each edge (u2,v2), try biclique with L={u1,u2}, R={v1} if possible
        for j in range(i+1, len(edge_list)):
            if time.time() > deadline - 25:
                break
            
            u2, v2 = edge_list[j]
            
            # Try L={u1, u2}, R={v1}
            if u2 in adj[v1]:  # u2-v1 must be an edge
                result = find_biclique_from_seed([u1, u2], [v1])
                if result:
                    L, R = result
                    key = (frozenset(L), frozenset(R))
                    key2 = (frozenset(R), frozenset(L))
                    if key not in seen_keys and key2 not in seen_keys:
                        seen_keys.add(key)
                        candidates.append((L, R))
            
            # Try L={u1}, R={v1, v2}
            if u1 in adj[v2]:  # u1-v2 must be an edge
                result = find_biclique_from_seed([u1], [v1, v2])
                if result:
                    L, R = result
                    key = (frozenset(L), frozenset(R))
                    key2 = (frozenset(R), frozenset(L))
                    if key not in seen_keys and key2 not in seen_keys:
                        seen_keys.add(key)
                        candidates.append((L, R))
            
            # Try L={u1, u2}, R={v1, v2}
            if (u2 in adj[v1]) and (u1 in adj[v2]):
                result = find_biclique_from_seed([u1, u2], [v1, v2])
                if result:
                    L, R = result
                    key = (frozenset(L), frozenset(R))
                    key2 = (frozenset(R), frozenset(L))
                    if key not in seen_keys and key2 not in seen_keys:
                        seen_keys.add(key)
                        candidates.append((L, R))
        
        # Also try single vertex bicliques: L={u1}, R=all neighbors
        result = find_biclique_from_seed([u1], list(adj[u1]))
        if result:
            L, R = result
            key = (frozenset(L), frozenset(R))
            key2 = (frozenset(R), frozenset(L))
            if key not in seen_keys and key2 not in seen_keys:
                seen_keys.add(key)
                candidates.append((L, R))
    
    # If no candidates found, fallback to individual edges
    if not candidates:
        result = []
        for u, v in edges:
            result.append(([int(u)], [int(v)]))
        return result
    
    # Precompute covered edges for each candidate
    cand_covers = []
    for L, R in candidates:
        covered = get_covered_edges(L, R)
        if covered:
            cand_covers.append(covered)
    
    # Now solve set cover: minimize number of bicliques to cover all edges
    # Use greedy with randomization and local search
    
    def greedy_set_cover(candidates, covers, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        uncovered = set(edge_set)
        selected = []
        remaining = list(range(len(candidates)))
        
        while uncovered and remaining:
            if time.time() > deadline - 10:
                break
            
            # Find candidate that covers most uncovered edges
            best_idx = -1
            best_count = 0
            
            # Randomize tie-breaking
            for idx in remaining:
                count = len(covers[idx] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = idx
                elif count == best_count and count > 0:
                    # Random tie-break
                    if rng.random() < 0.5:
                        best_idx = idx
            
            if best_idx == -1:
                # Can't cover remaining edges, add individual edges
                for e in uncovered:
                    u, v = e
                    selected.append(([u], [v]))
                break
            
            selected.append(candidates[best_idx])
            uncovered -= covers[best_idx]
            remaining.remove(best_idx)
        
        # If there are still uncovered edges, add them as single-edge bicliques
        for e in uncovered:
            u, v = e
            selected.append(([u], [v]))
        
        return selected
    
    # Run greedy set cover multiple times with different random seeds
    best_solution = None
    best_size = float('inf')
    
    # First, a deterministic greedy
    solution = greedy_set_cover(candidates, cand_covers)
    best_solution = solution
    best_size = len(solution)
    
    # Then randomized tries
    num_tries = 20
    for trial in range(num_tries):
        if time.time() > deadline - 5:
            break
        solution = greedy_set_cover(candidates, cand_covers, random_seed=trial)
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution
    
    # Local search: try to replace a biclique with multiple smaller ones, or merge
    # Actually, try to remove a biclique if its edges are covered by others
    if best_solution:
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
            
            # Try removing each biclique and see if coverage is maintained
            n_sol = len(best_solution)
            for i in range(n_sol):
                if time.time() > deadline - 2:
                    break
                
                # Check if biclique i's edges are covered by the rest
                edges_i = get_covered_edges(best_solution[i][0], best_solution[i][1])
                
                # Build union of all other bicliques' covers
                covered_by_others = set()
                for j in range(n_sol):
                    if j != i:
                        covered_by_others |= get_covered_edges(best_solution[j][0], best_solution[j][1])
                
                if edges_i.issubset(covered_by_others):
                    # Can remove biclique i
                    best_solution = best_solution[:i] + best_solution[i+1:]
                    improved = True
                    break  # Restart the loop
    
    # Final validation and formatting
    result = []
    for L, R in best_solution:
        L_clean = [int(x) for x in L]
        R_clean = [int(x) for x in R]
        if L_clean and R_clean:
            result.append((L_clean, R_clean))
    
    return result

# EVOLVE-BLOCK-END
