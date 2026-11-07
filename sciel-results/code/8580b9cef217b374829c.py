"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import copy

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    n = P['n']
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    def get_biclique_edges(L, R):
        """Get set of edges in graph between L and R."""
        es = set()
        for u in L:
            for v in R:
                if v in adj[u]:
                    es.add((min(u,v), max(u,v)))
        return es
    
    def is_valid_partition(bicliques):
        """Check if bicliques form a valid partition of all edges."""
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            for u in L:
                for v in R:
                    if v in adj[u]:
                        key = (min(u,v), max(u,v))
                        if key in covered:
                            return False
                        covered.add(key)
        return covered == edge_set
    
    def greedy_partition(seed=None):
        """Greedy biclique partition with randomization."""
        if seed is not None:
            random.seed(seed)
        
        remaining = set(edge_set)
        bicliques = []
        
        while remaining and time.time() < deadline:
            # Find a good biclique from remaining edges
            # Strategy: pick a random edge, try to extend it
            edge_list = list(remaining)
            random.shuffle(edge_list)
            
            best_L = None
            best_R = None
            best_count = 0
            
            for u, v in edge_list[:50]:  # Try up to 50 starting edges
                if time.time() > deadline:
                    break
                
                # Start with this edge
                L = {u}
                R = {v}
                
                # Extend L: add vertices connected to all in R
                changed = True
                while changed:
                    changed = False
                    for w in range(n):
                        if w not in L and w not in R:
                            if all(w in adj[x] for x in R):
                                L.add(w)
                                changed = True
                
                # Extend R: add vertices connected to all in L
                changed = True
                while changed:
                    changed = False
                    for w in range(n):
                        if w not in L and w not in R:
                            if all(w in adj[x] for x in L):
                                R.add(w)
                                changed = True
                
                # Count edges covered
                count = 0
                for uu in L:
                    for vv in R:
                        key = (min(uu,vv), max(uu,vv))
                        if key in remaining:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_L = L
                    best_R = R
            
            if best_L is None or best_count == 0:
                # Fallback: single edge
                u, v = next(iter(remaining))
                bicliques.append([list([u]), list([v])])
                remaining.discard((min(u,v), max(u,v)))
                continue
            
            L_list = list(best_L)
            R_list = list(best_R)
            
            # Only include edges that are in remaining
            covered = set()
            for u in L_list:
                for v in R_list:
                    key = (min(u,v), max(u,v))
                    if key in remaining:
                        covered.add(key)
            
            if covered:
                bicliques.append([L_list, R_list])
                remaining -= covered
            
            if not remaining:
                break
        
        # Handle leftover edges
        for u, v in remaining:
            bicliques.append([[u], [v]])
        
        return bicliques
    
    def local_search(bicliques, max_iters=200):
        """Try to merge bicliques."""
        bicliques = copy.deepcopy(bicliques)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            merged = False
            for i in range(len(bicliques)):
                if merged:
                    break
                for j in range(i+1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    # Try: L = L1∪L2, R = R1∩R2
                    new_L = list(set(L1) | set(L2))
                    new_R = list(set(R1) & set(R2))
                    
                    if new_R:
                        # Check if it's a valid biclique
                        valid = True
                        for u in new_L:
                            for v in new_R:
                                if v not in adj[u]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            # Check edge coverage matches
                            edges1 = get_biclique_edges(L1, R1)
                            edges2 = get_biclique_edges(L2, R2)
                            new_edges = get_biclique_edges(new_L, new_R)
                            
                            if new_edges == edges1 | edges2:
                                bicliques.pop(j)
                                bicliques.pop(i)
                                bicliques.append([new_L, new_R])
                                merged = True
                                break
                    
                    # Try: L = L1∩L2, R = R2∪R1
                    new_L2 = list(set(L1) & set(L2))
                    new_R2 = list(set(R1) | set(R2))
                    
                    if new_L2:
                        valid = True
                        for u in new_L2:
                            for v in new_R2:
                                if v not in adj[u]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            edges1 = get_biclique_edges(L1, R1)
                            edges2 = get_biclique_edges(L2, R2)
                            new_edges = get_biclique_edges(new_L2, new_R2)
                            
                            if new_edges == edges1 | edges2:
                                bicliques.pop(j)
                                bicliques.pop(i)
                                bicliques.append([new_L2, new_R2])
                                merged = True
                                break
        
        return bicliques
    
    # Run multiple greedy attempts with different seeds
    best_result = None
    best_len = float('inf')
    
    for seed in range(20):
        if time.time() > deadline:
            break
        
        bicliques = greedy_partition(seed)
        
        if not is_valid_partition(bicliques):
            continue
        
        if len(bicliques) < best_len:
            best_len = len(bicliques)
            best_result = bicliques
        
        # Local search
        if time.time() < deadline:
            improved = local_search(bicliques)
            if is_valid_partition(improved) and len(improved) < best_len:
                best_len = len(improved)
                best_result = improved
    
    # Fallback: star decomposition
    if best_result is None:
        best_result = [[[u], [v for a,v in P['edges'] if a==u]] for u in range(n) if any(a==u for a,v in P['edges'])]
    
    # Ensure valid types
    result = []
    for L, R in best_result:
        L = [int(x) for x in L]
        R = [int(x) for x in R]
        if L and R:
            result.append([L, R])
    
    return result
# EVOLVE-BLOCK-END
