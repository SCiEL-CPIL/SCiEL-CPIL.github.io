import time
import random
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = P['n']
    edges = P['edges']
    
    # Preprocess: Adjacency sets for fast intersection checks
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Edge set for O(1) lookup
    edge_set = set((min(u,v), max(u,v)) for u,v in edges)
    
    def is_biclique(L, R):
        # Check if all pairs in L x R are edges
        # Optimization: if L or R is empty, it's a trivial biclique (but we require non-empty)
        if not L or not R:
            return False
        for u in L:
            for v in R:
                if (min(u,v), max(u,v)) not in edge_set:
                    return False
        return True

    def grow_biclique(init_L, init_R):
        # Grow L and R to maximality
        L = set(init_L)
        R = set(init_R)
        
        # Ensure it's a valid biclique to start? 
        # The init from an edge (u,v) is always valid.
        # If we start from arbitrary sets, we should prune invalid parts first.
        # But here we start from edges or stars, which are valid.
        
        changed = True
        while changed:
            changed = False
            
            # Find vertices that can be added to L
            # A vertex u can be added to L if it is connected to ALL vertices in R
            candidates_L = set()
            for u in range(n):
                if u in L:
                    continue
                # Check if R is subset of adj[u]
                if R.issubset(adj[u]):
                    candidates_L.add(u)
            
            # Find vertices that can be added to R
            # A vertex v can be added to R if it is connected to ALL vertices in L
            candidates_R = set()
            for v in range(n):
                if v in R:
                    continue
                # Check if L is subset of adj[v]
                if L.issubset(adj[v]):
                    candidates_R.add(v)
            
            if candidates_L:
                # Add all candidates to L to reach maximality faster
                L.update(candidates_L)
                changed = True
            if candidates_R:
                R.update(candidates_R)
                changed = True
                
        return L, R

    # Generate Candidate Bicliques
    candidates = []
    
    # 1. Grow from each edge
    for u, v in edges:
        L, R = grow_biclique({u}, {v})
        if L and R:
            candidates.append((L, R))
            
    # 2. Grow from "Stars" - Start with L={u}, R=adj[u]
    # This might find large bicliques where one side is a star center
    for u in range(n):
        if adj[u]:
            L, R = grow_biclique({u}, adj[u])
            if L and R:
                candidates.append((L, R))
            
    # 3. Grow from random pairs (optional, if time permits)
    # Since n is small, we can try more random seeds
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline - 5: break
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if (min(u,v), max(u,v)) not in edge_set: continue
        L, R = grow_biclique({u}, {v})
        if L and R:
            candidates.append((L, R))

    # Deduplicate candidates (frozenset of L and R)
    unique_candidates = set()
    for L, R in candidates:
        # Normalize order? No, L and R are distinct sides.
        # But (L,R) and (R,L) are the same biclique.
        # However, in set cover, the identity of the set matters only by the edges it covers.
        # Two different (L,R) representations might cover the same set of edges.
        # Let's just store the set of edges covered by each candidate.
        pass

    # For efficiency in set cover, pre-compute the set of edges covered by each candidate
    # But we need to return the L, R lists.
    # So we keep (L, R, edge_set_covered)
    
    processed_candidates = []
    seen_edge_sets = set()
    for L, R in candidates:
        # Compute covered edges
        covered = set()
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e in edge_set:
                    covered.add(e)
        
        if not covered:
            continue
            
        # Deduplicate by covered edge set
        key = frozenset(covered)
        if key in seen_edge_sets:
            continue
        seen_edge_sets.add(key)
        
        processed_candidates.append((L, R, covered))
        
        if time.time() > deadline - 10:
            break

    # Greedy Set Cover
    uncovered = set(edge_set)
    result_bicliques = []
    
    # Sort candidates by size descending? Greedy usually just picks max gain each step.
    # We can optimize by pre-sorting, but gain changes as uncovered changes.
    
    while uncovered:
        if time.time() > deadline - 2:
            # Fallback: cover remaining edges with stars to ensure validity
            for e in uncovered:
                u, v = e
                result_bicliques.append(([u], [v]))
            break
            
        best_gain = -1
        best_candidate = None
        
        for L, R, covered in processed_candidates:
            # Calculate intersection with uncovered
            gain = len(covered & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_candidate = (L, R, covered)
                
        if best_candidate is None:
            # Should not happen if candidates cover all edges
            for e in uncovered:
                u, v = e
                result_bicliques.append(([u], [v]))
            break
            
        L, R, covered = best_candidate
        result_bicliques.append((L, R))
        uncovered -= covered
        
    # Final check and format
    # Ensure types are list of lists of ints
    final_result = []
    for L, R in result_bicliques:
        L_list = [int(v) for v in L]
        R_list = [int(v) for v in R]
        final_result.append([L_list, R_list])
        
    return final_result

# EVOLVE-BLOCK-END
