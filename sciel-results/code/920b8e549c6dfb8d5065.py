"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [list() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])
    
    # Precompute length-3 paths (single internal vertex)
    path_3 = [[[] for _ in range(n)] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if v in adj[u]:
                continue
            common = adj[u] & adj[v]
            for w in common:
                if w != u and w != v:
                    path_3[u][v].append(w)
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_result = None
    
    # Try to find the largest clique subdivision
    # Start with a high k and work down
    # The maximum possible k is limited by n
    # For a subdivision of size k, we need k branch vertices and (k choose 2) paths.
    # If paths have length 1 (edges), we need a clique of size k.
    # If paths have length 3 (1 internal vertex), we need k branch vertices and (k choose 2) - (edges in clique) internal vertices.
    
    # Let's first find the maximum clique size to get an upper bound on k if we only used edges.
    # But we can do better with paths.
    
    # Heuristic: Try k from 10 down to 2
    # For each k, try combinations of top degree vertices
    # Use bipartite matching to assign internal vertices for non-adjacent pairs
    
    def find_subdivision(k, candidates):
        # candidates: list of vertices to choose from
        # Returns a valid subdivision or None
        
        # To reduce search space, we can fix some vertices
        # But for now, let's just iterate combinations
        
        for combo in combinations(candidates, k):
            if time.time() > deadline:
                return None
                
            B = list(combo)
            B_set = set(B)
            
            non_adj_pairs = []
            for i in range(k):
                for j in range(i+1, k):
                    u, v = B[i], B[j]
                    if v not in adj[u]:
                        non_adj_pairs.append((u, v))
                        
            # Pigeonhole check
            available_vertices = n - k
            if len(non_adj_pairs) > available_vertices:
                continue
                
            # Build matching problem
            # Left: non_adj_pairs
            # Right: V \ B
            # Edge if w is a valid internal vertex for (u,v)
            
            pair_to_verts = {}
            for idx, (u, v) in enumerate(non_adj_pairs):
                valid_ws = [w for w in path_3[u][v] if w not in B_set]
                pair_to_verts[idx] = valid_ws
                
            # Check if any pair has no options
            possible = True
            for idx in range(len(non_adj_pairs)):
                if not pair_to_verts[idx]:
                    possible = False
                    break
                    
            if not possible:
                continue
                
            # Bipartite Matching (Kuhn's algorithm)
            # match_vertex[v] = pair_idx
            match_vertex = {v: -1 for v in range(n)}
            
            def try_kuhn(v, seen):
                # v is a pair index
                for u in pair_to_verts[v]:
                    if seen[u]:
                        continue
                    seen[u] = True
                    if match_vertex[u] == -1 or try_kuhn(match_vertex[u], seen):
                        match_vertex[u] = v
                        return True
                return False
            
            found_matching = True
            for i in range(len(non_adj_pairs)):
                seen = {v: False for v in range(n)}
                if not try_kuhn(i, seen):
                    found_matching = False
                    break
                    
            if not found_matching:
                continue
                
            # Construct result
            paths = []
            for i in range(k):
                for j in range(i+1, k):
                    u, v = B[i], B[j]
                    if v in adj[u]:
                        paths.append([u, v])
                    else:
                        # Find the matched vertex
                        # We need to find which pair index corresponds to (u,v)
                        # This is inefficient, let's store the mapping
                        pass
            
            # Re-do construction with proper mapping
            pair_idx_map = {}
            for idx, (u, v) in enumerate(non_adj_pairs):
                pair_idx_map[(u, v)] = idx
                pair_idx_map[(v, u)] = idx
            
            paths = []
            for i in range(k):
                for j in range(i+1, k):
                    u, v = B[i], B[j]
                    if v in adj[u]:
                        paths.append([u, v])
                    else:
                        idx = pair_idx_map[(u, v)]
                        # Find w such that match_vertex[w] == idx
                        # This is O(n) per pair, but n is small
                        w = None
                        for vert in range(n):
                            if match_vertex[vert] == idx:
                                w = vert
                                break
                        if w is None:
                            return None # Should not happen
                        paths.append([u, w, v])
                        
            return {'branches': B, 'paths': paths}
            
        return None

    # Strategy:
    # 1. Try to find a large subdivision.
    # 2. Start with high k.
    # 3. Use a pool of high-degree vertices.
    
    # We can also try to find the maximum clique first, as that gives a valid subdivision with length-1 paths.
    # But we can likely do better.
    
    # Let's try k from 10 down to 2
    # Pool size: maybe 15-20
    # If k=10, C(15,10) = 3003, which is manageable.
    # If k=12, C(15,12) = 455.
    # If k=15, C(15,15) = 1.
    
    # Let's try k from 12 down to 2
    # But wait, if we use length-3 paths, we need internal vertices.
    # The number of non-adjacent pairs in a set of k vertices from a dense graph might be small.
    
    # Let's start with k=10
    for k in range(10, 1, -1):
        if time.time() > deadline:
            break
            
        # Use top 15 vertices
        pool_size = min(15, n)
        candidates = sorted_vertices[:pool_size]
        
        result = find_subdivision(k, candidates)
        if result:
            best_result = result
            break
            
    # If no result found with length-3 paths, try to find a clique (length-1 paths)
    if best_result is None:
        # Find max clique
        # Simple backtracking
        def max_clique(vertices):
            if not vertices:
                return []
            best = []
            def backtrack(verts, current):
                nonlocal best
                if len(current) > len(best):
                    best = list(current)
                for i, v in enumerate(verts):
                    current.append(v)
                    backtrack([x for x in verts[i+1:] if x in adj[v]], current)
                    current.pop()
            backtrack(vertices, [])
            return best
            
        clique = max_clique(sorted_vertices[:15])
        if len(clique) >= 2:
            B = clique
            paths = []
            for i in range(len(B)):
                for j in range(i+1, len(B)):
                    paths.append([B[i], B[j]])
            best_result = {'branches': B, 'paths': paths}
        else:
            # Fallback to edge
            u, v = P['edges'][0]
            best_result = {'branches': [u, v], 'paths': [[u, v]]}
            
    # Ensure types are correct
    best_result['branches'] = list(best_result['branches'])
    for i in range(len(best_result['paths'])):
        best_result['paths'][i] = list(best_result['paths'][i])
        
    return best_result
# EVOLVE-BLOCK-END
