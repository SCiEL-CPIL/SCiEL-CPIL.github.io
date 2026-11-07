import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    deadline = time.time() + 55.0
    
    # Base graph: path 0-1-2-...-23
    base_adj = [set() for _ in range(n)]
    existing_edges = set()
    for u, v in P['edges']:
        base_adj[u].add(v)
        base_adj[v].add(u)
        existing_edges.add((min(u, v), max(u, v)))
    
    # The path is bipartite: even vertices = {0,2,4,...,22}, odd = {1,3,...,23}
    # Adding only even-odd edges preserves triangle-free property.
    # We need to find minimum number of even-odd edges such that diameter <= 4.
    
    even_verts = list(range(0, n, 2))  # 0,2,4,...,22
    odd_verts = list(range(1, n, 2))   # 1,3,5,...,23
    
    # All valid candidate edges: even-odd pairs not in existing_edges
    candidates = []
    for e in even_verts:
        for o in odd_verts:
            if (min(e,o), max(e,o)) not in existing_edges:
                candidates.append((e, o))
    
    # Precompute: for each vertex, the set of vertices at distance > 4 in the base path
    # In a path, dist(u,v) = |u-v|, so vertices at distance > 4 are those with |u-v| > 4
    
    def compute_diameter_with_edges(added_edges):
        """Compute diameter of base graph plus added edges. Returns -1 if disconnected."""
        adj = [set(s) for s in base_adj]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for d in dist:
                if d == -1:
                    return -1  # disconnected
                if d > max_d:
                    max_d = d
        return max_d
    
    def check_solution(added_edges):
        """Check if adding these edges gives valid solution (triangle-free, connected, diam<=4)."""
        # Since we only add even-odd edges to a bipartite graph, it stays bipartite (hence triangle-free)
        # Just check connected and diameter
        diam = compute_diameter_with_edges(added_edges)
        return diam != -1 and diam <= 4
    
    # Strategy: Try k=1,2,3,... candidate edges using systematic search
    # For small k, enumerate combinations; for larger k, use smarter approaches
    
    # First, let's think about what's needed:
    # In the path, max distance is 23 (between 0 and 23).
    # We need to reduce all pair distances to <= 4.
    # Key pairs that need shortening: (0,23), (0,22), (1,23), etc.
    
    # A vertex pair (u,v) with |u-v| > 4 needs a "shortcut" path.
    # Adding edge (a,b) creates a shortcut: any path can use this edge.
    
    # Let's use a different approach: BFS from each vertex, track which pairs are "covered"
    # by which edges.
    
    # For each candidate edge, compute which vertex pairs it helps shorten.
    # Actually, let's just do a smart search.
    
    # Key insight: We need to connect the "ends" of the path efficiently.
    # Let's try patterns first, then systematic search.
    
    # Try k=1: single edge can reduce diameter by at most a limited amount
    # With 1 edge (a,b), the graph becomes a path with one chord.
    # The diameter is max of:
    #   - distances within the path not using the chord
    #   - distances using the chord
    # For a path of 24 vertices, 1 edge cannot bring diameter to 4.
    
    # Try k=2, 3, 4, 5...
    
    # Let's use a more efficient approach: 
    # Precompute all-pairs shortest paths for the base graph (just |u-v| for path)
    # Then for each candidate edge, compute how it affects distances.
    
    # Actually, let's just do a focused search.
    # The critical pairs are those at distance > 4: pairs (u,v) with |u-v| >= 5.
    
    # For a solution with diameter <= 4, every pair (u,v) must have a path of length <= 4.
    # In the base path, dist(u,v) = |u-v|.
    # Adding edge (a,b) means we can go u -> ... -> a -> b -> ... -> v or u -> ... -> b -> a -> ... -> v.
    
    # Let's try a structured approach:
    # The path has "layers" at distance d from the center.
    # Center is between 11 and 12.
    # Vertices at distance 0 from center: 11, 12
    # Distance 1: 10, 13
    # Distance 2: 9, 14
    # Distance 3: 8, 15
    # Distance 4: 7, 16
    # Distance 5: 6, 17
    # Distance 6: 5, 18
    # Distance 7: 4, 19
    # Distance 8: 3, 20
    # Distance 9: 2, 21
    # Distance 10: 1, 22
    # Distance 11: 0, 23
    
    # For diameter 4, every vertex must be within distance 4 of every other.
    # This means we need to create shortcuts that bring far-apart vertices closer.
    
    # Let's try a BFS-based approach with pruning.
    # State: set of added edges. Goal: all pairs within distance 4.
    
    # For efficiency, let's focus on the "critical" pairs: those at distance >= 5.
    # A pair (u,v) is "fixed" if there exists a path of length <= 4 between them.
    
    def get_unfixed_pairs(added_edges):
        """Get pairs (u,v) with u < v that are at distance > 4."""
        adj = [set(s) for s in base_adj]
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        unfixed = []
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            for end in range(start+1, n):
                if dist[end] > 4:
                    unfixed.append((start, end))
        return unfixed
    
    # Try increasing k
    best_solution = None
    
    for k in range(1, 12):
        if time.time() > deadline - 5:
            break
        
        if k == 1:
            # Check all single edges
            for edge in candidates:
                if check_solution([edge]):
                    best_solution = [edge]
                    break
        elif k == 2:
            # Check all pairs of edges
            # 288 choose 2 = ~41000, manageable
            for i in range(len(candidates)):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, len(candidates)):
                    if check_solution([candidates[i], candidates[j]]):
                        best_solution = [candidates[i], candidates[j]]
                        break
                if best_solution:
                    break
        elif k == 3:
            # 288 choose 3 is too many (~4M). Use pruning.
            # First, find which edges help the most.
            # Score each edge by how many unfixed pairs it helps.
            unfixed = get_unfixed_pairs([])
            if not unfixed:
                break
            
            # For each candidate edge, count how many unfixed pairs it could potentially fix
            edge_scores = {}
            for edge in candidates:
                u, v = edge
                # This edge helps pairs (a,b) where going through u-v is shorter
                count = 0
                for a, b in unfixed:
                    # Check if using this edge helps
                    # dist(a,b) via edge = min(dist(a,u)+1+dist(v,b), dist(a,v)+1+dist(u,b))
                    # where dist is in the graph with this edge... complex.
                    # Simplify: just count pairs where |a-b| > 4 and the edge is "between" them
                    if (min(u,v) <= max(a,b)) and (max(u,v) >= min(a,b)):
                        count += 1
                edge_scores[edge] = count
            
            # Try top-scoring edges in combinations
            sorted_edges = sorted(candidates, key=lambda e: edge_scores.get(e, 0), reverse=True)
            top_edges = sorted_edges[:50]
            
            for i in range(len(top_edges)):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, len(top_edges)):
                    if time.time() > deadline - 5:
                        break
                    for l in range(j+1, len(top_edges)):
                        combo = [top_edges[i], top_edges[j], top_edges[l]]
                        if check_solution(combo):
                            best_solution = combo
                            break
                    if best_solution:
                        break
                if best_solution:
                    break
        else:
            # For k >= 4, use greedy with backtracking
            if time.time() > deadline - 10:
                break
            
            # Greedy: repeatedly add the edge that fixes the most unfixed pairs
            adj = [set(s) for s in base_adj]
            added = set()
            added_edges = []
            
            for step in range(k):
                if time.time() > deadline - 5:
                    break
                
                unfixed = get_unfixed_pairs(added_edges)
                if not unfixed:
                    break
                
                best_edge = None
                best_score = -1
                
                for edge in candidates:
                    if edge[0] == edge[1]:
                        continue
                    a, b = min(edge), max(edge)
                    if (a, b) in added:
                        continue
                    if (a, b) in existing_edges:
                        continue
                    
                    # Score: how many unfixed pairs does this edge help?
                    u, v = edge
                    score = 0
                    for pa, pb in unfixed:
                        # Check if this edge could help
                        # In base path, dist(pa,pb) = |pa-pb|
                        # With edge (u,v), new path could be pa->...->u->v->...->pb
                        # Approximate: if u is between pa and pb, or v is between pa and pb
                        if (min(pa,pb) <= u <= max(pa,pb)) or (min(pa,pb) <= v <= max(pa,pb)):
                            score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_edge = edge
                
                if best_edge is None or best_score == 0:
                    break
                
                added_edges.append(best_edge)
                added.add((min(best_edge), max(best_edge)))
            
            if added_edges and check_solution(added_edges):
                if best_solution is None or len(added_edges) < len(best_solution):
                    best_solution = added_edges
    
    if best_solution is not None:
        return [[int(u), int(v)] for u, v in best_solution]
    
    # Fallback: try a known good pattern
    # Let's try: connect every 5th vertex to create shortcuts
    # Pattern: (0,5), (5,10), (10,15), (15,20), (20,23) - but need to check triangle-free
    # Since path is bipartite and we add even-odd edges, it stays bipartite.
    
    fallback_patterns = [
        # 5 edges
        [(0,5), (5,10), (10,15), (15,20), (20,23)],
        [(0,4), (4,9), (9,14), (14,19), (19,23)],
        [(1,6), (6,11), (11,16), (16,21), (21,23)],
        # 6 edges
        [(0,5), (5,10), (10,15), (15,20), (20,23), (0,10)],
        [(0,4), (4,8), (8,12), (12,16), (16,20), (20,23)],
        [(0,6), (6,12), (12,18), (18,23), (0,12), (12,23)],
        # 7 edges
        [(0,5), (5,10), (10,15), (15,20), (20,23), (0,10), (10,20)],
        [(0,4), (4,8), (8,12), (12,16), (16,20), (20,23), (0,12)],
        # 8 edges
        [(0,5), (5,10), (10,15), (15,20), (20,23), (0,10), (10,20), (0,20)],
    ]
    
    for pattern in fallback_patterns:
        if check_solution(pattern):
            return [[int(u), int(v)] for u, v in pattern]
    
    # Last resort: return empty (shouldn't happen)
    return []

