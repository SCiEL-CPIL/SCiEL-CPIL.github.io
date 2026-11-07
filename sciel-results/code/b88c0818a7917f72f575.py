import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Precompute valid bipartite edges (must connect even to odd to preserve triangle-free)
    valid_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edges:
                continue
            if u % 2 != v % 2:
                valid_edges.append((u, v))
    
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    def build_adj(added_edges):
        adj = [set() for _ in range(n)]
        for u, v in base_edges:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def get_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] > max_dist:
                            max_dist = dist[v]
                        if dist[v] > 4:
                            # Early exit if we already exceed diameter
                            return max_dist
                        queue.append(v)
            if -1 in dist:
                return float('inf')
        return max_dist
    
    def is_valid(added_edges):
        adj = build_adj(added_edges)
        diam = get_diameter(adj)
        return diam <= 4
    
    # Strategy: Constructive approach using hub-based initialization
    # We want to find a small set of edges such that the graph has diameter <= 4
    
    # Key insight: If we add edges from a central hub to many vertices, we can reduce diameter.
    # But we need to maintain triangle-free property.
    
    # Let's try a different approach: 
    # 1. Start with a known valid dense solution
    # 2. Iteratively remove edges while maintaining validity
    
    # First, build a dense valid solution
    # Add all edges of the form (i, i+4) and (i, i+6) where parity allows
    # Actually, let's just add enough edges to guarantee diameter 4, then reduce
    
    # A simple valid dense solution: connect every even vertex to every odd vertex
    # This is a complete bipartite graph K(12,12) which has diameter 2
    # But that's too many edges. Let's be smarter.
    
    # Let's use the following strategy:
    # 1. Create a "star-like" structure with a few hubs
    # 2. Ensure all vertices are within distance 2 of a hub
    # 3. Then verify diameter <= 4
    
    # Hubs at positions that cover the path efficiently
    # If we place hubs at positions 3, 7, 11, 15, 19, 23 (or similar), 
    # and connect them to nearby vertices, we can get small diameter
    
    # Actually, let's try a constructive approach:
    # Add edges to create a "ladder" or "grid" structure
    
    # Let me think about this differently.
    # The path has 24 vertices. To get diameter 4, we need to "shorten" the path.
    # If we add an edge (i, j) where j - i = k, we reduce the distance between i and j from k to 1.
    
    # Optimal strategy: Add edges that create multiple "shortcuts" across the path.
    # For example, adding edges (0,4), (1,5), (2,6), ... creates a structure where
    # distances are reduced significantly.
    
    # Let's try: add edges (i, i+4) for all i where i+4 < n and parity matches (always does for even offset? No, 4 is even, so i and i+4 have same parity - NOT valid!)
    # So (i, i+4) is NOT valid because i and i+4 have the same parity.
    
    # Valid shortcuts must have odd length: (i, i+1) is base, (i, i+3), (i, i+5), (i, i+7), etc.
    
    # Let's add edges of the form (i, i+3) and (i, i+5) strategically.
    
    # Constructive approach:
    # 1. Start with a set of candidate edges
    # 2. Use greedy to build a valid solution
    # 3. Use local search to minimize
    
    def greedy_build():
        added_edges = set()
        adj = build_adj(added_edges)
        
        while True:
            if time.time() > deadline:
                break
            
            diam = get_diameter(adj)
            if diam <= 4:
                break
            
            # Find all pairs with distance > 4
            far_pairs = []
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                queue = deque([src])
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            if dist[v] > 4:
                                far_pairs.append((min(src, v), max(src, v)))
                            queue.append(v)
            
            if not far_pairs:
                break
            
            # For each candidate edge, compute how many far pairs it resolves
            best_edge = None
            best_score = -1
            
            for u, v in valid_edges:
                if (u, v) in added_edges:
                    continue
                
                # Temporarily add this edge and count how many far pairs are resolved
                new_set = added_edges | {(u, v)}
                new_adj = build_adj(new_set)
                
                # Count far pairs with new edge
                new_far_count = 0
                for src in range(n):
                    dist = [-1] * n
                    dist[src] = 0
                    queue = deque([src])
                    while queue:
                        a = queue.popleft()
                        for b in new_adj[a]:
                            if dist[b] == -1:
                                dist[b] = dist[a] + 1
                                if dist[b] > 4:
                                    new_far_count += 1
                                queue.append(b)
                
                score = len(far_pairs) - new_far_count
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
                
                # Early termination if we found an edge that makes diameter <= 4
                if new_far_count == 0:
                    best_edge = (u, v)
                    best_score = len(far_pairs)
                    break
            
            if best_edge is None or best_score <= 0:
                # No edge improves the situation, break
                break
            
            added_edges.add(best_edge)
            adj = build_adj(added_edges)
        
        return added_edges
    
    def local_search_reduce(added_edges):
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for edge in list(added_edges):
                if time.time() > deadline:
                    break
                new_set = added_edges - {edge}
                if is_valid(new_set):
                    added_edges = new_set
                    improved = True
                    break
        return added_edges
    
    # Try multiple approaches and pick the best
    
    best_result = None
    best_count = float('inf')
    
    # Approach 1: Greedy build
    res = greedy_build()
    if is_valid(res):
        res = local_search_reduce(res)
        if len(res) < best_count:
            best_count = len(res)
            best_result = res
    
    # Approach 2: Start with a dense valid solution and reduce
    # Dense solution: add all valid edges (this is K(12,12) minus the path edges, which is valid)
    dense_set = set(valid_edges)
    if is_valid(dense_set):
        dense_set = local_search_reduce(dense_set)
        if len(dense_set) < best_count:
            best_count = len(dense_set)
            best_result = dense_set
    
    # Approach 3: Hub-based constructive
    # Place hubs at strategic positions and connect them
    # Hubs at 0, 6, 12, 18 (even positions, spaced 6 apart)
    # Connect each hub to vertices within distance 2-3
    hub_edges = set()
    hubs = [0, 6, 12, 18]
    for h in hubs:
        for v in range(n):
            if v == h:
                continue
            if (h, v) in base_edges:
                continue
            if h % 2 != v % 2:
                # Check if adding this edge helps
                hub_edges.add((min(h, v), max(h, v)))
    
    if is_valid(hub_edges):
        hub_edges = local_search_reduce(hub_edges)
        if len(hub_edges) < best_count:
            best_count = len(hub_edges)
            best_result = hub_edges
    
    # Approach 4: Add edges (i, i+3) for all valid i, then reduce
    edges_i3 = set()
    for i in range(n - 3):
        j = i + 3
        if i % 2 != j % 2:  # Always true for offset 3
            edges_i3.add((i, j))
    
    if is_valid(edges_i3):
        edges_i3 = local_search_reduce(edges_i3)
        if len(edges_i3) < best_count:
            best_count = len(edges_i3)
            best_result = edges_i3
    
    # Approach 5: Add edges (i, i+5) for all valid i, then reduce
    edges_i5 = set()
    for i in range(n - 5):
        j = i + 5
        if i % 2 != j % 2:  # Always true for offset 5
            edges_i5.add((i, j))
    
    if is_valid(edges_i5):
        edges_i5 = local_search_reduce(edges_i5)
        if len(edges_i5) < best_count:
            best_count = len(edges_i5)
            best_result = edges_i5
    
    # Approach 6: Combine i+3 and i+5 edges, then reduce
    edges_combined = set()
    for i in range(n - 3):
        j = i + 3
        if i % 2 != j % 2:
            edges_combined.add((i, j))
    for i in range(n - 5):
        j = i + 5
        if i % 2 != j % 2:
            edges_combined.add((i, j))
    
    if is_valid(edges_combined):
        edges_combined = local_search_reduce(edges_combined)
        if len(edges_combined) < best_count:
            best_count = len(edges_combined)
            best_result = edges_combined
    
    # Approach 7: Randomized greedy
    import random
    random.seed(42)
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        added_edges = set()
        adj = build_adj(added_edges)
        
        while True:
            if time.time() > deadline:
                break
            
            diam = get_diameter(adj)
            if diam <= 4:
                break
            
            far_pairs = []
            for src in range(n):
                dist = [-1] * n
                dist[src] = 0
                queue = deque([src])
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            if dist[v] > 4:
                                far_pairs.append((min(src, v), max(src, v)))
                            queue.append(v)
            
            if not far_pairs:
                break
            
            candidates = [e for e in valid_edges if e not in added_edges]
            if not candidates:
                break
            
            random.shuffle(candidates)
            best_edge = None
            best_score = -1
            
            for u, v in candidates:
                new_set = added_edges | {(u, v)}
                new_adj = build_adj(new_set)
                
                new_far_count = 0
                for src in range(n):
                    dist = [-1] * n
                    dist[src] = 0
                    queue = deque([src])
                    while queue:
                        a = queue.popleft()
                        for b in new_adj[a]:
                            if dist[b] == -1:
                                dist[b] = dist[a] + 1
                                if dist[b] > 4:
                                    new_far_count += 1
                                queue.append(b)
                
                score = len(far_pairs) - new_far_count
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
                
                if new_far_count == 0:
                    best_edge = (u, v)
                    break
            
            if best_edge is None or best_score <= 0:
                break
            
            added_edges.add(best_edge)
            adj = build_adj(added_edges)
        
        if is_valid(added_edges):
            added_edges = local_search_reduce(added_edges)
            if len(added_edges) < best_count:
                best_count = len(added_edges)
                best_result = added_edges
    
    if best_result is None:
        # Absolute fallback: return the dense solution reduced
        dense_set = set(valid_edges)
        dense_set = local_search_reduce(dense_set)
        return [[int(u), int(v)] for u, v in dense_set]
    
    return [[int(u), int(v)] for u, v in best_result]
