import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Helper: Check if graph is triangle-free
    # G is a list of sets of neighbors
    def is_triangle_free(G):
        n = len(G)
        for u in range(n):
            for v in G[u]:
                if u < v:
                    # Check if any common neighbor exists
                    # Intersection of G[u] and G[v]
                    # Since we maintain symmetry, G[u] contains v and G[v] contains u
                    # We just need to check if there is w in G[u] \cap G[v]
                    # Optimized: iterate over smaller set
                    if len(G[u]) < len(G[v]):
                        for w in G[u]:
                            if w != v and w in G[v]:
                                return False
                    else:
                        for w in G[v]:
                            if w != u and w in G[u]:
                                return False
        return True

    # Helper: Find an independent set of size k
    # Returns a list of vertices or None
    def find_independent_set(G, k):
        n = len(G)
        # Use a simple backtracking with pruning
        # available: set of vertices that can still be added
        # current: list of vertices in current independent set
        
        # Heuristic: order vertices by degree (ascending) to prune faster?
        # For k=10, simple backtracking is usually fast enough for n=40
        
        # We can use a recursive function
        def backtrack(available, current):
            if len(current) == k:
                return current[:]
            if not available:
                return None
            
            # Pruning: if len(available) + len(current) < k, impossible
            if len(available) + len(current) < k:
                return None
            
            # Pick a vertex from available
            # Randomize order to avoid bias
            v = random.choice(list(available))
            available.remove(v)
            
            # Try including v
            # New available: vertices in available that are not adjacent to v
            new_available = set()
            for u in available:
                if u not in G[v]:
                    new_available.add(u)
            
            res = backtrack(new_available, current + [v])
            if res:
                return res
            
            # Try excluding v (implicitly handled by trying other choices in loop, 
            # but here we just picked one. To be thorough, we should iterate)
            # Actually, the above is not a complete search if we just pick one random v.
            # We need to iterate over all choices or use a better search strategy.
            
            # Let's rewrite backtrack to iterate
            return None
        
        # Better Backtracking:
        def bt(available, current):
            if len(current) == k:
                return current[:]
            if not available:
                return None
            if len(available) + len(current) < k:
                return None
            
            # Pick a pivot vertex to branch on
            # Simplest: pick first available
            # To speed up, we can use a coloring bound, but for k=10 it's overkill
            # Let's just iterate
            verts = list(available)
            random.shuffle(verts)
            
            for v in verts:
                # Include v
                new_available = set()
                for u in available:
                    if u != v and u not in G[v]:
                        new_available.add(u)
                
                res = bt(new_available, current + [v])
                if res:
                    return res
                
                # Exclude v: remove v from available for subsequent iterations
                available.discard(v)
            
            return None
        
        return bt(set(range(len(G))), [])

    def is_valid(G):
        if not is_triangle_free(G):
            return False
        if find_independent_set(G, 10):
            return False
        return True

    # Local Search
    def local_search(N, time_limit):
        # Initialize random graph
        # To have a chance of being triangle-free, start sparse?
        # But we need alpha < 10.
        # Random graph G(N, 0.5) has many triangles.
        # Let's start with a random graph and let the search fix it.
        
        G = [set() for _ in range(N)]
        # Random edges
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.5:
                    G[i].add(j)
                    G[j].add(i)
        
        best_graph = None
        best_score = -1 # 0: invalid, 1: valid
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Check triangles
            triangles_found = False
            # Find a triangle
            # We can just check is_triangle_free, but we need to know WHICH triangle to break
            # Let's find one triangle
            tri = None
            for u in range(N):
                if tri: break
                for v in G[u]:
                    if u < v and tri: break
                    # Check common neighbors
                    # Optimization: if degrees are small, this is fast
                    # For random graph, degrees ~ N/2 = 20.
                    # Intersection size ~ 5.
                    common = G[u].intersection(G[v])
                    if common:
                        w = next(iter(common))
                        tri = (u, v, w)
                        break
            
            if tri:
                # Remove a random edge from the triangle
                u, v, w = tri
                edges_to_break = [(u, v), (v, w), (u, w)]
                a, b = random.choice(edges_to_break)
                G[a].discard(b)
                G[b].discard(a)
                continue
            
            # No triangles. Check independent set
            ind_set = find_independent_set(G, 10)
            if not ind_set:
                # Valid!
                # Convert to list of edges
                edges = []
                for i in range(N):
                    for j in G[i]:
                        if i < j:
                            edges.append([i, j])
                return {'n': N, 'edges': edges}
            else:
                # Add an edge to break the independent set
                # Pick two non-adjacent vertices from ind_set
                # They are non-adjacent by definition of independent set
                a, b = random.sample(ind_set, 2)
                G[a].add(b)
                G[b].add(a)
                
    # Try N from 40 down to 26
    # We know 26 is valid (Andrasfai).
    # We will try 40, 39, 38...
    # Allocate time to larger N first? Or equal?
    # Larger N is harder. Maybe give 40 more time?
    # Let's give 20s to 40, 20s to 39, etc.
    
    for N in range(40, 25, -1):
        if time.time() > deadline - 10:
            break
        time_alloc = min(15, deadline - time.time() - 5)
        if time_alloc <= 0:
            break
        
        res = local_search(N, time_alloc)
        if res:
            return res
    
    # Fallback: Return the seed graph (Andrasfai 26)
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
