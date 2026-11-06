import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s
    
    def has_c4(adj, n):
        """Check if graph with adjacency matrix adj has any C4.
        A C4 exists if there are two vertices with >= 2 common neighbors."""
        # For each pair of vertices, count common neighbors
        # If any pair has >= 2 common neighbors, there's a C4
        # But this is O(n^3) which is too slow for n=47
        # Better: for each pair (u,v), compute |N(u) ∩ N(v)|
        # If >= 2, C4 exists
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    # u and v are adjacent; C4 would need two other vertices
                    # Actually, C4 is u-a-v-b-u where a,b are not u,v
                    # Common neighbors of u and v (excluding each other if adjacent)
                    # If u,v adjacent: C4 needs a,b such that u-a, a-v, v-b, b-u
                    # a is common neighbor of u and v (not v itself)
                    # b is common neighbor of u and v (not u itself)
                    # So we need at least 2 common neighbors of u and v
                    # (since u and v are adjacent, they're not in each other's neighbor set in simple graph... wait)
                    # In simple graph, adj[u,v]=1 means u is neighbor of v.
                    # Common neighbors: vertices w such that adj[u,w]=1 and adj[v,w]=1
                    # If u,v adjacent, w cannot be u or v (simple graph, no self-loops)
                    # So count common neighbors; if >= 2, C4 exists
                    pass
                else:
                    # u,v not adjacent; C4 would be u-a-v-b-u
                    # a is neighbor of both u and v
                    # b is neighbor of both u and v
                    # a != b, a != u,v, b != u,v
                    # So need at least 2 common neighbors
                    pass
        # Actually for any pair (u,v), if |N(u) ∩ N(v)| >= 2, there's a C4
        # (the two common neighbors plus u,v form a C4, whether u,v are adjacent or not)
        # Wait: if u,v adjacent and have common neighbors a,b, then u-a-v-b-u is a C4
        # if u,v not adjacent and have common neighbors a,b, then u-a-v-b-u... 
        #   u-a, a-v, v-b, b-u: yes that's a C4
        # So: C4 exists iff there exist u,v with |N(u) ∩ N(v)| >= 2
        # Let me just check this
        for u in range(n):
            for v in range(u+1, n):
                common = np.sum(adj[u] & adj[v])
                if common >= 2:
                    return True
        return False
    
    def count_c4s(adj, n):
        """Count number of pairs with >= 2 common neighbors (proxy for C4 count)"""
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                common = np.sum(adj[u] & adj[v])
                if common >= 2:
                    count += 1
        return count
    
    def min_degree(adj, n):
        return int(np.min(adj.sum(axis=1)))
    
    def is_valid(adj, n, target_min_deg):
        if min_degree(adj, n) < target_min_deg:
            return False
        if has_c4(adj, n):
            return False
        return True
    
    def greedy_build(n, target_min_deg, seed):
        """Greedy: start empty, add edges that don't create C4, try to reach min degree."""
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=np.uint8)
        
        # Try to add edges in random order, prioritizing vertices with low degree
        for iteration in range(200):  # multiple passes
            if time.time() > deadline:
                break
            # Sort vertices by degree (ascending)
            degrees = adj.sum(axis=1)
            order = list(range(n))
            rng.shuffle(order)
            
            for u in order:
                if degrees[u] >= target_min_deg:
                    continue
                # Try to add edges from u to vertices with low degree
                candidates = [v for v in range(n) if v != u and not adj[u, v] and degrees[v] < target_min_deg]
                rng.shuffle(candidates)
                for v in candidates:
                    if degrees[u] >= target_min_deg:
                        break
                    # Check if adding (u,v) creates a C4
                    # Adding (u,v) creates C4 if there exist a,b with 
                    # u-a, a-v, v-b, b-u (new edge u-v)
                    # i.e., a is neighbor of u and v, b is neighbor of u and v
                    # After adding u-v, common neighbors of u and v include a,b
                    # Actually, adding edge (u,v): C4 forms if there are two vertices
                    # a,b such that a is adj to u and v, b is adj to u and v
                    # (a != b, a != u,v, b != u,v)
                    # So count current common neighbors of u and v
                    common = np.sum(adj[u] & adj[v])
                    if common < 2:
                        adj[u, v] = 1
                        adj[v, u] = 1
                        degrees[u] += 1
                        degrees[v] += 1
        
        return adj
    
    def local_repair(adj, n, target_min_deg, seed):
        """Try to fix C4s by removing edges, maintaining min degree."""
        rng = random.Random(seed)
        adj = adj.copy()
        degrees = adj.sum(axis=1)
        
        for iteration in range(500):
            if time.time() > deadline:
                break
            # Find a C4
            c4_found = False
            for u in range(n):
                for v in range(u+1, n):
                    common = np.sum(adj[u] & adj[v])
                    if common >= 2:
                        # Find the common neighbors
                        common_neighbors = np.where(adj[u] & adj[v])[0]
                        # Remove an edge to break the C4
                        # Options: remove (u,a), (v,a), (u,b), (v,b) for a,b in common
                        # Try removing an edge that doesn't violate min degree
                        a, b = common_neighbors[0], common_neighbors[1]
                        options = [(u,a), (v,a), (u,b), (v,b), (u,v) if adj[u,v] else None]
                        options = [o for o in options if o is not None]
                        rng.shuffle(options)
                        removed = False
                        for (x, y) in options:
                            if degrees[x] > target_min_deg and degrees[y] > target_min_deg:
                                adj[x, y] = 0
                                adj[y, x] = 0
                                degrees[x] -= 1
                                degrees[y] -= 1
                                removed = True
                                c4_found = True
                                break
                        if removed:
                            break
                if c4_found:
                    break
            
            if not c4_found:
                # No C4 found, check validity
                if min_degree(adj, n) >= target_min_deg:
                    break
                # Need to add edges to increase min degree
                # Find vertex with min degree
                min_v = int(np.argmin(degrees))
                if degrees[min_v] < target_min_deg:
                    # Try to add edges from min_v
                    candidates = [v for v in range(n) if v != min_v and not adj[min_v, v]]
                    rng.shuffle(candidates)
                    for v in candidates:
                        if degrees[min_v] >= target_min_deg:
                            break
                        common = np.sum(adj[min_v] & adj[v])
                        if common < 2:
                            adj[min_v, v] = 1
                            adj[v, min_v] = 1
                            degrees[min_v] += 1
                            degrees[v] += 1
        
        return adj
    
    best_result = {'n': 39, 'edges': []}
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
        target_min_deg = n - 39
        if target_min_deg < 0:
            target_min_deg = 0
        
        found = False
        for seed in range(100):
            if time.time() > deadline:
                break
            if found:
                break
            
            # Greedy build
            adj = greedy_build(n, target_min_deg, seed)
            
            # Check if valid
            if min_degree(adj, n) >= target_min_deg and not has_c4(adj, n):
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u, v]:
                            edges.append([u, v])
                best_result = {'n': n, 'edges': edges}
                found = True
                break
            
            # Try local repair
            adj = local_repair(adj, n, target_min_deg, seed)
            
            if min_degree(adj, n) >= target_min_deg and not has_c4(adj, n):
                edges = []
                for u in range(n):
                    for v in range(u+1, n):
                        if adj[u, v]:
                            edges.append([u, v])
                best_result = {'n': n, 'edges': edges}
                found = True
                break
    
    return best_result
