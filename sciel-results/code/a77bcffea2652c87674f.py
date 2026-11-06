import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    def is_c4_free(n, adj):
        # adj is a list of sets
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    continue
                # Check if there are >= 2 common neighbors
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return False
        return True

    def find_c4(n, adj):
        """Returns (u, v, w, x) if C4 found, else None.
        u-v-w-x-u. u,v non-adjacent.
        """
        for u in range(n):
            for v in range(u+1, n):
                if v in adj[u]:
                    continue
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    # Get two common neighbors
                    neighbors = list(common)
                    w = neighbors[0]
                    x = neighbors[1]
                    return (u, v, w, x)
        return None

    def make_c4_free(n, adj):
        """Removes edges to ensure no C4. Returns modified adj."""
        # Copy adj
        adj = [set(s) for s in adj]
        changed = True
        while changed:
            changed = False
            c4 = find_c4(n, adj)
            if c4:
                u, v, w, x = c4
                # Remove one edge, e.g., (u, w)
                adj[u].remove(w)
                adj[w].remove(u)
                changed = True
        return adj

    def has_independent_set_of_size_k(n, adj, k):
        """Returns True if there is an independent set of size k.
        Returns the set if found, else None."""
        # Simple backtracking
        # To speed up, order vertices by degree (highest first?)
        # Actually, for independent set, picking high degree vertices first
        # removes many candidates, which is good for pruning if we want to AVOID finding one?
        # No, we want to FIND one to prove it exists.
        # If we want to prove it DOES NOT exist, we want to prune quickly.
        # If we pick a vertex with high degree, we remove many neighbors.
        # If we include it, we lose many options.
        # If we exclude it, we keep options.
        
        # Let's use a standard DFS.
        # State: (current_set, available_vertices)
        # available_vertices: vertices that can be added to current_set (not in set, not neighbors of set)
        
        # Optimization: If len(available) < k - len(current), return None.
        
        def dfs(current, available):
            if len(current) == k:
                return current
            if len(available) < k - len(current):
                return None
            
            # Pick a vertex from available
            # Heuristic: Pick vertex with highest degree in induced subgraph?
            # Or just pick the first one.
            # Let's pick a vertex that maximizes pruning?
            # If we include v, we remove N(v) from available.
            # If we exclude v, we just remove v.
            
            # Let's try picking a vertex with high degree in the remaining graph to see if it helps.
            # Actually, for finding a set of size k, it's often better to pick vertices that are "isolated" in the remaining graph?
            # No, if we pick an isolated vertex, we don't remove neighbors, so the branch is "wider".
            # We want to find a solution.
            # If a solution exists, picking a "good" vertex helps.
            
            # Let's just pick the first available vertex.
            # To improve, we can sort available by degree.
            # Let's try: Pick vertex with highest degree in `available` subgraph?
            # This might prune the "include" branch heavily, forcing "exclude".
            # If the graph is dense, "include" is bad.
            
            v = next(iter(available))
            # Better heuristic: Pick vertex with max degree in induced subgraph of available
            # This is O(n^2) per step, might be slow.
            # Let's stick to simple or pre-sorted order.
            
            # Option 1: Include v
            new_available = available - {v} - adj[v]
            res = dfs(current | {v}, new_available)
            if res:
                return res
            
            # Option 2: Exclude v
            res = dfs(current, available - {v})
            if res:
                return res
            
            return None
        
        return dfs(set(), set(range(n)))

    def try_build(n):
        """Tries to build a valid graph of size n."""
        # Try several random graphs
        for attempt in range(500):
            if time.time() > deadline:
                return None
            
            # Generate random graph G(n, 0.3)
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.3:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Make C4-free
            adj = make_c4_free(n, adj)
            
            # Check alpha
            iso_set = has_independent_set_of_size_k(n, adj, 11)
            
            if not iso_set:
                return adj
            
            # If alpha >= 11, try to repair
            # iso_set is an independent set of size 11
            # We need to add edges within iso_set to break it
            # But adding edges must not create C4s
            repaired = False
            for _ in range(20): # Max repair steps
                if time.time() > deadline:
                    break
                
                # Find a new independent set
                iso_set = has_independent_set_of_size_k(n, adj, 11)
                if not iso_set:
                    return adj
                
                # Try to add an edge within iso_set
                vertices = list(iso_set)
                random.shuffle(vertices)
                added = False
                for i in range(len(vertices)):
                    for j in range(i+1, len(vertices)):
                        u, v = vertices[i], vertices[j]
                        if v in adj[u]:
                            continue
                        
                        # Check if adding (u, v) creates C4
                        # Check if there is a path of length 3 between u and v
                        # i.e., exists w in N(u), x in N(v) such that w ~ x
                        # And w != v, x != u (implied by non-adjacent)
                        
                        # Optimization: 
                        # Common neighbors of u and v must be 0?
                        # If they have common neighbors, adding edge makes triangle? No, C4.
                        # If they have 2 common neighbors, C4 exists already?
                        # Wait, if they have 2 common neighbors, the graph is NOT C4-free.
                        # But we ensured C4-free. So they have at most 1 common neighbor.
                        # If they have 1 common neighbor w, adding edge (u,v) creates triangle u-v-w-u.
                        # Triangles are allowed. C4 is forbidden.
                        # Does adding (u,v) create a NEW C4?
                        # A C4 involving the new edge (u,v) would be u-v-x-y-u.
                        # This requires a path of length 3 from u to v in the OLD graph.
                        # Path u-a-b-v.
                        # a in N(u), b in N(v), a ~ b.
                        
                        creates_c4 = False
                        for a in adj[u]:
                            if a == v: continue
                            for b in adj[v]:
                                if b == u: continue
                                if b in adj[a]:
                                    creates_c4 = True
                                    break
                            if creates_c4:
                                break
                        if not creates_c4:
                            # Safe to add
                            adj[u].add(v)
                            adj[v].add(u)
                            added = True
                            break
                    if added:
                        break
                
                if not added:
                    break # Stuck
            
            if repaired:
                 # Re-check after repair
                 iso_set = has_independent_set_of_size_k(n, adj, 11)
                 if not iso_set:
                     return adj

        return None

    # Iterate n from 43 down to 30
    for n in range(43, 29, -1):
        if time.time() > deadline:
            break
        adj = try_build(n)
        if adj:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
    
    # Fallback to seed
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
