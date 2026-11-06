import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target n values to try, descending from max
    # We aim for n=47, then 46.
    
    def is_c4_free(n, edges):
        """
        Check if graph is C4-free.
        Returns True if no C4 exists.
        """
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each pair of vertices (u, v), count common neighbors.
        # If common neighbors >= 2, there is a C4.
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u].intersection(adj[v])
                if len(common) > 1:
                    return False
        return True

    def min_degree(n, edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return min(deg)

    def greedy_construct(n_target, seed=42):
        """
        Deterministic greedy construction with C4 avoidance.
        We try to add edges to maximize min degree, ensuring no C4 is formed.
        """
        rng = random.Random(seed)
        
        # Initialize empty graph
        adj = [set() for _ in range(n_target)]
        edges = []
        
        # Generate all possible edges
        all_edges = [(u, v) for u in range(n_target) for v in range(u + 1, n_target)]
        rng.shuffle(all_edges)
        
        # To make it more deterministic and structured, we can prioritize edges
        # between low-degree vertices. But simple shuffle + greedy check is robust.
        
        # We want min degree >= n_target - 39.
        req_deg = n_target - 39
        
        # Greedy: For each edge in shuffled order, check if adding it creates a C4.
        # If not, add it.
        # This is O(E^2 * deg) which is fine for n=47.
        
        def creates_c4(u, v):
            # Check if adding edge (u, v) creates a C4.
            # A C4 is formed if there exists a path of length 3 between u and v.
            # Path: u - x - y - v.
            # This exists if there is an edge (x, y) such that x is neighbor of u and y is neighbor of v.
            # Efficient check:
            # For each neighbor x of u, check if any neighbor y of x is a neighbor of v.
            # i.e., is there y in adj[x] such that y in adj[v]?
            # If yes, then u-x-y-v is a path of length 3. Adding u-v closes the C4.
            
            for x in adj[u]:
                # Check common neighbors of x and v
                # If common neighbors > 0, then there is a path u-x-y-v.
                # Wait, if y is in adj[x] and y in adj[v], then u-x-y-v is a path.
                # Yes.
                if adj[x].intersection(adj[v]):
                    return True
            return False

        for u, v in all_edges:
            if not creates_c4(u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
        
        # Check if min degree requirement is met
        md = min_degree(n_target, edges)
        if md >= req_deg:
            return edges
        return None

    def local_search_improve(n_target, initial_edges, seed=42):
        """
        Start with initial_edges, try to improve min degree by swapping edges or adding edges.
        """
        rng = random.Random(seed)
        adj = [set() for _ in range(n_target)]
        edges = list(initial_edges)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        req_deg = n_target - 39
        deg = [len(adj[i]) for i in range(n_target)]
        
        def creates_c4(u, v):
            for x in adj[u]:
                if adj[x].intersection(adj[v]):
                    return True
            return False

        improved = True
        while improved and min(deg) < req_deg:
            improved = False
            # Find lowest degree vertex
            min_deg = min(deg)
            low_verts = [i for i in range(n_target) if deg[i] == min_deg]
            
            # Try to add an edge incident to a low-degree vertex
            for v in low_verts:
                # Find non-neighbors
                non_neighbors = [u for u in range(n_target) if u != v and u not in adj[v]]
                rng.shuffle(non_neighbors)
                
                for u in non_neighbors:
                    if not creates_c4(u, v):
                        # Add edge
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append((u, v))
                        deg[u] += 1
                        deg[v] += 1
                        improved = True
                        break
                if improved:
                    break
        
        if min(deg) >= req_deg:
            return edges
        return None

    best_result = {'n': 39, 'edges': []}
    
    # Try n=47, 46
    for n_target in [47, 46]:
        if time.time() > deadline:
            break
            
        found = False
        
        # Strategy 1: Greedy construction with multiple seeds
        # Since the graph is sparse and C4-free, greedy should work well.
        # We try a few seeds to find one that achieves the min degree.
        for seed in range(50):
            if time.time() > deadline:
                break
            edges = greedy_construct(n_target, seed)
            if edges is not None:
                best_result = {'n': n_target, 'edges': edges}
                found = True
                break
        
        if found:
            if n_target == 47:
                break
            continue
            
        # Strategy 2: If greedy failed, try starting from a polarity graph subset
        # and doing local search.
        # Generate polarity graph over GF(7) (49 vertices)
        q = 7
        n_poly = q * q
        vertices = [(x, y) for x in range(q) for y in range(q)]
        
        def vid(x, y):
            return x * q + y
            
        edges_poly = []
        for i in range(n_poly):
            x1, y1 = vertices[i]
            for j in range(i + 1, n_poly):
                x2, y2 = vertices[j]
                if (x1 * x2 + y1 * y2) % q == 1:
                    edges_poly.append((i, j))
        
        # Randomly select n_target vertices from n_poly
        for seed in range(50):
            if time.time() > deadline:
                break
            rng = random.Random(seed)
            keep = set(rng.sample(range(n_poly), n_target))
            old_to_new = {old: new for new, old in enumerate(sorted(keep))}
            
            new_edges = []
            for u, v in edges_poly:
                if u in keep and v in keep:
                    new_edges.append((old_to_new[u], old_to_new[v]))
            
            # Check if it's already good
            if min_degree(n_target, new_edges) >= n_target - 39:
                best_result = {'n': n_target, 'edges': new_edges}
                found = True
                break
            
            # Try local search to improve
            improved_edges = local_search_improve(n_target, new_edges, seed)
            if improved_edges is not None:
                best_result = {'n': n_target, 'edges': improved_edges}
                found = True
                break
                
        if found:
            if n_target == 47:
                break
            continue

    return best_result
