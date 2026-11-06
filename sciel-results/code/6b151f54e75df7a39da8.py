import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        remaining = set(edges)
        cliques = []
        
        # Precompute adjacency sets for faster neighbor checks?
        # adj[u] = set of v such that (u,v) in edges
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        while remaining:
            if time.time() > deadline:
                break
            
            # 1. Find vertex v with max degree in 'remaining'
            # We need degree in the *remaining* graph.
            # We can compute this by scanning 'remaining'.
            degrees = [0] * n
            for u, v in remaining:
                degrees[u] += 1
                degrees[v] += 1
            
            # Pick v with max degree
            # If ties, pick smallest index? Or random?
            # Smallest index is deterministic.
            v = 0
            max_deg = -1
            for i in range(n):
                if degrees[i] > max_deg:
                    max_deg = degrees[i]
                    v = i
            
            if max_deg == 0:
                # Should not happen if remaining is not empty
                break

            # 2. Find neighbors of v in 'remaining'
            # These are u such that (min(u,v), max(u,v)) is in remaining
            neighbors = set()
            for u, w in remaining:
                if u == v:
                    neighbors.add(w)
                elif w == v:
                    neighbors.add(u)
            
            # 3. Find a maximal clique in G[neighbors]
            # Greedy: pick u in candidates with max degree in candidates
            C = set()
            candidates = set(neighbors)
            
            while candidates:
                # Find u in candidates with max degree in the subgraph induced by candidates
                # Degree of u = |{w in candidates | (u,w) in edges}|
                # Note: edges is the original graph. We are checking if the edge exists in the original graph.
                # The clique must be a clique in the original graph.
                best_u = None
                best_deg = -1
                
                # Optimization: if len(candidates) is small, this is fast.
                for u in candidates:
                    deg = 0
                    # Count neighbors of u in candidates
                    # We can iterate candidates and check adjacency
                    # Or iterate adj[u] and check if in candidates
                    # adj[u] is a set. candidates is a set.
                    # len(adj[u] & candidates) is efficient?
                    # Yes, set intersection is O(min(len1, len2)).
                    deg = len(adj[u] & candidates)
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
                
                if best_u is None:
                    break

                C.add(best_u)
                # Update candidates: must be neighbors of best_u in original graph AND in current candidates
                # candidates = candidates & adj[best_u]
                candidates = candidates & adj[best_u]
            
            # 4. Form clique
            clique_vertices = C | {v}
            # Convert to list
            clique_list = list(clique_vertices)
            
            # 5. Remove edges
            # Iterate pairs in clique_list
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    e = (min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j]))
                    if e in remaining:
                        remaining.remove(e)
            
            cliques.append(clique_list)
            
        return cliques

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
