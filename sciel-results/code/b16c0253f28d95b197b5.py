import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = generate_graph(n, p, seed)
        if not edges:
            return []
        
        # Adjacency list for fast lookup
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        remaining_edges = set(edges)
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: return individual edges for the rest to ensure validity
                # This is unlikely to happen with the greedy approach on these sizes
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # 1. Select a seed edge
            # Heuristic: Maximize the number of common neighbors (triangles)
            best_edge = None
            max_common = -1
            
            # To speed up, we can sample or iterate. With n=64, iterating is fine.
            for u, v in remaining_edges:
                # Count common neighbors in remaining graph
                # common = adj[u] & adj[v]
                # But we only care about edges in remaining_edges.
                # Actually, if (u,v) is in remaining, and (u,w) in remaining and (v,w) in remaining, then {u,v,w} is a triangle.
                # Let's just count |adj[u] \cap adj[v]|.
                # Since we remove edges, adj should reflect remaining graph?
                # It's easier to maintain a set of remaining edges and check membership.
                
                # Optimization: Pre-compute or just iterate.
                # For n=64, |E| ~ 1000. Iterating 1000 edges * 64 neighbors is 64k ops.
                # This is done ~100-200 times. Total ~10M ops. Very fast.
                
                common = 0
                # We can just intersect the adjacency sets
                # But adj needs to be updated as edges are removed.
                # Let's maintain adj dynamically.
                
                pass # Placeholder, logic moved below
                
            # Dynamic adjacency update is needed.
            # Let's restructure to update adj after removing a clique.
            
            # Re-implement loop with dynamic adj
            
            # Pick edge with max common neighbors
            best_edge = None
            max_common = -1
            for u, v in remaining_edges:
                # Intersection of neighbors
                # To avoid creating large sets, just count
                common = len(adj[u] & adj[v])
                if common > max_common:
                    max_common = common
                    best_edge = (u, v)
            
            if best_edge is None:
                break
                
            u, v = best_edge
            clique = [u, v]
            
            # 2. Expand clique greedily
            # Candidates are vertices connected to all in clique
            # Start with candidates = adj[u] & adj[v]
            candidates = adj[u] & adj[v]
            
            while candidates:
                # Pick candidate with highest degree in remaining graph
                # Degree in remaining graph is len(adj[x])
                # We want to add a vertex that connects to many others to keep future cliques large?
                # Or just pick one that allows expansion.
                # Heuristic: Pick vertex with max degree in remaining graph.
                best_candidate = max(candidates, key=lambda x: len(adj[x]))
                
                # Check if best_candidate is connected to all in clique
                # It is, by construction of candidates intersection?
                # Initially candidates = N(u) \cap N(v).
                # If we add w, new candidates = candidates \ {w} \cap N(w).
                
                clique.append(best_candidate)
                
                # Update candidates
                # New candidates must be connected to best_candidate
                candidates = candidates & adj[best_candidate]
                candidates.discard(best_candidate)
                
            # 3. Remove edges in clique
            clique_set = set(clique)
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if a > b:
                        a, b = b, a
                    e = (a, b)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        adj[a].remove(b)
                        adj[b].remove(a)
                        
            cliques.append(clique)
            
        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Emergency fallback for second instance if first took too long
            # But we should be well within time.
            pass
        results.append(solve_instance(n, p, seed))
        
    return results
