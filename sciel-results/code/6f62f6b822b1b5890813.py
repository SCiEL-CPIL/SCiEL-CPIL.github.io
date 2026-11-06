import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def bron_kerbosch_max_clique(vertices, adj, deadline, time_budget):
        """Find a large clique using Bron-Kerbosch with pivot, time-limited."""
        if time.time() > deadline:
            return set()
        
        # vertices is a frozenset of candidate vertices
        # adj maps vertex -> set of neighbors
        # We want to find a large clique within the subgraph induced by vertices
        
        best_clique = set()
        
        def bk(R, P, X):
            nonlocal best_clique
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = set(R)
                return
            if not P:
                return
            
            # Choose pivot: vertex in P ∪ X with max degree in P
            pivot = None
            max_deg = -1
            for v in list(P) + list(X):
                deg = len(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    pivot = v
            
            # Branch on P \ N(pivot)
            candidates = P - adj[pivot] if pivot is not None else P
            
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bk(new_R, new_P, new_X)
                P.discard(v)
                X.add(v)
                if time.time() > deadline:
                    return
        
        # Initial call
        P = set(vertices)
        X = set()
        bk(set(), P, X)
        
        return best_clique

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets
        adj = {u: set() for u in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(edges)
        partition = []
        
        while remaining_edges and time.time() < deadline:
            # Find an edge to start from - use most constrained edge
            # (edge with fewest common neighbors in remaining graph)
            best_edge = None
            min_common = float('inf')
            
            for e in list(remaining_edges):
                u, v = e
                common = adj[u] & adj[v]
                # Count common neighbors that still have both edges remaining
                common_count = 0
                for w in common:
                    eu = (min(u,w), max(u,w))
                    ev = (min(v,w), max(v,w))
                    if eu in remaining_edges and ev in remaining_edges:
                        common_count += 1
                
                if common_count < min_common:
                    min_common = common_count
                    best_edge = e
                if time.time() > deadline:
                    break
            
            if best_edge is None:
                break
            
            u, v = best_edge
            
            # Build the subgraph of vertices that can join {u,v}
            # A vertex w can join if edges (u,w) and (v,w) are in remaining_edges
            candidates = set()
            for w in adj[u]:
                if w == v:
                    continue
                eu = (min(u,w), max(u,w))
                ev = (min(v,w), max(v,w))
                if eu in remaining_edges and ev in remaining_edges:
                    candidates.add(w)
            
            # Find max clique in the subgraph induced by {u, v} ∪ candidates
            # where all edges must be in remaining_edges
            all_verts = {u, v} | candidates
            
            # Build local adjacency restricted to remaining edges
            local_adj = {}
            for x in all_verts:
                local_adj[x] = set()
            for x in all_verts:
                for y in adj[x]:
                    if y in all_verts:
                        e = (min(x,y), max(x,y))
                        if e in remaining_edges:
                            local_adj[x].add(y)
            
            # Use Bron-Kerbosch to find a large clique
            clique = bron_kerbosch_max_clique(all_verts, local_adj, deadline, time_budget)
            
            # Ensure u and v are in the clique (they should be, but verify)
            if u not in clique or v not in clique:
                # Fallback: just use {u, v}
                clique = {u, v}
            
            if len(clique) < 2:
                clique = {u, v}
            
            clique_list = sorted(clique)
            
            # Verify it's a valid clique
            valid = True
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if (a, b) not in remaining_edges:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                clique_list = [u, v]
                clique = {u, v}
            
            partition.append(clique_list)
            
            # Remove all edges within this clique
            to_remove = set()
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    to_remove.add((a, b))
            remaining_edges -= to_remove
        
        # Fallback for remaining edges
        if remaining_edges:
            for u, v in remaining_edges:
                partition.append([u, v])
        
        return partition

    result1 = solve_instance(32, 0.6, 601, 60)
    result2 = solve_instance(64, 0.55, 602, 65)
    
    return [result1, result2]
