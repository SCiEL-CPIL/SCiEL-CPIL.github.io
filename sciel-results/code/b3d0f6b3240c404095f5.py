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

    def bron_kerbosch_max_clique(vertices, adj, time_budget):
        """Find a large clique using Bron-Kerbosch with pivot, time-limited."""
        start_bk = time.time()
        if time.time() > start_bk + time_budget:
            return set()
        
        best_clique = set()
        
        def bk(R, P, X):
            nonlocal best_clique
            if time.time() > start_bk + time_budget:
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
                if time.time() > start_bk + time_budget:
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
            # Find vertices that are part of remaining edges
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
                
            if not active_vertices:
                break

            # Build local adjacency restricted to remaining edges
            local_adj = {v: set() for v in active_vertices}
            for u, v in remaining_edges:
                local_adj[u].add(v)
                local_adj[v].add(u)
            
            # Use small time budget per clique search to ensure progress
            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                break
            # Use up to 1 second or 10% of remaining time, whichever is smaller
            bk_budget = min(1.0, remaining_time * 0.1)
            
            # Find max clique in the subgraph induced by active_vertices
            clique = bron_kerbosch_max_clique(active_vertices, local_adj, bk_budget)
            
            if len(clique) < 2:
                # Fallback: pick an arbitrary edge
                u, v = next(iter(remaining_edges))
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
                u, v = next(iter(remaining_edges))
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
