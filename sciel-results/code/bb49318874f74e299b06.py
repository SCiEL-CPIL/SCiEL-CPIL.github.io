def construct():
    from itertools import combinations
    
    def generate_edges(n, p, seed):
        from random import Random
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def get_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def greedy_clique_partition(n, edges):
        if not edges:
            return []
        
        adj = get_adjacency(n, edges)
        remaining_edges = set(edges)
        cliques = []
        
        # Find a good clique greedily
        def find_large_clique(remaining_edges, adj, n):
            if not remaining_edges:
                return []
            
            # Pick vertex with highest degree in remaining graph
            deg = {v: 0 for v in range(n)}
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Find vertex with max degree
            max_deg = max(deg.values())
            candidates = [v for v in range(n) if deg[v] == max_deg]
            start = candidates[0]
            
            # Greedily build clique starting from start
            clique = [start]
            # Neighbors of start in remaining graph
            neighbors = set(adj[start]) & set(v for e in remaining_edges for v in e if v != start)
            
            while neighbors:
                # Pick neighbor with max common neighbors in current clique
                best = None
                best_score = -1
                for v in neighbors:
                    # Check if v can join clique: v must be adjacent to all in clique
                    if all(w in adj[v] for w in clique):
                        # Score by degree in remaining
                        score = deg[v]
                        if score > best_score:
                            best_score = score
                            best = v
                if best is None:
                    break
                clique.append(best)
                # Update neighbors: must be adjacent to all in clique
                neighbors = {v for v in neighbors if all(w in adj[v] for w in clique)}
            
            return clique
        
        while remaining_edges:
            clique = find_large_clique(remaining_edges, adj, n)
            if not clique or len(clique) < 2:
                # Take a single edge
                edge = next(iter(remaining_edges))
                cliques.append(list(edge))
                remaining_edges.remove(edge)
            else:
                # Check all edges in clique are in remaining_edges
                clique_edges = set(tuple(sorted(pair)) for pair in combinations(clique, 2))
                if clique_edges.issubset(remaining_edges):
                    cliques.append(sorted(clique))
                    remaining_edges -= clique_edges
                else:
                    # Fallback: take one edge
                    edge = next(iter(remaining_edges))
                    cliques.append(list(edge))
                    remaining_edges.remove(edge)
        
        return cliques
    
    def solve_instance(n, p, seed):
        edges = generate_edges(n, p, seed)
        return greedy_clique_partition(n, edges)
    
    return [solve_instance(n, p, seed) for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602))]
