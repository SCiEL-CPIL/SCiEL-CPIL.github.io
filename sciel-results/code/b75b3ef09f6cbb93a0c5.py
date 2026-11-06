"""
Improved program for Clique Partitions of Dense Random Graphs.

Strategy:
1. Generate the graphs as specified.
2. Use a multi-start greedy set-cover heuristic to partition edges into cliques.
   - The core heuristic iteratively finds a large clique in the remaining graph.
   - To escape local optima, we run the greedy process multiple times with
     randomized tie-breaking and vertex ordering.
   - We keep the partition with the minimum total number of cliques.
   - Optimized clique finding using degree-based greedy expansion.
"""
import time

def construct():
    from random import Random
    import collections
    import itertools
    import time

    def generate_edges(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_large_clique(adj, active_vertices, rng):
        """
        Finds a large clique in the subgraph induced by active_vertices using adj.
        Uses a randomized greedy heuristic to explore different cliques.
        """
        if not active_vertices:
            return []
        
        best_clique = []
        best_size = 0
        
        # Shuffle active vertices to randomize start points
        active_list = list(active_vertices)
        rng.shuffle(active_list)
        
        # Limit the number of start vertices to try to save time if graph is large
        # For N=64, trying all is fine, but we can cap it if needed.
        # Let's try all, but break early if we find a clique of size N (perfect).
        
        for start in active_list:
            clique = [start]
            candidates = set(adj[start])
            candidates.intersection_update(active_vertices)
            
            while candidates:
                # Pick vertex with max degree in candidates
                # To add randomness, we can pick randomly from top-k if degrees are close,
                # or just use a simple random tie-breaker.
                max_deg = -1
                top_candidates = []
                for v in candidates:
                    deg = 0
                    for u in adj[v]:
                        if u in candidates:
                            deg += 1
                    if deg > max_deg:
                        max_deg = deg
                        top_candidates = [v]
                    elif deg == max_deg:
                        top_candidates.append(v)
                
                if not top_candidates:
                    break
                    
                # Randomly select from top candidates to diversify
                next_v = rng.choice(top_candidates)
                
                clique.append(next_v)
                # Update candidates to be intersection of current candidates and neighbors of next_v
                new_candidates = set()
                for u in adj[next_v]:
                    if u in candidates:
                        new_candidates.add(u)
                candidates = new_candidates
                
            if len(clique) > best_size:
                best_size = len(clique)
                best_clique = clique
                
                # Early termination if we found a maximal possible clique?
                # Hard to know max possible without solving Max Clique.
                # But if best_size is close to len(active_vertices), we might stop.
                if best_size >= len(active_vertices) - 1:
                    break
                    
        return best_clique

    def solve_instance(n, edges, rng):
        remaining_edges = set(edges)
        cliques = []
        
        while remaining_edges:
            # Build adjacency set for the remaining edges
            adj = collections.defaultdict(set)
            for u, v in remaining_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
                
            best_clique = find_large_clique(adj, active_vertices, rng)
            
            if best_clique and len(best_clique) >= 2:
                cliques.append(best_clique)
                # Remove all edges within best_clique from remaining_edges
                to_remove = set()
                for u, v in itertools.combinations(best_clique, 2):
                    if u > v:
                        u, v = v, u
                    to_remove.add((u, v))
                remaining_edges -= to_remove
            else:
                # Fallback: pick an arbitrary edge
                if remaining_edges:
                    u, v = next(iter(remaining_edges))
                    cliques.append([u, v])
                    remaining_edges.discard((u, v))
                else:
                    break
                    
        return cliques

    def multi_start_solve(n, edges, num_starts=20, base_seed=42):
        best_cliques = None
        best_count = float('inf')
        
        for i in range(num_starts):
            rng = Random(base_seed + i)
            cliques = solve_instance(n, edges, rng)
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques
                
        return best_cliques

    # Instance 1
    n1, p1, seed1 = 32, 0.6, 601
    edges1 = generate_edges(n1, p1, seed1)
    cliques1 = multi_start_solve(n1, edges1, num_starts=50)
    
    # Instance 2
    n2, p2, seed2 = 64, 0.55, 602
    edges2 = generate_edges(n2, p2, seed2)
    cliques2 = multi_start_solve(n2, edges2, num_starts=50)
    
    return [cliques1, cliques2]

if __name__ == "__main__":
    # Quick self-test
    start_time = time.time()
    res = construct()
    end_time = time.time()
    print(f"Instance 1 cliques: {len(res[0])}")
    print(f"Instance 2 cliques: {len(res[1])}")
    print(f"Time: {end_time - start_time:.2f}s")
