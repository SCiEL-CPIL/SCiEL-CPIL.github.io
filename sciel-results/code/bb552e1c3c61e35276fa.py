# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    def random_graph(n, p, seed):
        """Generate random graph edges using the same method as the seed program."""
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve_clique_partition(n, p, seed, time_budget):
        """
        Solve the clique partition problem for a given random graph.
        Strategy:
        1. Generate the graph
        2. Find all triangles
        3. Use a greedy algorithm with triangle packing to minimize clique count
        4. Handle remaining edges as individual cliques (size 2)
        """
        import random
        
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency structure
        adj = defaultdict(set)
        edge_to_idx = {}
        for i, (u, v) in enumerate(edges):
            adj[u].add(v)
            adj[v].add(u)
            edge_to_idx[(u, v)] = i
            edge_to_idx[(v, u)] = i
        
        # Find all triangles
        triangles = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            triangles.append((u, v, w))
        
        # Build edge-to-triangles mapping
        edge_triangles = defaultdict(list)
        for tri in triangles:
            u, v, w = tri
            for e in [(u, v), (v, w), (u, w)]:
                e_sorted = tuple(sorted(e))
                edge_triangles[e_sorted].append(tri)
        
        # Greedy triangle packing with randomization
        best_result = None
        best_count = float('inf')
        
        num_restarts = min(500, max(50, int((time_budget - (time.time() - start_time)) * 10)))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            rng = random.Random(seed + restart * 7919)
            
            # Shuffle triangles for randomization
            shuffled_triangles = triangles[:]
            rng.shuffle(shuffled_triangles)
            
            used_edges = set()
            selected_triangles = []
            
            # Process triangles in shuffled order
            for tri in shuffled_triangles:
                u, v, w = tri
                edges_in_tri = [tuple(sorted((u, v))), tuple(sorted((v, w))), tuple(sorted((u, w)))]
                
                # Check if all three edges are unused
                if all(e not in used_edges for e in edges_in_tri):
                    selected_triangles.append(tri)
                    for e in edges_in_tri:
                        used_edges.add(e)
            
            # Build the clique partition
            cliques = [list(tri) for tri in selected_triangles]
            
            # Add remaining edges as size-2 cliques
            for i, (u, v) in enumerate(edges):
                e_sorted = tuple(sorted((u, v)))
                if e_sorted not in used_edges:
                    cliques.append([u, v])
            
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_result = cliques[:].copy()
            
            # Early termination if we've found a good solution
            # Theoretical minimum: each triangle saves 1 clique (3 edges -> 1 clique vs 3 cliques)
            # So minimum possible = m - len(selected_triangles) where m = total edges
            if best_count <= len(edges) - len(triangles):
                break
        
        if best_result is None:
            # Fallback: all edges as individual cliques
            best_result = [[u, v] for u, v in edges]
        
        return best_result
    
    # Solve both instances
    time_per_instance = (deadline - time.time()) / 2 - 5  # Leave buffer
    
    result1 = solve_clique_partition(32, 0.6, 601, time_per_instance)
    
    remaining_time = deadline - time.time() - 10
    result2 = solve_clique_partition(64, 0.55, 602, max(5, remaining_time))
    
    return [result1, result2]

# EVOLVE-BLOCK-END
