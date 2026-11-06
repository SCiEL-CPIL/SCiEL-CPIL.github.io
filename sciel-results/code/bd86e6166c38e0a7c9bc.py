import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill

    n = 16

    # Helper: check if adding edge (u,v) to adjacency sets keeps it triangle-free
    def can_add_edge(adj, edge_set, u, v):
        if u == v:
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            return False
        # Check for common neighbors
        common = adj[u] & adj[v]
        if common:
            return False
        return True

    # Helper: check if graph is 3-colorable using backtracking with ordering
    def is_3_colorable(adj):
        color = [-1] * n
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        pos = {v: i for i, v in enumerate(order)}
        
        # For each vertex in order, track which colors are forbidden
        # Use bitmasks: color c is forbidden if any neighbor has color c
        
        def dfs(idx):
            if idx == n:
                return True
            u = order[idx]
            # Determine forbidden colors from already-colored neighbors
            forbidden = 0
            for v in adj[u]:
                if color[v] != -1:
                    forbidden |= (1 << color[v])
            
            for c in range(3):
                if forbidden & (1 << c):
                    continue
                color[u] = c
                if dfs(idx + 1):
                    return True
                color[u] = -1
            return False
        
        return dfs(0)

    # Helper: count edges from adjacency sets
    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    # Helper: extract edge list from adjacency sets
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    # Strategy: Start with known 4-chromatic triangle-free structures and greedily add edges
    # The Grötzsch graph has 11 vertices and 20 edges. We embed it in 16 vertices.
    
    def groetzch_embedded():
        """Return adjacency sets for a Grötzsch graph embedded in 16 vertices (0-10 used, 11-15 isolated initially)"""
        adj = [set() for _ in range(n)]
        # 5-cycle: 0-1-2-3-4-0
        cycle = [(0,1), (1,2), (2,3), (3,4), (4,0)]
        for u, v in cycle:
            adj[u].add(v)
            adj[v].add(u)
        # Second 5-cycle shifted: 5-6-7-8-9-5
        for u, v in cycle:
            u2, v2 = u + 5, v + 5
            adj[u2].add(v2)
            adj[v2].add(u2)
            # Cross edges: u connected to v+5 and v connected to u+5
            adj[u].add(v + 5)
            adj[v + 5].add(u)
            adj[v].add(u + 5)
            adj[u + 5].add(v)
        # Central vertex 10 connected to all of 5-9
        for u in range(5, 10):
            adj[10].add(u)
            adj[u].add(10)
        return adj

    def greedy_maximize(adj, seed=None):
        """Greedily add edges while maintaining triangle-free and 4-chromatic properties"""
        if seed is not None:
            random.seed(seed)
        
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))
        
        improved = True
        while improved:
            improved = False
            # Generate candidate edges
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (u, v) not in edge_set:
                        candidates.append((u, v))
            
            random.shuffle(candidates)
            
            for u, v in candidates:
                if can_add_edge(adj, edge_set, u, v):
                    # Try adding the edge
                    adj[u].add(v)
                    adj[v].add(u)
                    edge_set.add((u, v))
                    
                    # Check if still not 3-colorable
                    if not is_3_colorable(adj):
                        improved = True
                    else:
                        # Revert
                        adj[u].remove(v)
                        adj[v].remove(u)
                        edge_set.remove((u, v))
        
        return adj

    # Strategy 2: Start from scratch with random triangle-free graphs
    def random_triangle_free():
        """Generate a random triangle-free graph"""
        adj = [set() for _ in range(n)]
        edge_set = set()
        
        # Randomly add edges while maintaining triangle-free
        all_pairs = list(combinations(range(n), 2))
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if can_add_edge(adj, edge_set, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
        
        return adj

    # Strategy 3: Use the complete bipartite graph K_{8,8} as a base (triangle-free, 64 edges)
    # But K_{8,8} is 2-colorable. We need to modify it to make it 4-chromatic.
    # Actually, let's think differently.
    
    # Best approach: Use known constructions. The Mycielski construction can increase chromatic number
    # while preserving triangle-free property.
    
    # Let's use a more systematic approach:
    # 1. Start with Grötzsch graph (11 vertices, 20 edges, 4-chromatic, triangle-free)
    # 2. Add 5 more vertices and greedily add edges
    # 3. Also try other starting points
    
    best_adj = None
    best_count = -1
    
    # Try multiple seeds for the Grötzsch-based approach
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        
        adj = groetzch_embedded()
        adj = greedy_maximize(adj, seed=seed)
        count = count_edges(adj)
        
        if count > best_count:
            best_count = count
            best_adj = [set(s) for s in adj]  # Copy
    
    # Try random triangle-free graphs
    for seed in range(100):
        if time.time() > deadline - 5:
            break
        
        random.seed(seed)
        adj = random_triangle_free()
        
        # Check if it's 4-chromatic
        if not is_3_colorable(adj):
            adj = greedy_maximize(adj, seed=seed)
            count = count_edges(adj)
            if count > best_count:
                best_count = count
                best_adj = [set(s) for s in adj]
    
    # If we found a good graph, return it
    if best_adj is not None:
        return get_edges(best_adj)
    
    # Fallback: return the Grötzsch graph embedded in 16 vertices
    adj = groetzch_embedded()
    return get_edges(adj)

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
