import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    def seed_graph():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    best_result = seed_graph()
    
    # Helper functions
    def has_c5(n, edge_set):
        """Check if graph has a 5-cycle"""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        for a in range(n):
            for b in adj[a]:
                for c in adj[b]:
                    if c == a:
                        continue
                    for d in adj[c]:
                        if d in (a, b):
                            continue
                        for e in adj[d]:
                            if e == a and e not in (b, c, d):
                                return True
        return False
    
    def max_independent_set_size(n, edge_set):
        """Find size of maximum independent set"""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        best = 0
        # Use greedy with backtracking for small graphs
        # For n<=35, this should be feasible
        def dfs(remaining, current_size, current_set):
            nonlocal best
            if not remaining:
                if current_size > best:
                    best = current_size
                return
            if current_size + len(remaining) <= best:
                return
            
            # Pick a vertex to branch on
            v = next(iter(remaining))
            remaining_list = list(remaining)
            remaining_list.remove(v)
            
            # Option 1: Don't include v
            dfs(set(remaining_list), current_size, current_set)
            
            # Option 2: Include v (remove v and its neighbors)
            new_remaining = set(remaining_list)
            for u in adj[v]:
                if u in new_remaining:
                    new_remaining.remove(u)
            dfs(new_remaining, current_size + 1, current_set | {v})
        
        dfs(set(range(n)), 0, set())
        return best
    
    # Try to construct n=33 graph
    # Strategy: Start with 8*K4, try to add vertex 32 with various connections
    # and check constraints
    
    def try_add_vertex_to_seed():
        """Try to add vertex 32 to 8*K4 seed"""
        n = 33
        base_edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    base_edges.append([base + i, base + j])
        
        # Try all possible neighborhoods for vertex 32
        # Vertex 32 can connect to any subset of vertices 0..31
        # But we need to check C5-free and alpha < 9
        
        # To be efficient, note that connecting to 2+ vertices in a K4 creates C5
        # So vertex 32 can connect to at most 1 vertex per K4
        # But that gives alpha >= 9 (as shown before)
        
        # Alternative: Modify the K4s
        # What if we remove one edge from one K4 and use that to connect vertex 32?
        
        # Try: Remove edge (0,1) from K4_0, then connect 32 to 0 and 1
        # K4_0 vertices: 0,1,2,3. Edges: (0,2),(0,3),(1,2),(1,3),(2,3)
        # Connect 32 to 0 and 1.
        # Check C5: 32-0-2-3-1-32? 
        # 32-0 (edge), 0-2 (edge), 2-3 (edge), 3-1 (edge), 1-32 (edge). Yes, C5!
        
        # What if we remove more edges?
        # This is getting complex. Let me try a different approach.
        
        return None
    
    # Alternative: Try a completely different construction
    # What about a graph based on a 5x5 grid or similar?
    
    # Let me try: Complete graph K8, blow up each vertex to 4 vertices
    # No, that would have C5s.
    
    # What about the line graph of K_{4,4}?
    # Line graph of K_{4,4} has 16 vertices, each representing an edge of K_{4,4}
    # Two vertices are adjacent if the corresponding edges share an endpoint
    # This is C5-free? Let me check.
    # Actually, line graphs of bipartite graphs are perfect, so they have no odd cycles of length > 3
    # Wait, that's not right. Line graphs can have C5s.
    
    # Let me try a random local search
    def random_local_search():
        """Random local search for n=33 C5-free graph with alpha < 9"""
        n = 33
        random.seed(42)
        
        # Start with a random graph
        for trial in range(50):
            if time.time() > deadline - 10.0:
                break
            
            # Initialize with sparse random graph
            edges = set()
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.3:
                        edges.add((i, j))
            
            # Local search: try to add edges to reduce alpha, while maintaining C5-free
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                alpha = max_independent_set_size(n, list(edges))
                if alpha < 9:
                    # Check C5
                    if not has_c5(n, list(edges)):
                        edge_list = [[u, v] for u, v in sorted(edges)]
                        return {'n': n, 'edges': edge_list}
                
                # Try adding a random edge
                all_possible = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edges]
                if not all_possible:
                    break
                
                for _ in range(10):
                    u, v = random.choice(all_possible)
                    new_edges = edges | {(u, v)}
                    if not has_c5(n, list(new_edges)):
                        new_alpha = max_independent_set_size(n, list(new_edges))
                        if new_alpha <= alpha:
                            edges = new_edges
                            improved = True
                            break
        
        return None
    
    # Try random local search
    result = random_local_search()
    if result is not None:
        best_result = result
    else:
        # Fall back to seed
        best_result = seed_graph()
    
    return best_result
