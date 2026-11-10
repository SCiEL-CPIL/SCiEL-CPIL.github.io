"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Start with Grötzsch graph on vertices 0-10
    # Grötzsch graph: Mycielski graph of C5
    # C5: vertices 0-4, edges (0,1),(1,2),(2,3),(3,4),(4,0)
    # Mycielski: add vertices 5-9 as copies, vertex 10 as universal to 5-9
    # Edges: original C5 edges, each original vertex connected to copy of its neighbors, and 10 connected to all copies
    
    def get_groetzsch_edges():
        edges = set()
        # C5 edges
        for i in range(5):
            edges.add(frozenset([i, (i+1)%5]))
        # Copy vertices 5-9
        for i in range(5):
            a = i
            b = (i+1)%5
            c = (i-1)%5
            # Copy of i (vertex 5+i) connects to b and c
            edges.add(frozenset([5+i, b]))
            edges.add(frozenset([5+i, c]))
        # Vertex 10 connects to all copies 5-9
        for i in range(5):
            edges.add(frozenset([10, 5+i]))
        return edges
    
    def is_triangle_free(edge_set):
        """Check if graph is triangle-free using adjacency"""
        adj = {i: set() for i in range(n)}
        for e in edge_set:
            u, v = tuple(e)
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[u]:
                        if w > v and w in adj[v]:
                            return False
        return True
    
    def is_not_3_colorable(edge_set):
        """Check if graph is NOT 3-colorable using backtracking"""
        adj = {i: set() for i in range(n)}
        for e in edge_set:
            u, v = tuple(e)
            adj[u].add(v)
            adj[v].add(u)
        
        # Order vertices by degree (descending) for better pruning
        deg_order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        color = [-1] * n
        color[deg_order[0]] = 0
        
        def backtrack(idx):
            if idx == n:
                return True
            v = deg_order[idx]
            # Try all 3 colors
            for c in range(3):
                conflict = False
                for nb in adj[v]:
                    if color[nb] == c:
                        conflict = True
                        break
                if not conflict:
                    color[v] = c
                    if backtrack(idx+1):
                        return True
                    color[v] = -1
            return False
        
        return not backtrack(1)
    
    def can_add_edge(edge_set, u, v):
        """Check if adding edge (u,v) maintains triangle-free and not 3-colorable"""
        if frozenset([u,v]) in edge_set:
            return False
        # Check triangle: if any common neighbor exists, adding edge creates triangle
        adj_u = set()
        adj_v = set()
        for e in edge_set:
            a, b = tuple(e)
            if a == u:
                adj_u.add(b)
            elif b == u:
                adj_u.add(a)
            if a == v:
                adj_v.add(b)
            elif b == v:
                adj_v.add(a)
        if adj_u & adj_v:
            return False
        # Check 3-colorability with new edge
        new_set = edge_set | {frozenset([u,v])}
        return is_not_3_colorable(new_set)
    
    # Initialize with Grötzsch graph
    best_edges = get_groetzsch_edges()
    best_count = len(best_edges)
    
    # All possible edges
    all_edges = [frozenset([i,j]) for i in range(n) for j in range(i+1, n)]
    
    # Greedy improvement with random restarts
    iterations = 0
    while time.time() < deadline and iterations < 50:
        iterations += 1
        improved = False
        # Try to add edges in random order
        missing = [e for e in all_edges if e not in best_edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline:
                break
            u, v = tuple(e)
            if can_add_edge(best_edges, u, v):
                best_edges = best_edges | {e}
                best_count += 1
                improved = True
        
        if not improved:
            # Perturbation: remove some edges to escape local optimum
            if best_count > 15 and len(best_edges) > 10:
                num_remove = random.randint(2, 4)
                to_remove = random.sample(list(best_edges), min(num_remove, len(best_edges)))
                for e in to_remove:
                    best_edges.discard(e)
                best_count = len(best_edges)
            else:
                break
    
    return [list(e) for e in best_edges]
# EVOLVE-BLOCK-END
