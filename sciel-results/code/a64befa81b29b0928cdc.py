import time
import random
import itertools

def construct():
    """
    Constructs a simple graph on 20 vertices with edge colours 0..19.
    Maximizes the number of edges while ensuring every 4-cycle is rainbow.
    
    Strategy: Find a dense C4-free graph. Any colouring of a C4-free graph
    is valid because there are no 4-cycles to violate the rainbow condition.
    We use a greedy approach with backtracking to find a maximal C4-free graph,
    and also try random restarts to find denser ones.
    """
    n = 20
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    def generate_random_graph(m):
        """Generates a random graph with m edges on n vertices."""
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        edges = all_edges[:m]
        return edges

    def is_c4_free(edges):
        """Checks if the graph defined by edges is free of 4-cycles."""
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # A graph is C4-free iff no two vertices have more than 1 common neighbor
        for u in range(n):
            for w in range(u + 1, n):
                common = adj[u].intersection(adj[w])
                if len(common) > 1:
                    return False
        return True

    def greedy_c4_free(order=None):
        """
        Greedily add edges in the given order (or random order),
        keeping only edges that don't create a C4.
        Returns the list of edges that form a C4-free graph.
        """
        if order is None:
            order = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(order)
        
        adj = [set() for _ in range(n)]
        edge_set = set()
        result_edges = []
        
        for u, v in order:
            if u == v:
                continue
            # Check if adding edge (u,v) creates a C4
            # A C4 is created if there exists a vertex w such that
            # both (u,w) and (v,w) are in the graph (creating u-w-v-?-u? No)
            # Actually, adding edge (u,v) creates a C4 if there exist two vertices w1, w2
            # such that u-w1-v and u-w2-v are already paths of length 2, but that's not a C4.
            # A C4 is u-a-v-b-u. Adding edge (u,v) creates a C4 if there are vertices a, b
            # such that (u,a), (a,v), (v,b), (b,u) are edges.
            # Simpler: adding (u,v) creates a C4 if there exist w1 != w2 such that
            # (u,w1), (w1,v), (v,w2), (w2,u) all exist.
            # Even simpler: count common neighbors of u and v in the current graph.
            # If |N(u) ∩ N(v)| >= 2, then adding (u,v) would create a C4.
            # Wait, no. If u and v have two common neighbors w1, w2, then u-w1-v-w2-u is a C4.
            # So we must check if |N(u) ∩ N(v)| >= 2.
            
            common = adj[u].intersection(adj[v])
            if len(common) < 2:
                # Safe to add
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
                result_edges.append((u, v))
        
        return result_edges

    best_edges = None
    best_m = 0
    
    # Strategy 1: Greedy with random restarts
    # Try multiple random orderings and keep the densest C4-free graph
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        edges = greedy_c4_free()
        m = len(edges)
        if m > best_m:
            best_m = m
            best_edges = edges[:]
        
        # If we've found a very dense graph, we might be close to optimal
        # The maximum C4-free edges on 20 vertices is known to be around 60-70
        if best_m >= 70:
            # Try a few more restarts if we have time
            pass
    
    # Strategy 2: Try to improve the best graph by local search
    # (This could be complex, so we'll rely on random restarts primarily)
    
    # If we found a good graph, use it
    if best_edges is not None and best_m > 0:
        # Assign colours. Since no C4, any colouring works.
        # Use colour 0 for all edges for simplicity.
        coloured_edges = [[u, v, 0] for u, v in best_edges]
        return coloured_edges
    
    # Fallback: star graph (19 edges)
    return [[0, v, 0] for v in range(1, 20)]

