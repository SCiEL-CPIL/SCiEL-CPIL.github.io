"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a graph on 40 vertices with 401 edges minimizing the largest book size.
    Strategy: Start with a random 40-vertex graph with 401 edges, then use simulated annealing
    to minimize the maximum book size (number of triangles containing any edge).
    """
    n = 40
    target_edges = 401
    deadline = time.time() + 110  # Leave 10s safety margin for 120s limit
    
    def compute_max_book(adj):
        """Compute the maximum book size (max triangles per edge)."""
        max_book = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    # Count common neighbors
                    cnt = 0
                    for w in range(n):
                        if w != u and w != v and adj[u][w] and adj[v][w]:
                            cnt += 1
                    if cnt > max_book:
                        max_book = cnt
        return max_book
    
    def edge_triangle_count(adj, u, v):
        """Count triangles containing edge (u, v)."""
        cnt = 0
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                cnt += 1
        return cnt
    
    # Initialize with a random graph
    def random_graph():
        edges = set()
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs[:target_edges]:
            edges.add((u, v))
        return edges
    
    def edges_to_adj(edges):
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        return adj
    
    # Start with a good initial guess: balanced bipartite + extra edges
    def initial_graph():
        edges = set()
        # Balanced bipartite K20,20 has 400 edges
        for i in range(20):
            for j in range(20, 40):
                edges.add((i, j))
        # Add 1 more edge - try to pick one that minimizes triangles
        # In K20,20, adding an edge within a part creates triangles
        # Adding edge (0,1) creates triangles with all 20 vertices in other part
        # But we need exactly 401 edges, so we must add 1
        # Let's try different single edges and pick best
        best_edge = None
        best_book = float('inf')
        # Try some candidate edges within parts
        candidates = [(i, j) for i in range(20) for j in range(i+1, 20)] + \
                     [(i, j) for i in range(20, 40) for j in range(i+1, 40)]
        random.shuffle(candidates)
        for u, v in candidates[:50]:  # Check a subset
            if (u, v) in edges or (v, u) in edges:
                continue
            test_edges = edges | {(u, v)}
            adj = edges_to_adj(test_edges)
            book = compute_max_book(adj)
            if book < best_book:
                best_book = book
                best_edge = (u, v)
        if best_edge is None:
            best_edge = (0, 1)
        edges.add(best_edge)
        return edges
    
    # Simulated annealing
    def simulated_annealing(edges):
        adj = edges_to_adj(edges)
        current_book = compute_max_book(adj)
        best_edges = set(edges)
        best_book = current_book
        
        # Precompute neighbor lists for speed
        def get_neighbors(u):
            return [v for v in range(n) if adj[u][v]]
        
        # Generate all edges list
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        step = 0
        
        while time.time() < deadline:
            step += 1
            
            # Pick a random edge to remove
            if not edges:
                break
            u1, v1 = random.choice(list(edges))
            
            # Pick a random non-edge to add
            # Find a non-edge
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not adj[i][j]:
                        non_edges.append((i, j))
            if not non_edges:
                break
            u2, v2 = random.choice(non_edges)
            
            # Try the swap
            # Remove (u1, v1), add (u2, v2)
            old_book = current_book
            
            # Update adjacency
            adj[u1][v1] = False
            adj[v1][u1] = False
            adj[u2][v2] = True
            adj[v2][u2] = True
            
            # Compute new max book
            new_book = compute_max_book(adj)
            
            # Accept or reject
            if new_book < old_book:
                # Accept
                edges.discard((u1, v1))
                edges.add((u2, v2))
                current_book = new_book
                if new_book < best_book:
                    best_book = new_book
                    best_edges = set(edges)
            else:
                # Reject, undo
                adj[u1][v1] = True
                adj[v1][u1] = True
                adj[u2][v2] = False
                adj[v2][u2] = False
                # Sometimes accept worse moves (simulated annealing)
                if T > T_min:
                    delta = new_book - old_book
                    if delta > 0:
                        prob = 1.0 / (1.0 + delta)  # Simple acceptance
                        if random.random() < prob:
                            edges.discard((u1, v1))
                            edges.add((u2, v2))
                            current_book = new_book
            
            T *= cooling
        
        return best_edges, best_book
    
    # Run simulated annealing with multiple restarts
    best_overall_edges = None
    best_overall_book = float('inf')
    
    # First, get a good initial solution
    edges = initial_graph()
    adj = edges_to_adj(edges)
    current_book = compute_max_book(adj)
    best_overall_edges = set(edges)
    best_overall_book = current_book
    
    # Run SA
    sa_edges, sa_book = simulated_annealing(set(edges))
    if sa_book < best_overall_book:
        best_overall_book = sa_book
        best_overall_edges = sa_edges
    
    # Try a few more restarts if time permits
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        # Random restart
        edges = random_graph()
        adj = edges_to_adj(edges)
        current_book = compute_max_book(adj)
        if current_book < best_overall_book:
            best_overall_book = current_book
            best_overall_edges = set(edges)
        
        sa_edges, sa_book = simulated_annealing(set(edges))
        if sa_book < best_overall_book:
            best_overall_book = sa_book
            best_overall_edges = sa_edges
    
    # Convert back to list format
    result = []
    for u, v in best_overall_edges:
        result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END
