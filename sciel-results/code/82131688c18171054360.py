import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    # We construct G as the complement of H.
    # m_G = 401 => m_H = 780 - 401 = 379.
    # H will be almost 19-regular.
    
    def generate_random_regular_graph(n, k):
        """Generates a random k-regular graph on n vertices."""
        if (n * k) % 2 != 0:
            raise ValueError("n*k must be even")
        
        # Use configuration model with retries for simplicity and speed for small n
        while True:
            # Create list of stubs
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            
            random.shuffle(stubs)
            
            edges = set()
            valid = True
            for i in range(0, n * k, 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (min(u,v), max(u,v)) in edges:
                    valid = False
                    break
                edges.add((min(u,v), max(u,v)))
            
            if valid and len(edges) == n * k // 2:
                return list(edges)
    
    # Initial H: 19-regular graph (380 edges)
    # Then remove 1 edge to get 379 edges.
    
    def get_max_book_size(edges_G):
        """
        edges_G: list of (u, v) tuples.
        Returns max book size.
        """
        # Build adjacency sets for speed
        adj = [set() for _ in range(n)]
        for u, v in edges_G:
            adj[u].add(v)
            adj[v].add(u)
            
        max_book = 0
        # For each edge (u, v), count common neighbors
        for u, v in edges_G:
            # Common neighbors are in intersection of adj[u] and adj[v]
            # len(adj[u] & adj[v])
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book

    def get_max_book_size_from_H(edges_H):
        """
        edges_H: list of (u, v) tuples in H.
        G is complement of H.
        Edges in G are pairs NOT in H.
        """
        # Build adjacency sets for H
        adj_H = [set() for _ in range(n)]
        for u, v in edges_H:
            adj_H[u].add(v)
            adj_H[v].add(u)
            
        max_book = 0
        # Iterate over all pairs (u, v) in G
        # These are pairs NOT in H
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in adj_H[u]: # Not an edge in H, so it is in G
                    # Book size in G for (u, v) is number of w such that (u,w) in G and (v,w) in G
                    # (u,w) in G <=> (u,w) NOT in H
                    # (v,w) in G <=> (v,w) NOT in H
                    # So w is NOT in N_H(u) and NOT in N_H(v)
                    # w != u, w != v
                    # Count w in range(n) \ {u, v} such that w not in adj_H[u] and w not in adj_H[v]
                    
                    # Optimization:
                    # Total vertices = n
                    # Vertices in N_H(u) U N_H(v) U {u, v}
                    # Since (u,v) not in H, u not in N_H(v) and v not in N_H(u)
                    # So {u, v} is disjoint from N_H(u) U N_H(v)
                    # Size = |N_H(u) U N_H(v)| + 2
                    # |N_H(u) U N_H(v)| = |N_H(u)| + |N_H(v)| - |N_H(u) intersect N_H(v)|
                    
                    nu = len(adj_H[u])
                    nv = len(adj_H[v])
                    # Intersection size
                    # For small n, set intersection is fast
                    inter = len(adj_H[u] & adj_H[v])
                    union_size = nu + nv - inter
                    total_blocked = union_size + 2
                    book_size = n - total_blocked
                    
                    if book_size > max_book:
                        max_book = book_size
        return max_book

    # Initial construction
    # 19-regular graph on 40 nodes
    initial_H = generate_random_regular_graph(n, 19)
    
    # Remove one edge to get 379 edges
    # Pick a random edge to remove
    edge_to_remove = random.choice(initial_H)
    H = [e for e in initial_H if e != edge_to_remove]
    
    current_score = get_max_book_size_from_H(H)
    best_H = H[:]
    best_score = current_score
    
    # Local Search
    # Swap: Remove edge (u,v) from H, Add edge (x,y) to H.
    # Must maintain simple graph.
    # Degrees change: u,v lose 1; x,y gain 1.
    # We want to keep it close to regular to keep book sizes low.
    # Ideally, we swap to restore regularity or improve the max book.
    
    # To ensure we don't get stuck, we can allow degree fluctuations but penalize high degrees?
    # The formula book_G(u,v) = n - 2 - |N_H(u) U N_H(v)| depends on degrees.
    # If degrees are 19, |N| = 19.
    # If degrees are 18, |N| = 18.
    # Lower degree in H -> Higher book in G?
    # book = 38 - (nu + nv - inter).
    # If nu, nv decrease, the term (nu + nv) decreases, so book increases.
    # So we want H to be as regular as possible, ideally 19-regular.
    # But we MUST have 379 edges. So we MUST have at least two vertices of degree 18.
    # So we should minimize the impact of these degree 18 vertices.
    
    # Strategy: 
    # 1. Start with 19-regular H0.
    # 2. Remove edge (u,v). H has u,v deg 18.
    # 3. Search for swaps that improve max book size.
    
    # Precompute adjacency sets for H
    adj_H = [set() for _ in range(n)]
    for u, v in H:
        adj_H[u].add(v)
        adj_H[v].add(u)
        
    # Helper to update adj sets after swap
    def swap_edges(H, adj_H, edge_remove, edge_add):
        u, v = edge_remove
        x, y = edge_add
        
        # Check validity
        if (x, y) in adj_H[x]:
            return False, H, adj_H
        if (u, v) not in adj_H[u]:
            return False, H, adj_H
            
        # Perform swap
        adj_H[u].remove(v)
        adj_H[v].remove(u)
        adj_H[x].add(y)
        adj_H[y].add(x)
        
        H.remove(edge_remove)
        H.append(edge_add)
        
        return True, H, adj_H

    def undo_swap(H, adj_H, edge_remove, edge_add):
        u, v = edge_remove
        x, y = edge_add
        
        adj_H[x].remove(y)
        adj_H[y].remove(x)
        adj_H[u].add(v)
        adj_H[v].add(u)
        
        H.remove(edge_add)
        H.append(edge_remove)
        
        return H, adj_H

    while time.time() < deadline:
        # Pick a random edge to remove from H
        edge_remove = random.choice(H)
        u, v = edge_remove
        
        # Pick a random non-edge to add to H
        # To bias towards regularity, we can pick x, y such that 
        # deg_H(x) < 19 and deg_H(y) < 19? 
        # Or just random.
        
        # Random non-edge
        while True:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if (min(x,y), max(x,y)) in adj_H[x]:
                continue
            edge_add = (min(x,y), max(x,y))
            break
            
        # Perform swap
        ok, H_new, adj_new = swap_edges(H, adj_H, edge_remove, edge_add)
        if not ok:
            continue
            
        new_score = get_max_book_size_from_H(H_new)
        
        if new_score < best_score:
            best_score = new_score
            best_H = H_new[:]
            H = H_new
            adj_H = adj_new
        else:
            # Undo
            H, adj_H = undo_swap(H, adj_H, edge_remove, edge_add)
            
        # Optional: Simulated Annealing accept worse moves?
        # For now, pure hill climbing.
        
    # Convert best_H to G edges
    # G edges are all pairs NOT in best_H
    edges_G = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in adj_H[u]: # Note: adj_H is from the last state, which is best_H if we tracked correctly?
                 # Wait, if we undo, adj_H is restored.
                 # If we accept, H and adj_H are updated.
                 # So at the end, H and adj_H represent best_H.
                 pass
    
    # Rebuild adj sets for best_H to be safe
    adj_best = [set() for _ in range(n)]
    for u, v in best_H:
        adj_best[u].add(v)
        adj_best[v].add(u)
        
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj_best[u]:
                edges_G.append([u, v])
                
    return edges_G
