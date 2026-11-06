import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    n = 40
    m = 401
    
    # Use bitmask representation: adj[i] is an integer where bit j is set if edge (i,j) exists
    # This allows O(1) intersection via bitwise AND and popcount
    
    def popcount(x):
        return bin(x).count('1')
    
    # Initialize with a random graph with 401 edges
    # Better: start with a structured graph
    # Let's start with a random graph and optimize
    
    random.seed(42)
    
    # Start: random graph with exactly m edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    init_edges = all_edges[:m]
    
    adj = [0] * n  # bitmask adjacency
    edge_set = set()
    for u, v in init_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v))
    
    def compute_max_book(adj, edge_list):
        max_bk = 0
        for u, v in edge_list:
            bk = popcount(adj[u] & adj[v])
            if bk > max_bk:
                max_bk = bk
        return max_bk
    
    edge_list = list(edge_set)
    max_book = compute_max_book(adj, edge_list)
    
    best_adj = adj[:]
    best_edges = set(edge_set)
    best_max = max_book
    
    # Precompute: for each edge, its book size
    def compute_all_books(adj, edge_set):
        books = {}
        for (u, v) in edge_set:
            books[(u, v)] = popcount(adj[u] & adj[v])
        return books
    
    books = compute_all_books(adj, edge_set)
    
    # Simulated annealing
    temp = 5.0
    cooling = 0.9995
    min_temp = 0.01
    
    # Efficient delta computation for edge swap: remove (u,v), add (x,y)
    # Only edges incident to u, v, x, y can have their book change
    
    def affected_edges(edge_set, u, v, x, y):
        """Return edges whose book might change when removing (u,v) and adding (x,y)"""
        affected = set()
        # Edges incident to u, v, x, y
        for w in range(n):
            if w == u or w == v or w == x or w == y:
                continue
            for a, b in [(u, w), (v, w), (x, w), (y, w)]:
                if a < b:
                    if (a, b) in edge_set:
                        affected.add((a, b))
                else:
                    if (b, a) in edge_set:
                        affected.add((b, a))
        # Also the edges (u,v) and (x,y) themselves
        if (u, v) in edge_set or (v, u) in edge_set:
            affected.add((min(u,v), max(u,v)))
        if (x, y) in edge_set or (y, x) in edge_set:
            affected.add((min(x,y), max(x,y)))
        return affected
    
    while time.time() < deadline and temp > min_temp:
        # Pick a random edge to remove
        u, v = random.choice(list(edge_set))
        
        # Pick a random non-edge to add
        # Non-edges: total possible = 40*39/2 = 780, we have 401, so 379 non-edges
        # Just pick random x,y and check
        for _ in range(50):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if (x, y) in edge_set or (y, x) in edge_set:
                continue
            if x == u and y == v:
                continue
            if x == v and y == u:
                continue
            break
        else:
            temp *= cooling
            continue
        
        # Compute delta in max book
        # Save current state
        old_adj_u = adj[u]
        old_adj_v = adj[v]
        old_adj_x = adj[x]
        old_adj_y = adj[y]
        
        # Apply change
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[x] |= (1 << y)
        adj[y] |= (1 << x)
        
        edge_set.discard((u, v))
        edge_set.discard((v, u))
        if x < y:
            edge_set.add((x, y))
        else:
            edge_set.add((y, x))
        
        # Recompute books for affected edges only
        # Get neighbors of u, v, x, y
        neighbors = set()
        for w in range(n):
            if (1 << w) & (old_adj_u | old_adj_v | old_adj_x | old_adj_y | adj[u] | adj[v] | adj[x] | adj[y]):
                neighbors.add(w)
        
        # For each affected edge, compute new book
        new_max = 0
        # Check all edges that could be affected
        # Edges incident to u, v, x, y
        affected = set()
        for w in range(n):
            for a, b in [(u, w), (v, w), (x, w), (y, w)]:
                if a == b:
                    continue
                key = (min(a,b), max(a,b))
                if key in edge_set:
                    affected.add(key)
        
        for (a, b) in affected:
            bk = popcount(adj[a] & adj[b])
            if bk > new_max:
                new_max = bk
        
        # Also check if there are edges not incident to u,v,x,y that might have increased?
        # No - removing an edge can only decrease books, adding (x,y) only affects edges incident to x or y
        # Wait, that's not quite right. Adding edge (x,y) increases the book of edge (a,b) if a is neighbor of both x and y... 
        # Actually no. The book of edge (a,b) = |N(a) ∩ N(b)|. Adding edge (x,y) changes N(x) and N(y).
        # So only edges incident to x or y can have their book changed.
        # Similarly removing (u,v) only changes N(u) and N(v), so only edges incident to u or v are affected.
        # So affected = edges incident to u, v, x, or y. Correct.
        
        delta = new_max - max_book
        
        if delta <= 0 or random.random() < np.exp(-delta / temp):
            # Accept
            max_book = new_max
            if max_book < best_max:
                best_max = max_book
                best_adj = adj[:]
                best_edges = set(edge_set)
        else:
            # Reject: undo
            adj[u] = old_adj_u
            adj[v] = old_adj_v
            adj[x] = old_adj_x
            adj[y] = old_adj_y
            edge_set.add((u, v))
            if x < y:
                edge_set.discard((x, y))
            else:
                edge_set.discard((y, x))
        
        temp *= cooling
    
    # Return best found
    result = [[u, v] for u, v in best_edges]
    assert len(result) == m
    return result
