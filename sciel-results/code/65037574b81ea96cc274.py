import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    # Start with K20,20 plus one edge (0,1)
    edge_set = set()
    for u in range(20):
        for v in range(20, 40):
            edge_set.add(frozenset((u, v)))
    edge_set.add(frozenset((0, 1)))
    
    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for e in edge_set:
            a, b = tuple(e)
            adj[a].add(b)
            adj[b].add(a)
        return adj
    
    def get_book_size(e, adj):
        a, b = tuple(e)
        return len(adj[a] & adj[b])

    def get_max_book(edge_set, adj):
        max_b = 0
        for e in edge_set:
            b = get_book_size(e, adj)
            if b > max_b:
                max_b = b
        return max_b

    adj = build_adj(edge_set)
    current_book = get_max_book(edge_set, adj)
    best_book = current_book
    best_edge_set = set(edge_set)
    best_adj = [s.copy() for s in adj]
    
    random.seed(42)
    T = 5.0
    T_min = 0.1
    cooling = 0.9995
    
    # List of all possible edges for random addition
    all_edges = [frozenset((i, j)) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline and T > T_min:
        # Select edge to remove
        e_remove = random.choice(list(edge_set))
        u, v = tuple(e_remove)
        
        # Select edge to add
        e_add = random.choice(all_edges)
        
        if e_add == e_remove or e_add in edge_set:
            T *= cooling
            continue
        
        a, b = tuple(e_add)
        
        # Identify affected vertices
        affected_vertices = {u, v, a, b}
        
        # Identify affected edges: all edges incident to any vertex in affected_vertices
        affected_edges = set()
        for x in affected_vertices:
            for y in adj[x]:
                affected_edges.add(frozenset((x, y)))
        
        # Store old book sizes for affected edges
        old_books = {}
        for e in affected_edges:
            if e in edge_set:
                old_books[e] = get_book_size(e, adj)
        
        # Apply move
        edge_set.remove(e_remove)
        edge_set.add(e_add)
        adj[u].remove(v)
        adj[v].remove(u)
        adj[a].add(b)
        adj[b].add(a)
        
        # Calculate new book sizes for affected edges that are now in edge_set
        # And check if any new edge has a high book size
        new_max = 0
        for e in affected_edges:
            if e in edge_set:
                b_size = get_book_size(e, adj)
                if b_size > new_max:
                    new_max = b_size
        
        # If the previous global max edge was NOT in affected_edges, the global max is max(prev_max, new_max)
        # If it WAS in affected_edges, we need to be careful.
        # Since we only updated affected edges, if the old max edge was affected, its value changed.
        # If the old max edge was NOT affected, its value is still current_book.
        # So new_global_max = max(current_book if e_old_max not in affected else 0, new_max)
        # But we don't track which edge was the max.
        # Safest and still fast: compute max over all edges? 
        # M=401. 401 set intersections. 
        # Let's estimate cost. Set intersection of size ~20 is fast.
        # 401 * 100ns = 40us. 
        # Loop overhead in Python is higher. Maybe 1ms per iteration?
        # 120s / 1ms = 120,000 iterations. That's plenty.
        
        # Let's just recompute global max to be safe and simple.
        new_global_max = get_max_book(edge_set, adj)
        
        delta = new_global_max - current_book
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_book = new_global_max
            if current_book < best_book:
                best_book = current_book
                best_edge_set = set(edge_set)
                best_adj = [s.copy() for s in adj]
        else:
            # Revert
            edge_set.remove(e_add)
            edge_set.add(e_remove)
            adj[a].remove(b)
            adj[b].remove(a)
            adj[u].add(v)
            adj[v].add(u)
            
        T *= cooling
    
    return [list(e) for e in best_edge_set]
