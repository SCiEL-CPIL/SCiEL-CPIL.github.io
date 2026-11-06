import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n = 40
    target_edges = 401
    
    def compute_books(edges, n):
        """Compute book size for each edge. Returns dict edge->book_size and max_book."""
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u,v), max(u,v)))
        
        books = {}
        max_book = 0
        for (u, v) in edge_set:
            common = len(adj[u] & adj[v])
            books[(u, v)] = common
            if common > max_book:
                max_book = common
        return books, max_book
    
    def edge_key(u, v):
        return (min(u, v), max(u, v))
    
    # Initialize: random graph with 401 edges
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.seed(42)
    
    # Try multiple random starts
    best_edges = None
    best_max_book = float('inf')
    
    for attempt in range(20):
        if time.time() > deadline - 30:
            break
            
        if attempt == 0:
            # Start with K20,20 + 1 edge as baseline
            edges = [(a, b) for a in range(20) for b in range(20, 40)]
            edges.append((0, 1))
        else:
            # Random graph
            edges = random.sample(all_possible, target_edges)
        
        books, max_book = compute_books(edges, n)
        
        if max_book < best_max_book:
            best_max_book = max_book
            best_edges = list(edges)
        
        # Local search: try to reduce max book
        edges_set = set(edge_key(u, v) for u, v in edges)
        edges_list = list(edges)
        current_max = max_book
        no_improve = 0
        max_iterations = 5000
        it = 0
        
        while it < max_iterations and time.time() < deadline - 10:
            it += 1
            books, current_max = compute_books(edges_list, n)
            
            # Find edge with max book
            max_edge = None
            for (u, v), bk in books.items():
                if bk == current_max:
                    max_edge = (u, v)
                    break
            
            if max_edge is None:
                break
            
            u, v = max_edge
            # Try to reduce book of this edge by swapping
            # Pick a neighbor w of u (w != v), and x not adjacent to v (x != u)
            # Swap (u,w) with (v,x) or similar
            
            neighbors_u = list(books.keys())
            # Get actual neighbors
            adj_u = [x for x in range(n) if x != u and edge_key(u, x) in edges_set]
            adj_v = [x for x in range(n) if x != v and edge_key(v, x) in edges_set]
            
            improved = False
            for w in random.sample(adj_u, min(5, len(adj_u))):
                if w == v:
                    continue
                # Try removing (u,w) and adding (v, x) for some x
                # x should be not adjacent to v, and adding (v,x) shouldn't create too many triangles
                non_adj_v = [x for x in range(n) if x != v and x != u and edge_key(v, x) not in edges_set]
                if not non_adj_v:
                    continue
                for x in random.sample(non_adj_v, min(3, len(non_adj_v))):
                    # Check: removing (u,w) and adding (v,x)
                    # Must ensure (v,x) is valid and (u,w) exists
                    if edge_key(u, w) not in edges_set or edge_key(v, x) in edges_set:
                        continue
                    # Simulate
                    new_edges_set = set(edges_set)
                    new_edges_set.remove(edge_key(u, w))
                    new_edges_set.add(edge_key(v, x))
                    
                    # Quick check: does this reduce max book?
                    # Compute new book for affected edges
                    # Edges affected: those involving u, w, v, x
                    affected = set()
                    for a, b in new_edges_set:
                        if a in (u, w, v, x) or b in (u, w, v, x):
                            affected.add((a, b))
                    
                    # Build adj for quick computation
                    adj_new = [set() for _ in range(n)]
                    for a, b in new_edges_set:
                        adj_new[a].add(b)
                        adj_new[b].add(a)
                    
                    new_max = 0
                    for a, b in affected:
                        bk = len(adj_new[a] & adj_new[b])
                        if bk > new_max:
                            new_max = bk
                        if new_max > current_max:
                            break
                    
                    if new_max <= current_max:
                        # Accept
                        edges_set = new_edges_set
                        edges_list = [list(e) for e in new_edges_set]
                        current_max = new_max
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                no_improve = 0
                if current_max < best_max_book:
                    best_max_book = current_max
                    best_edges = [list(e) for e in edges_set]
            else:
                no_improve += 1
                if no_improve > 200:
                    break
        
        if best_max_book <= 7:
            break
    
    # Return best found
    if best_edges is None:
        # Fallback: K20,20 + 1 edge
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return [[u, v] for u, v in best_edges]
