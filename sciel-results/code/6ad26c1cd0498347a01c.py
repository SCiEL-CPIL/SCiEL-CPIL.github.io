import random
import math

def construct():
    n = 40
    m = 401
    
    # Use bitmasks for fast common neighbor calculation
    # adj[u] is an integer where bit v is set if (u,v) is an edge
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def common_neighbors(u, v):
        return (adj[u] & adj[v]).bit_count()
        
    # Initialize with random graph
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    edge_set = set(all_pairs[:m])
    
    for u, v in edge_set:
        add_edge(u, v)
        
    # Calculate initial books
    books = {}
    for u, v in edge_set:
        books[(u, v)] = common_neighbors(u, v)
        
    max_book = max(books.values())
    best_max = max_book
    best_edge_set = set(edge_set)
    best_adj = adj[:]
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    iterations = 50000
    
    # Precompute list of all pairs for quick non-edge selection?
    # No, picking a random non-edge from a list of ~760 is fast enough.
    # But we need to ensure we don't pick an existing edge.
    
    for _ in range(iterations):
        if max_book <= 7:
            break
            
        # Pick a random edge to remove
        u, v = random.choice(list(edge_set))
        
        # Pick a random non-edge to add
        # To be efficient, we can pick random pairs until we find a non-edge
        # Or maintain a list of non-edges. Maintaining a list is O(1) removal/addition if we use a set and a list?
        # Let's just pick random pairs. Probability of picking a non-edge is (780-401)/780 ~ 0.48.
        # So on average 2 tries.
        
        while True:
            x, y = random.choice(all_pairs)
            if (x, y) not in edge_set:
                break
                
        # Calculate delta
        # Edges affected by removing (u,v):
        # 1. (u,v) itself (removed)
        # 2. Edges (u, w) for w in N(u) \ {v}
        # 3. Edges (v, w) for w in N(v) \ {u}
        
        # Edges affected by adding (x,y):
        # 1. (x,y) itself (added)
        # 2. Edges (x, w) for w in N(x)
        # 3. Edges (y, w) for w in N(y)
        
        # We need to update the book sizes for these edges.
        # Let's collect all affected edges into a set.
        affected = set()
        
        # Edges incident to u or v
        for w in range(n):
            if w != u and w != v:
                if (u, w) in edge_set or (w, u) in edge_set: # Check if edge exists
                    # Note: edge_set contains (min, max)
                    e1 = (min(u, w), max(u, w))
                    if e1 in edge_set:
                        affected.add(e1)
                if (v, w) in edge_set or (w, v) in edge_set:
                    e2 = (min(v, w), max(v, w))
                    if e2 in edge_set:
                        affected.add(e2)
                        
        # Edges incident to x or y
        for w in range(n):
            if w != x and w != y:
                e3 = (min(x, w), max(x, w))
                if e3 in edge_set:
                    affected.add(e3)
                e4 = (min(y, w), max(y, w))
                if e4 in edge_set:
                    affected.add(e4)
                    
        # Also the new edge (x,y)
        new_edge = (min(x, y), max(x, y))
        # And the removed edge (u,v)
        old_edge = (min(u, v), max(u, v))
        
        # Save old book values
        old_books = {}
        for e in affected:
            if e in books:
                old_books[e] = books[e]
        
        # Perform swap
        remove_edge(u, v)
        edge_set.discard(old_edge)
        if old_edge in books:
            del books[old_edge]
            
        add_edge(x, y)
        edge_set.add(new_edge)
        
        # Update books for affected edges
        for e in affected:
            a, b = e
            books[e] = common_neighbors(a, b)
            
        # Add book for new edge
        books[new_edge] = common_neighbors(x, y)
        
        new_max = max(books.values())
        
        # Acceptance criterion
        if new_max < max_book:
            max_book = new_max
            if new_max < best_max:
                best_max = new_max
                best_edge_set = set(edge_set)
                best_adj = adj[:]
        else:
            # Revert
            remove_edge(x, y)
            edge_set.discard(new_edge)
            if new_edge in books:
                del books[new_edge]
                
            add_edge(u, v)
            edge_set.add(old_edge)
            
            # Restore books
            for e in affected:
                if e in old_books:
                    books[e] = old_books[e]
                else:
                    # This edge was not in books before, so it shouldn't be now?
                    # Wait, affected edges are those that EXISTED before or NOW?
                    # My 'affected' set construction was based on EXISTING edges before swap?
                    # No, I checked `if e in edge_set` BEFORE swap.
                    # So for edges that existed, I restored them.
                    # What about edges that didn't exist but are in 'affected'?
                    # The loop `for w in range(n)` added edges if they were in `edge_set`.
                    # So `affected` only contains edges that existed before the swap (except we didn't add the new one yet).
                    # So restoring from `old_books` should cover all existing edges in `affected`.
                    pass
            
            # Re-add old edge book
            books[old_edge] = old_books.get(old_edge, common_neighbors(u, v))
            
        T *= cooling
        
    return [[u, v] for u, v in best_edge_set]
