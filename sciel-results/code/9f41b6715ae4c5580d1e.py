import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    
    n = 40
    target_edges = 401
    
    # Initialization: K20,20 + 1 random intra-partition edge
    adj = [0] * n
    edges = set()
    books = {} # key: (u, v) with u < v, value: book size
    
    def add_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        if (u, v) not in edges:
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            # Calculate initial book size
            common = adj[u] & adj[v]
            books[(u, v)] = common.bit_count()
            
    def remove_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        if (u, v) in edges:
            edges.remove((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            del books[(u, v)]

    # Build K20,20
    for u in range(20):
        for v in range(20, 40):
            add_edge(u, v)
            
    # Add 1 more edge (intra-partition)
    # To ensure we start with a valid configuration and potentially better local structure,
    # we pick a random pair in A or B.
    while True:
        u = random.randint(0, 19)
        v = random.randint(0, 19)
        if u != v:
            add_edge(u, v)
            break
    # If random failed (unlikely), fallback
    if len(edges) < 401:
         add_edge(0, 1)

    # Ensure exactly 401 edges
    while len(edges) < 401:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v and (min(u,v), max(u,v)) not in edges:
            add_edge(u, v)
            
    # Helper to get current max book and sum of squares from books dict
    def get_stats():
        max_b = 0
        sum_sq = 0
        for b in books.values():
            if b > max_b: max_b = b
            sum_sq += b * b
        return max_b, sum_sq

    current_max, current_sum_sq = get_stats()
    best_max = current_max
    best_sum_sq = current_sum_sq
    best_edges = edges.copy()
    best_books = books.copy()
    
    # Simulated Annealing parameters
    T = 1.0
    cooling = 0.9999
    
    # Pre-compute list of all possible edges for faster non-edge selection?
    # There are 40*39/2 = 780 possible edges.
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start_time < time_limit:
        # Pick an edge to remove: 
        # Bias towards removing edges with high book sizes to reduce max book quickly
        # But also keep some randomness.
        if random.random() < 0.5:
            # Remove an edge with high book size
            # Find max book edge
            max_edge = max(books, key=books.get)
            # With some probability pick a random one from top 10%
            sorted_edges = sorted(books.keys(), key=lambda x: books[x], reverse=True)
            top_k = max(1, len(sorted_edges)//10)
            if random.random() < 0.8:
                u_rem, v_rem = sorted_edges[random.randint(0, top_k-1)]
            else:
                u_rem, v_rem = random.choice(list(edges))
        else:
            u_rem, v_rem = random.choice(list(edges))
        
        # Find a random non-edge
        # Using a set of edges allows O(1) check
        while True:
            u_add = random.randint(0, n-1)
            v_add = random.randint(0, n-1)
            if u_add == v_add: continue
            if u_add > v_add: u_add, v_add = v_add, u_add
            if (u_add, v_add) not in edges:
                break
        
        # Perform swap tentatively
        # We need to update books for all edges incident to u_rem, v_rem, u_add, v_add
        # Let's collect affected edges
        affected_edges = set()
        
        # Edges incident to u_rem
        for w in range(n):
            if adj[u_rem] & (1 << w):
                a, b = min(u_rem, w), max(u_rem, w)
                affected_edges.add((a, b))
        # Edges incident to v_rem
        for w in range(n):
            if adj[v_rem] & (1 << w):
                a, b = min(v_rem, w), max(v_rem, w)
                affected_edges.add((a, b))
        # Edges incident to u_add
        for w in range(n):
            if adj[u_add] & (1 << w):
                a, b = min(u_add, w), max(u_add, w)
                affected_edges.add((a, b))
        # Edges incident to v_add
        for w in range(n):
            if adj[v_add] & (1 << w):
                a, b = min(v_add, w), max(v_add, w)
                affected_edges.add((a, b))
        
        # Store old values
        old_books = {e: books.get(e) for e in affected_edges}
        
        remove_edge(u_rem, v_rem)
        add_edge(u_add, v_add)
        
        # Recalculate books for affected edges
        # Note: add_edge already calculated book for (u_add, v_add).
        # We need to recalculate for other affected edges because their common neighbors changed.
        for e in affected_edges:
            u, v = e
            if (u, v) in edges:
                common = adj[u] & adj[v]
                books[(u, v)] = common.bit_count()
            # If (u,v) was removed and is in affected_edges, it's already deleted from books by remove_edge
            # But if it's still there (e.g. self loop? no, simple graph), we should ensure it's gone.
            # remove_edge deletes (u_rem, v_rem) from books.
            # If e == (u_rem, v_rem), it is not in edges anymore.
            
        new_max, new_sum_sq = get_stats()
        
        # Acceptance criteria
        accepted = False
        if new_max < best_max:
            accepted = True
        elif new_max == best_max and new_sum_sq < best_sum_sq:
            accepted = True
        elif new_max <= best_max:
            # Simulated annealing acceptance for equal or slightly worse?
            # Actually, if new_max > best_max, we might still accept it to escape local minima.
            # Let's define delta.
            delta = new_sum_sq - best_sum_sq
            # If max book increased, penalty is huge.
            if new_max > best_max:
                # Only accept if we are early in the process or with very low probability
                if T > 0.01:
                    # Probability based on how much worse it is
                    # Simple heuristic: accept with prob exp(-delta/T) if delta is small
                    # But max book increase is discrete.
                    # Let's just accept with small prob if new_max == best_max + 1
                    if random.random() < T * 0.1:
                         accepted = True
            else:
                 # new_max <= best_max
                 # If new_max == best_max and new_sum_sq >= best_sum_sq
                 # Accept with prob exp(-(new_sum_sq - best_sum_sq)/T)
                 if random.random() < math.exp(-delta / T) if T > 0 else 0:
                     accepted = True
                     # Actually, if delta is positive, exp is < 1. If delta is negative, exp > 1 (accept).
                     # We already handled new_sum_sq < best_sum_sq above.
                     # So here delta >= 0.
        
        if accepted:
            best_max = new_max
            best_sum_sq = new_sum_sq
            best_edges = edges.copy()
            best_books = books.copy()
        else:
            # Rollback
            remove_edge(u_add, v_add)
            add_edge(u_rem, v_rem)
            # Restore books
            for e, val in old_books.items():
                if val is not None:
                    books[e] = val
                # If val is None, it was not in books, so it shouldn't be added back unless it's the removed edge?
                # Wait, remove_edge deleted (u_rem, v_rem). add_edge added (u_add, v_add).
                # The affected_edges set includes edges that existed before OR after.
                # If an edge was removed, it's in old_books with its value.
                # If an edge was added, it's in old_books with None (if it didn't exist before) or its old value (if it did? no, it was a non-edge).
                # Actually, if (u_add, v_add) was a non-edge, it wasn't in books. So old_books.get((u_add, v_add)) is None.
                # When we rollback, we remove (u_add, v_add). It is deleted from books.
                # We add (u_rem, v_rem). It is added to books with new value.
                # But we need to restore the *old* values for other affected edges.
                # The logic above `books[e] = val` restores them.
                # What about (u_rem, v_rem)? It was in old_books. So it gets restored.
                # What about (u_add, v_add)? It was NOT in old_books (val is None). So `if val is not None` skips it.
                # But add_edge(u_rem, v_rem) adds it to books.
                # So books state is consistent.
        
        T *= cooling
        if T < 1e-6: T = 1e-6

    # Restore best solution
    edges = best_edges
    books = best_books
    
    # Format output
    return [[u, v] for u, v in edges]
