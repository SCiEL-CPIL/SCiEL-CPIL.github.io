import time
import random
import math

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 118.0

    # Bitmask helpers
    def get_common_mask(adj, u, v):
        return adj[u] & adj[v]

    def compute_initial_state(edges, adj):
        book = [[0] * n for _ in range(n)]
        book_count = [0] * (n + 1)
        max_book = 0
        
        for u, v in edges:
            common = adj[u] & adj[v]
            b = common.bit_count()
            book[u][v] = b
            book[v][u] = b
            book_count[b] += 1
            if b > max_book:
                max_book = b
        return book, book_count, max_book

    def update_book_after_add(book, book_count, max_book, adj, u, v):
        # Add edge (u, v)
        common = adj[u] & adj[v]
        b = common.bit_count()
        
        # Update book sizes for edges (u, w) and (v, w) for w in common
        T = common
        while T:
            lsb = T & -T
            w = lsb.bit_length() - 1
            T ^= lsb
            
            # Edge (u, w)
            old_b = book[u][w]
            book_count[old_b] -= 1
            book[u][w] = old_b + 1
            book[w][u] = old_b + 1
            book_count[old_b + 1] += 1
            if old_b + 1 > max_book:
                max_book = old_b + 1
                
            # Edge (v, w)
            old_b = book[v][w]
            book_count[old_b] -= 1
            book[v][w] = old_b + 1
            book[w][v] = old_b + 1
            book_count[old_b + 1] += 1
            if old_b + 1 > max_book:
                max_book = old_b + 1
                
        # The edge (u, v) itself
        book[u][v] = b
        book[v][u] = b
        book_count[b] += 1
        if b > max_book:
            max_book = b
            
        return book, book_count, max_book

    def update_book_after_remove(book, book_count, max_book, adj, u, v):
        # Remove edge (u, v)
        common = adj[u] & adj[v]
        b = book[u][v]
        
        # Remove edge (u, v) from count
        book_count[b] -= 1
        book[u][v] = 0
        book[v][u] = 0
        
        # Update book sizes for edges (u, w) and (v, w) for w in common
        T = common
        while T:
            lsb = T & -T
            w = lsb.bit_length() - 1
            T ^= lsb
            
            # Edge (u, w)
            old_b = book[u][w]
            book_count[old_b] -= 1
            book[u][w] = old_b - 1
            book[w][u] = old_b - 1
            book_count[old_b - 1] += 1
            
            # Edge (v, w)
            old_b = book[v][w]
            book_count[old_b] -= 1
            book[v][w] = old_b - 1
            book[w][v] = old_b - 1
            book_count[old_b - 1] += 1

        # If the removed edge was the max, we might need to recompute max
        # But since we only decrease other books, the new max is either the old max (if another edge had it)
        # or lower.
        # Efficiently finding new max:
        if book_count[max_book] == 0:
            while max_book > 0 and book_count[max_book] == 0:
                max_book -= 1
                
        return book, book_count, max_book

    def simulate_swap(edges, adj, book, book_count, max_book, u, v, x, y):
        # Simulate removing (u,v) and adding (x,y)
        # We need to be careful with the order of updates if they interact.
        # (u,v) and (x,y) are distinct edges.
        # Common neighbors of (u,v) and (x,y) might overlap.
        
        # 1. Remove (u,v)
        common_uv = adj[u] & adj[v]
        b_uv = book[u][v]
        
        # Temporarily decrement counts for edges affected by removal
        # We will track changes in a dictionary or list to revert if rejected
        
        changes = [] # (book, i, j, delta)
        
        # Effect of removing (u,v):
        # book[u][v] becomes 0
        changes.append((u, v, -b_uv))
        changes.append((v, u, -b_uv))
        
        T = common_uv
        while T:
            lsb = T & -T
            w = lsb.bit_length() - 1
            T ^= lsb
            # book[u][w] decreases by 1
            changes.append((u, w, -1))
            changes.append((w, u, -1))
            # book[v][w] decreases by 1
            changes.append((v, w, -1))
            changes.append((w, v, -1))
            
        # 2. Add (x,y)
        common_xy = adj[x] & adj[y]
        b_xy = common_xy.bit_count()
        
        # Effect of adding (x,y):
        # book[x][y] becomes b_xy
        changes.append((x, y, b_xy))
        changes.append((y, x, b_xy))
        
        T = common_xy
        while T:
            lsb = T & -T
            w = lsb.bit_length() - 1
            T ^= lsb
            # book[x][w] increases by 1
            changes.append((x, w, 1))
            changes.append((w, x, 1))
            # book[y][w] increases by 1
            changes.append((y, w, 1))
            changes.append((w, y, 1))
            
        # Calculate new max book size
        # We need to know the new values of all affected books.
        # Affected books are those in the set of edges touched by changes.
        # Let's compute the new max by checking the new values of these specific edges.
        
        # To do this efficiently, we can just apply the changes to a copy of the relevant book values?
        # Or, we can calculate the new max from scratch for the affected edges?
        # The number of affected edges is at most 2 * (deg(u) + deg(v) + deg(x) + deg(y)) which is small.
        
        # However, we also need to know if any *unaffected* edge was the max and remains the max.
        # So new_max = max( max(affected_new_values), max_unaffected_old_value )
        
        # max_unaffected_old_value is just max_book, UNLESS the edge that had max_book is affected.
        # If the edge with max_book is affected, its value changes, so we must recompute.
        
        # Simple approach:
        # 1. Identify the set of edges that change.
        # 2. Calculate their new values.
        # 3. The new global max is max( max(new_values), max_book_if_max_edge_unaffected )
        
        # To check if max_edge is unaffected:
        # We don't track which edge has max_book.
        # So we must assume max_book might change.
        # We can calculate the new max by taking the max of:
        # - The new values of all affected edges.
        # - The old max_book (if the edge holding it is not affected).
        
        # Since we don't know if the max edge is affected, we can't just use old max_book.
        # But we can note that if we remove an edge with book size < max_book, and add an edge with book size < max_book,
        # and no other edge increases to > max_book, then max_book stays the same.
        
        # Let's just compute the new max for the affected edges, and also keep track of the old max.
        # If the old max edge is not in the affected set, the new max is max(old_max, max_affected_new).
        # If the old max edge IS in the affected set, we must ensure we don't miss it.
        # Actually, if the old max edge is affected, its new value is in `affected_new`.
        # So `max(affected_new)` covers it.
        # The only risk is if an *unaffected* edge had the max_book, and it is still the max.
        # In that case, `max_affected_new` might be lower than `max_book`.
        # So `new_max = max(max_affected_new, max_book)` is correct ONLY IF the edge with `max_book` is unaffected OR if it is affected and its new value is included in `max_affected_new`.
        # Wait. If the edge with `max_book` is unaffected, its value is `max_book`. So `max(..., max_book)` is correct.
        # If the edge with `max_book` is affected, its new value is in `affected_new`. So `max(affected_new)` includes it.
        # So `new_max = max(max_affected_new, max_book)` is ALWAYS correct?
        # No. If the edge with `max_book` is affected and its value *decreases*, then `max_book` is no longer a valid value for that edge.
        # But `max_book` as a *number* might still be the max of *other* unaffected edges.
        # Example: Edge A has book 10 (max). Edge B has book 9.
        # We affect Edge A, reducing it to 5.
        # `affected_new` = [5].
        # `max_book` = 10.
        # `max(5, 10) = 10`.
        # But the true max is 9 (from Edge B).
        # So we need to know if there is another edge with book size 10.
        
        # To handle this correctly, we should maintain `book_count`.
        # In the simulation, we can track the changes in `book_count` for the affected edges.
        
        # Let's do a lightweight simulation of `book_count` for the affected edges.
        
        # This is getting complex. Given n=40, we can just recompute the max book size from scratch for the whole graph?
        # No, that's O(m) = 401 operations.
        # If we do 100,000 iterations, that's 40 million ops. Feasible in Python? Maybe slow.
        # 40 million simple ops in Python might take 2-4 seconds.
        # We have 118 seconds. So we can afford ~20-30 million recomputations? No, 401 * 10000 = 4 million.
        # So we can do ~100,000 full recomputations.
        
        # Let's try a simpler SA that recomputes max book size from scratch on every accepted move?
        # Or even on every proposed move?
        # Proposed moves: 100,000.
        # Cost per proposal: O(n^2) to compute all book sizes?
        # No, we can compute book sizes in O(n^2) using bitmasks?
        # For each pair (u,v), `book[u][v] = (adj[u] & adj[v]).bit_count()`.
        # There are 780 pairs.
        # 780 bit counts per move.
        # 100,000 moves * 780 = 78 million bit counts.
        # Python `bit_count` is fast. 78 million might take 10-20 seconds.
        # This is a viable strategy: **Full Recomputation SA**.
        
    def compute_all_books(adj):
        book = [[0] * n for _ in range(n)]
        max_b = 0
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1: # Check if edge exists
                    b = (adj[u] & adj[v]).bit_count()
                    book[u][v] = b
                    book[v][u] = b
                    if b > max_b:
                        max_b = b
        return book, max_b

    # Strategy:
    # 1. Init: Random graph with 401 edges.
    # 2. SA Loop:
    #    - Pick random edge to remove, random non-edge to add.
    #    - Compute new max book size (full recompute).
    #    - Accept if better or with prob exp(-delta/T).
    
    # To speed up, we can use the incremental update for accepted moves to keep `book` matrix updated,
    # but use full recompute to check the new max for the *proposed* move?
    # No, full recompute is the bottleneck.
    
    # Let's use the incremental update for the *current* state, and for the *proposed* state,
    # we can estimate the new max.
    # But to be safe and simple, let's try the full recompute approach first, but limit iterations.
    
    edges = set()
    all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
    random.shuffle(all_possible)
    for u, v in all_possible:
        if len(edges) >= m:
            break
        edges.add((u, v))
        
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    book, current_max = compute_all_books(adj)
    best_max = current_max
    best_edges = set(edges)
    best_adj = list(adj)
    
    T = 5.0
    T_end = 0.01
    max_iters = 50000
    cooling = (T_end / T) ** (1.0 / max_iters)
    
    non_edges = set(all_possible) - edges
    
    for it in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Pick random edge to remove
        a, b = random.choice(list(edges))
        
        # Pick random non-edge to add
        if not non_edges:
            break
        c, d = random.choice(list(non_edges))
        
        # Simulate swap
        # Temporarily update adj
        adj[a] &= ~(1 << b)
        adj[b] &= ~(1 << a)
        adj[c] |= 1 << d
        adj[d] |= 1 << c
        
        # Compute new max book
        new_book, new_max = compute_all_books(adj)
        
        # Delta
        delta = new_max - current_max
        
        # Acceptance
        accept = False
        if delta <= 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta / max(T, 1e-10))
            
        if accept:
            edges.discard((a, b))
            edges.add((c, d))
            non_edges.discard((c, d))
            non_edges.add((a, b))
            
            current_max = new_max
            book = new_book
            
            if current_max < best_max:
                best_max = current_max
                best_edges = set(edges)
                best_adj = list(adj)
        else:
            # Revert adj
            adj[a] |= 1 << b
            adj[b] |= 1 << a
            adj[c] &= ~(1 << d)
            adj[d] &= ~(1 << c)
            
        T *= cooling
        
    return [[u, v] for u, v in best_edges]

# Run construct
# The function construct is the entry point.
