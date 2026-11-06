import random
import time
import math

def construct():
    n = 40
    m = 401
    deadline = time.time() + 110.0  # Safety margin
    
    # Initialize random graph
    edges_set = set()
    while len(edges_set) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            if u > v: u, v = v, u
            edges_set.add((u, v))
            
    # Build adjacency list and bitsets
    adj = [set() for _ in range(n)]
    adj_bits = [0] * n
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        
    # Calculate book sizes
    # books maps normalized edge (u,v) with u<v to count of common neighbors
    books = {}
    for u in range(n):
        for v in adj[u]:
            if u < v:
                common = adj_bits[u] & adj_bits[v]
                books[(u, v)] = bin(common).count('1')
                
    current_max = max(books.values()) if books else 0
    best_edges = set(edges_set)
    best_max_book = current_max
    
    # Simulated Annealing
    T = 5.0
    decay = 0.995
    
    def get_affected_edges(u, w, z):
        """
        Get all edges that might change book size when swapping (u,w) with (u,z).
        The change is in the neighbor set of u, w, and z.
        Any edge incident to u, w, or z might change.
        """
        affected = set()
        # Edges incident to u
        for v in adj[u]:
            e = (min(u, v), max(u, v))
            affected.add(e)
        # Edges incident to w
        for v in adj[w]:
            e = (min(w, v), max(w, v))
            affected.add(e)
        # Edges incident to z
        for v in adj[z]:
            e = (min(z, v), max(z, v))
            affected.add(e)
        return affected

    def compute_books_for_edges(affected_edges):
        """
        Compute book sizes for a set of edges.
        Returns a dict of edge -> size.
        Also returns the max size among them.
        """
        new_books = {}
        max_b = 0
        for e in affected_edges:
            u, v = e
            # Check if edge still exists
            if v in adj[u]:
                common = adj_bits[u] & adj_bits[v]
                b = bin(common).count('1')
                new_books[e] = b
                if b > max_b:
                    max_b = b
            # If edge doesn't exist, it's not in new_books
        return new_books, max_b

    def update_global_max(old_max, old_books_subset, new_books_subset, all_books):
        """
        This is tricky because we only updated a subset.
        We need to know the max of the *unchanged* edges too.
        We can maintain a histogram or just recompute global max if the subset contains the old max.
        For simplicity and speed, since n=40 and m=401, recomputing global max is O(m) ~ 400 ops.
        This is fast enough to do every step if we don't have too many steps.
        However, we can be smarter.
        If the old max was in the affected set, we need to check the new max of affected + max of unaffected.
        We don't track max of unaffected easily.
        Let's just recompute global max from `books` dict after updating it.
        """
        pass

    step = 0
    while time.time() < deadline:
        step += 1
        
        # Pick a random edge to modify
        # To be effective, pick an edge that is part of a large book or just random?
        # Random is safer for exploration.
        # Let's pick a random edge (u, v) from the current graph.
        # Then pick a neighbor w of u to remove, and a non-neighbor z of u to add.
        
        # Get a random edge
        # We can pick a random vertex u, then a random neighbor v.
        u = random.randint(0, n-1)
        if not adj[u]:
            continue
        v = random.choice(list(adj[u]))
        
        # We are going to swap an edge incident to u.
        # Let's remove (u, w) where w is a neighbor of u.
        # And add (u, z) where z is not a neighbor of u.
        
        # To target high books, we could pick w such that (u,w) has high book size?
        # Or pick u,v with high book size, then modify u or v?
        # Let's stick to: Pick random u. Pick random neighbor w of u. Pick random non-neighbor z of u.
        
        if len(adj[u]) == n - 1:
            continue # u is connected to everyone, can't add z
        
        w = random.choice(list(adj[u]))
        
        # Find non-neighbors of u
        # non_neighbors = [x for x in range(n) if x not in adj[u] and x != u]
        # This is O(n). n=40. Fast.
        non_neighbors = [x for x in range(n) if x not in adj[u]]
        if not non_neighbors:
            continue
            
        z = random.choice(non_neighbors)
        
        # Ensure we are not just swapping (u,w) with (u,z) where z is already connected? 
        # z is non-neighbor, so (u,z) is not an edge.
        # w is neighbor, so (u,w) is an edge.
        
        # Perform the swap
        # 1. Remove (u, w)
        adj[u].remove(w)
        adj[w].remove(u)
        adj_bits[u] &= ~(1 << w)
        adj_bits[w] &= ~(1 << u)
        
        # 2. Add (u, z)
        adj[u].add(z)
        adj[z].add(u)
        adj_bits[u] |= (1 << z)
        adj_bits[z] |= (1 << u)
        
        # 3. Identify affected edges
        affected = get_affected_edges(u, w, z)
        
        # 4. Compute new book sizes for affected edges
        new_books_subset, new_max_in_affected = compute_books_for_edges(affected)
        
        # 5. Update the global books dictionary
        # First, delete old values for affected edges
        for e in affected:
            if e in books:
                del books[e]
        
        # Add new values
        for e, b in new_books_subset.items():
            books[e] = b
            
        # 6. Calculate new global max
        # Since we only updated a subset, we need to be careful.
        # If the previous global max was in the affected set, it might have changed.
        # If it wasn't, the global max is max(old_global_max, new_max_in_affected).
        # But we don't know if old_global_max was in affected.
        # Simplest: recompute max(books.values()).
        # O(m) = 401. Very fast.
        new_global_max = max(books.values()) if books else 0
        
        # 7. Acceptance
        if new_global_max <= current_max:
            # Accept
            current_max = new_global_max
            if current_max < best_max_book:
                best_max_book = current_max
                best_edges = set(books.keys()) # Keys are the edges
        else:
            # Reject or accept with probability
            delta = new_global_max - current_max
            if random.random() < math.exp(-delta / T):
                # Accept
                current_max = new_global_max
                if current_max < best_max_book:
                    best_max_book = current_max
                    best_edges = set(books.keys())
            else:
                # Revert
                # 1. Remove (u, z)
                adj[u].remove(z)
                adj[z].remove(u)
                adj_bits[u] &= ~(1 << z)
                adj_bits[z] &= ~(1 << u)
                
                # 2. Add (u, w)
                adj[u].add(w)
                adj[w].add(u)
                adj_bits[u] |= (1 << w)
                adj_bits[w] |= (1 << u)
                
                # 3. Restore books for affected edges
                # We need to recompute them or save old values.
                # Recomputing is safer.
                for e in affected:
                    if e in books:
                        del books[e]
                
                # Recompute for affected edges with restored state
                new_books_subset, _ = compute_books_for_edges(affected)
                for e, b in new_books_subset.items():
                    books[e] = b
                    
                # current_max should be restored to previous value
                # Since we reverted, the state is identical to before the move.
                # So current_max is still valid.

        T *= decay
        if T < 0.1:
            break
            
    # Return best found
    return [list(e) for e in best_edges]
