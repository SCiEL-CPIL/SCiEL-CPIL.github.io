import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 148.0
    
    n = 40
    random.seed(42)
    
    # Use bitmask representation for fast intersection
    # adj_mask[i] is a 40-bit integer where bit j is set if edge (i,j) exists
    
    def random_graph_with_m_edges(m):
        """Generate a random graph with exactly m edges."""
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        chosen = all_edges[:m]
        edge_set = set()
        adj_mask = [0] * n
        for u, v in chosen:
            edge_set.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return edge_set, adj_mask
    
    def compute_all_books(edge_set, adj_mask):
        """Compute book size for each edge using bitmask intersection."""
        books = {}
        for (u, v) in edge_set:
            common = adj_mask[u] & adj_mask[v]
            books[(u, v)] = bin(common).count('1')
        return books
    
    def get_max_book(books):
        return max(books.values()) if books else 0
    
    # Initialize with a random graph (spreads triangles more evenly than K20,20)
    edge_set, adj_mask = random_graph_with_m_edges(401)
    books = compute_all_books(edge_set, adj_mask)
    current_book = get_max_book(books)
    best_book = current_book
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    # Precompute all possible edges as tuples
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    all_edges_set = set(all_edges)
    
    T = 3.0
    T_min = 0.05
    cooling = 0.9998
    
    def try_move(e_remove, e_add):
        """Try a move: remove e_remove, add e_add. Returns (delta, affected_books)."""
        u, v = e_remove
        a, b = e_add
        
        # Check validity
        if e_add in edge_set:
            return None
        
        # Identify affected vertices
        affected = {u, v, a, b}
        
        # Identify affected edges (edges incident to affected vertices)
        affected_edges = set()
        for x in affected:
            mask = adj_mask[x]
            # Iterate over set bits
            m = mask
            while m:
                y = (m & -m).bit_length() - 1
                m &= m - 1
                e = (min(x, y), max(x, y))
                if e in edge_set or e == e_add:
                    affected_edges.add(e)
        
        # Store old book sizes for affected edges
        old_books = {}
        for e in affected_edges:
            if e in books:
                old_books[e] = books[e]
        
        # Apply move to adj_mask
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        # Update books for affected edges
        new_books = {}
        for e in affected_edges:
            if e == e_remove:
                continue  # removed
            x, y = e
            if e == e_add or e in edge_set:
                common = adj_mask[x] & adj_mask[y]
                new_books[e] = bin(common).count('1')
        
        # Compute new global max
        # For edges not in affected_edges, their book size is unchanged
        # We need max of:
        #   - new_books values
        #   - old books for unaffected edges
        
        # Find max among unaffected edges
        max_unaffected = 0
        for e, b_val in books.items():
            if e not in affected_edges and e != e_remove:
                if b_val > max_unaffected:
                    max_unaffected = b_val
        
        max_affected = max(new_books.values()) if new_books else 0
        new_global_max = max(max_unaffected, max_affected)
        
        # Revert adj_mask
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        adj_mask[a] &= ~(1 << b)
        adj_mask[b] &= ~(1 << a)
        
        delta = new_global_max - current_book
        return (delta, affected_edges, new_books, old_books)
    
    # Main SA loop
    while time.time() < deadline and T > T_min:
        # Smart move selection: prefer removing edges with high book sizes
        # to reduce the maximum
        if random.random() < 0.6:
            # Remove an edge with high book size
            sorted_edges = sorted(books.items(), key=lambda x: -x[1])
            # Pick from top 20%
            top_k = max(1, len(sorted_edges) // 5)
            e_remove = random.choice(sorted_edges[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
        
        # Choose edge to add: prefer edges that create fewer triangles
        # or random
        if random.random() < 0.5:
            # Random add
            e_add = random.choice(all_edges)
        else:
            # Try to add edge with low common neighbors
            candidates = [e for e in all_edges if e not in edge_set]
            if not candidates:
                T *= cooling
                continue
            # Score by common neighbors (lower is better)
            def edge_score(e):
                x, y = e
                return bin(adj_mask[x] & adj_mask[y]).count('1')
            # Pick from lowest scoring candidates
            candidates.sort(key=edge_score)
            k = min(len(candidates), 50)
            e_add = random.choice(candidates[:k])
        
        if e_add == e_remove or e_add in edge_set:
            T *= cooling
            continue
        
        result = try_move(e_remove, e_add)
        if result is None:
            T *= cooling
            continue
        
        delta, affected_edges, new_books, old_books = result
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept: apply the move
            u, v = e_remove
            a, b = e_add
            
            edge_set.remove(e_remove)
            edge_set.add(e_add)
            
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            
            # Update books dictionary
            for e in affected_edges:
                if e == e_remove:
                    if e in books:
                        del books[e]
                elif e in new_books:
                    books[e] = new_books[e]
                elif e in books:
                    # Recompute if needed
                    x, y = e
                    books[e] = bin(adj_mask[x] & adj_mask[y]).count('1')
            
            # Make sure e_add is in books
            if e_add not in books:
                x, y = e_add
                books[e_add] = bin(adj_mask[x] & adj_mask[y]).count('1')
            
            current_book = get_max_book(books)
            if current_book < best_book:
                best_book = current_book
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
        # If rejected, no changes needed (try_move already reverted)
        
        T *= cooling
    
    return [list(e) for e in best_edge_set]
