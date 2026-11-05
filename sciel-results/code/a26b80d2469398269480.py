import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n = 40
    m = 401
    random.seed(42)
    
    # Bitmask representation for fast intersection
    adj_mask = [0] * n
    
    def compute_books(adj_mask):
        """Compute book size for each edge using bitmask intersection."""
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    common = adj_mask[u] & adj_mask[v]
                    books[(u, v)] = bin(common).count('1')
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    def get_sum_sq_books(books):
        return sum(b*b for b in books.values())

    def balanced_bipartite_init(n, m):
        """
        Initialize with a balanced bipartite graph plus internal structure.
        Split vertices into two groups of 20: A={0..19}, B={20..39}.
        Add all cross edges (400 edges), then add 1 internal edge.
        This gives a graph close to K_{20,20} with minimal internal structure.
        """
        adj_mask = [0] * n
        edge_set = set()
        
        # Add all bipartite edges between A and B
        for i in range(20):
            for j in range(20, 40):
                u, v = i, j
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        # We have 400 edges. Add 1 more edge.
        # Add an internal edge in A to break symmetry slightly
        # Choose edge (0, 1) within A
        u, v = 0, 1
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        return edge_set, adj_mask

    # Initialize with balanced bipartite structure
    edge_set, adj_mask = balanced_bipartite_init(n, m)
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    books = compute_books(adj_mask)
    current_max_book = get_max_book(books)
    current_sum_sq = get_sum_sq_books(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    # If we already hit 7, we can't do better.
    if best_max_book == 7:
        return [list(e) for e in best_edge_set]

    # Simulated annealing with aggressive cooling and targeted moves
    T = 3.0
    T_min = 0.05
    cooling = 0.999
    step_count = 0
    
    # Precompute non-edge list for faster sampling
    # We'll maintain this dynamically
    
    while time.time() < deadline and T > T_min:
        # Strategy: Focus on reducing the maximum book
        # Find edges with the highest book sizes
        sorted_books = sorted(books.items(), key=lambda x: -x[1])
        
        # Select an edge to remove: prefer high-book edges
        if random.random() < 0.7:
            # Pick from top 15% of edges by book size
            top_k = max(1, len(sorted_books) // 7)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        u, v = min(u, v), max(u, v)
        
        # Select an edge to add: prefer edges that would have low book size
        # Sample candidates and pick the one with minimum common neighbors
        candidates = []
        for _ in range(50):
            a, b = random.choice(all_possible)
            a, b = min(a, b), max(a, b)
            if (a, b) not in edge_set and (a, b) != e_remove:
                candidates.append((a, b))
        
        if not candidates:
            T *= cooling
            continue
            
        # Pick candidate with minimum common neighbors
        best_cand = None
        best_common = 100
        for a, b in candidates:
            common = bin(adj_mask[a] & adj_mask[b]).count('1')
            if common < best_common:
                best_common = common
                best_cand = (a, b)
        
        if best_cand is None:
            T *= cooling
            continue
            
        a, b = best_cand
        
        # Calculate affected vertices
        affected_verts = {u, v, a, b}
        
        # Save old book values for affected edges
        old_books = {}
        for e in edge_set:
            x, y = e
            if x in affected_verts or y in affected_verts:
                old_books[e] = books[e]
        
        # Apply the move
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add((a, b))
        
        # Recompute books for affected edges
        new_books = {}
        for e in edge_set:
            x, y = e
            if x in affected_verts or y in affected_verts:
                common = adj_mask[x] & adj_mask[y]
                new_books[e] = bin(common).count('1')
        
        # Update global books dict
        for e in old_books:
            if e in new_books:
                books[e] = new_books[e]
            else:
                del books[e]
        for e in new_books:
            if e not in old_books:
                books[e] = new_books[e]
                
        new_max_book = get_max_book(books)
        new_sum_sq = get_sum_sq_books(books)
        
        # Acceptance criterion
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            if new_sum_sq < current_sum_sq:
                accept = True
            else:
                # Allow small increases in sum_sq with temperature
                if random.random() < math.exp(-(new_sum_sq - current_sum_sq) / (T * 500)):
                    accept = True
        else:
            # Max book increased - only accept with high temperature
            if random.random() < math.exp(-(new_max_book - current_max_book) / T):
                accept = True
                
        if accept:
            current_max_book = new_max_book
            current_sum_sq = new_sum_sq
            
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
                
                if best_max_book == 7:
                    # We hit the theoretical minimum
                    return [list(e) for e in best_edge_set]
        else:
            # Revert
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            
            edge_set.remove((a, b))
            edge_set.add(e_remove)
            
            for e in new_books:
                if e in old_books:
                    books[e] = old_books[e]
                else:
                    del books[e]
            for e in old_books:
                if e not in new_books:
                    books[e] = old_books[e]
                    
        T *= cooling
        step_count += 1
        
    return [list(e) for e in best_edge_set]
