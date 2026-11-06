import time
import random
import numpy as np

def construct():
    N = 40
    M = 401
    # Safe deadline to avoid timeout
    deadline = time.time() + 110
    
    def get_edges(masks):
        edges = []
        for i in range(N):
            m = masks[i]
            for j in range(i + 1, N):
                if (m >> j) & 1:
                    edges.append([i, j])
        return edges

    def compute_books(masks):
        books = {}
        max_b = 0
        for i in range(N):
            m_i = masks[i]
            for j in range(i + 1, N):
                if (m_i >> j) & 1:
                    b = (m_i & masks[j]).bit_count()
                    books[(i, j)] = b
                    if b > max_b:
                        max_b = b
        return books, max_b
    
    def compute_books_vectorized(adj):
        """Compute all book sizes using numpy adjacency matrix"""
        books = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i, j]:
                    b = int(np.dot(adj[i], adj[j]))
                    books[i, j] = b
                    books[j, i] = b
        return books

    all_possible = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_masks = None
    best_max_book = float('inf')
    
    # Precompute non-edges for each edge to enable vectorized swap evaluation
    def get_non_edges(masks):
        non_edges = set()
        for i in range(N):
            m_i = masks[i]
            for j in range(i + 1, N):
                if not ((m_i >> j) & 1):
                    non_edges.add((i, j))
        return non_edges
    
    def evaluate_swap(masks, edge_to_remove, edge_to_add):
        """Evaluate the change in max book when swapping edges"""
        u, v = edge_to_remove
        x, y = edge_to_add
        
        # Compute delta in book sizes for affected edges
        # Edges affected are those sharing a vertex with (u,v) or (x,y)
        
        # For now, just compute the new max book directly (simpler)
        masks[u] ^= (1 << v)
        masks[v] ^= (1 << u)
        masks[x] |= (1 << y)
        masks[y] |= (1 << x)
        
        new_books, new_max_book = compute_books(masks)
        
        # Revert
        masks[u] ^= (1 << v)
        masks[v] ^= (1 << u)
        masks[x] ^= (1 << y)
        masks[y] ^= (1 << x)
        
        return new_max_book
    
    # Multiple restarts of local search
    # Each restart uses a fresh random graph
    while time.time() < deadline:
        # Initialize random graph
        masks = [0] * N
        random.shuffle(all_possible)
        for u, v in all_possible[:M]:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            
        current_books, current_max_book = compute_books(masks)
        
        if current_max_book < best_max_book:
            best_max_book = current_max_book
            best_masks = masks[:]
            
        # Local Search: Target the max edge
        # We try to reduce the max book size by swapping the edge with the largest book
        # with a random non-edge.
        
        # Limit the number of steps per restart to ensure progress
        max_steps = 300 
        steps = 0
        stuck_count = 0
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            
            # Find the edge with the maximum book size
            max_edge = None
            max_val = -1
            for e, b in current_books.items():
                if b > max_val:
                    max_val = b
                    max_edge = e
            
            if max_val < best_max_book:
                best_max_book = max_val
                best_masks = masks[:]
                
            # If we hit the theoretical lower bound (approx 7), we can stop early
            if max_val <= 7:
                break
                
            u, v = max_edge
            
            # Get all non-edges
            non_edges = get_non_edges(masks)
            non_edge_list = list(non_edges)
            
            if not non_edge_list:
                break
            
            # Vectorized batch evaluation: test many candidates at once
            batch_size = min(200, len(non_edge_list))
            candidates = random.sample(non_edge_list, batch_size)
            
            best_candidate = None
            best_new_max = max_val
            
            for x, y in candidates:
                # Skip if it's the same edge or adjacent in a way that doesn't help
                if (x == u and y == v) or (x == v and y == u):
                    continue
                if x == u or x == v or y == u or y == v:
                    # This swap will change books of many edges, might be worth testing
                    pass
                
                new_max_book = evaluate_swap(masks, (u, v), (x, y))
                
                if new_max_book < best_new_max:
                    best_new_max = new_max_book
                    best_candidate = (x, y)
            
            if best_candidate is not None and best_new_max < max_val:
                # Apply the best swap
                x, y = best_candidate
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
                masks[x] |= (1 << y)
                masks[y] |= (1 << x)
                
                current_books, max_val = compute_books(masks)
                stuck_count = 0
            else:
                # Try neutral moves or restart
                stuck_count += 1
                if stuck_count > 15:
                    break # Stuck, restart
            
    return get_edges(best_masks)
