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

    def compute_books_vectorized(masks):
        """Vectorized computation of all book sizes using bit operations."""
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

    def batch_evaluate_swaps(masks, u, v, candidates):
        """
        Evaluate multiple swaps at once: remove edge (u,v), add each candidate edge.
        Returns list of (candidate, new_max_book) for each valid candidate.
        """
        results = []
        # Precompute the effect of removing (u,v)
        # When we remove (u,v), the book sizes of edges incident to u or v may decrease
        # Edges (u, x) where x is common neighbor of u and v: book decreases by 1
        # Edges (v, x) where x is common neighbor of u and v: book decreases by 1
        
        # For each candidate (x, y), we need to compute the new max book
        # This is expensive to do fully vectorized, but we can optimize
        
        for (x, y) in candidates:
            if x == u and y == v:
                continue
            if (masks[x] >> y) & 1:
                continue
            
            # Apply swap
            old_u = masks[u]
            old_v = masks[v]
            old_x = masks[x]
            old_y = masks[y]
            
            masks[u] ^= (1 << v)
            masks[v] ^= (1 << u)
            masks[x] |= (1 << y)
            masks[y] |= (1 << x)
            
            # Compute new max book efficiently
            # Only edges incident to u, v, x, y may have changed book sizes
            # For edges not incident to these, book size is unchanged
            
            new_max = 0
            
            # Check edges incident to u
            for j in range(N):
                if j != u and (masks[u] >> j) & 1:
                    b = (masks[u] & masks[j]).bit_count()
                    if b > new_max:
                        new_max = b
            
            # Check edges incident to v
            for j in range(N):
                if j != v and (masks[v] >> j) & 1:
                    b = (masks[v] & masks[j]).bit_count()
                    if b > new_max:
                        new_max = b
            
            # Check edges incident to x
            for j in range(N):
                if j != x and (masks[x] >> j) & 1:
                    b = (masks[x] & masks[j]).bit_count()
                    if b > new_max:
                        new_max = b
            
            # Check edges incident to y
            for j in range(N):
                if j != y and (masks[y] >> j) & 1:
                    b = (masks[y] & masks[j]).bit_count()
                    if b > new_max:
                        new_max = b
            
            # Revert
            masks[u] = old_u
            masks[v] = old_v
            masks[x] = old_x
            masks[y] = old_y
            
            results.append(((x, y), new_max))
        
        return results

    all_possible = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_masks = None
    best_max_book = float('inf')
    
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
            improved = False
            
            # Generate a batch of candidate non-edges
            # Sample from all non-edges
            non_edges = []
            for (i, j) in all_possible:
                if not ((masks[i] >> j) & 1):
                    if not (i == u and j == v):
                        non_edges.append((i, j))
            
            if not non_edges:
                break
            
            # Sample a subset of candidates for batch evaluation
            batch_size = min(50, len(non_edges))
            candidates = random.sample(non_edges, batch_size)
            
            # Evaluate all candidates in batch
            eval_results = batch_evaluate_swaps(masks, u, v, candidates)
            
            # Find the best candidate
            best_candidate = None
            best_new_max = max_val
            
            for (cand, new_max) in eval_results:
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_candidate = cand
            
            if best_candidate is not None:
                # Apply the best swap
                x, y = best_candidate
                masks[u] ^= (1 << v)
                masks[v] ^= (1 << u)
                masks[x] |= (1 << y)
                masks[y] |= (1 << x)
                
                # Recompute books
                current_books, max_val = compute_books(masks)
                improved = True
                stuck_count = 0
            else:
                # No improvement found in this batch
                stuck_count += 1
                if stuck_count > 15:
                    break # Stuck, restart
            
    return get_edges(best_masks)
