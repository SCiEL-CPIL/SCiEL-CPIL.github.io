import random
import numpy as np
import time

def construct():
    n = 40
    target_edges = 401
    
    def calc_books(adj):
        books = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    books[u, v] = bin(adj[u] & adj[v]).count('1')
                    books[v, u] = books[u, v]
        return books

    def get_max_book(books):
        return np.max(books)

    def get_sum_sq_books(books):
        return np.sum(books * books)

    # Initialize with a random graph
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_max_book = float('inf')
    best_edges = None
    
    # Use multiple restarts to avoid local minima
    for restart in range(10):
        random.seed(42 + restart)
        random.shuffle(all_edges)
        edges = all_edges[:target_edges]
        
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        books = calc_books(adj)
        current_max_book = get_max_book(books)
        current_sum_sq = get_sum_sq_books(books)
        
        # Local Search
        # We will try to reduce the max book size by swapping edges.
        # Move: Remove (u,v), Add (x,y).
        # Constraint: (x,y) must not be an existing edge, and (u,v) must be an existing edge.
        # Objective: Reduce max book size. If max book size stays same, reduce sum of squares.
        
        no_improve_count = 0
        max_iterations = 20000
        T = 1.0 # Temperature for simulated annealing-like acceptance
        
        for it in range(max_iterations):
            if no_improve_count > 5000:
                break
                
            # Pick a random existing edge to remove
            # We prioritize removing edges with high book size
            # Create a list of existing edges
            existing_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        existing_edges.append((u, v))
                        
            if not existing_edges:
                break
                
            # Weighted selection: higher book size -> higher probability of removal
            weights = [books[u, v] + 1 for u, v in existing_edges]
            u, v = random.choices(existing_edges, weights=weights)[0]
            
            # Pick a random non-existing edge to add
            # We prioritize adding edges with low book size potential
            # This is tricky to sample directly. Let's just pick random non-edge.
            # To speed up, we can precompute non-edges, but n is small.
            # Let's pick random u2, v2 until we find a non-edge.
            
            # Optimization: Instead of random search for non-edge, 
            # we can pick from the complement.
            # But for simplicity and speed, let's just try random pairs.
            # Since graph is dense (401/780 ~ 51%), random pair is non-edge with ~50% chance.
            
            added = False
            for _ in range(50): # Try 50 times to find a non-edge
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if not (adj[x] & (1 << y)):
                    # Check if (x,y) is the same as (u,v)
                    if (x == u and y == v) or (x == v and y == u):
                        continue
                    # Valid non-edge
                    break
            else:
                continue # Failed to find non-edge in 50 tries
            
            # Perform swap
            # Remove (u,v)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            # Add (x,y)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)
            
            # Recalculate books
            # Efficient update:
            # Removing (u,v) decreases book size of (u,w) and (v,w) for all common neighbors w.
            # Adding (x,y) increases book size of (x,w) and (y,w) for all common neighbors w.
            # However, full recalculation is O(n^2) which is 1600 ops. 20000 iters * 1600 = 32M ops.
            # This might be slow in Python.
            # Let's try to update incrementally.
            
            # Actually, let's just recalculate books for the affected nodes?
            # Or just recalculate all books? 
            # 20000 iterations * 40*40/2 edges * popcount... 
            # bin(x).count('1') is fast but not instant.
            # Let's optimize the book calculation using numpy or bit operations.
            
            # Let's try a different strategy: 
            # Only accept if max book decreases, or if max book is same and sum_sq decreases.
            
            new_books = calc_books(adj)
            new_max_book = get_max_book(new_books)
            new_sum_sq = get_sum_sq_books(new_books)
            
            delta_max = new_max_book - current_max_book
            delta_sum = new_sum_sq - current_sum_sq
            
            accept = False
            if delta_max < 0:
                accept = True
            elif delta_max == 0 and delta_sum < 0:
                accept = True
            elif delta_max < 0 and delta_sum < 0:
                accept = True
            else:
                # Simulated annealing acceptance for plateau breaking
                # If delta_max == 0 and delta_sum > 0, accept with prob exp(-delta_sum / T)
                if delta_max == 0:
                    prob = np.exp(-delta_sum / T)
                    if random.random() < prob:
                        accept = True
                # If delta_max > 0, never accept (we want to minimize max)
                
            if accept:
                current_max_book = new_max_book
                current_sum_sq = new_sum_sq
                books = new_books
                no_improve_count = 0
            else:
                # Revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                adj[x] &= ~(1 << y)
                adj[y] &= ~(1 << x)
                no_improve_count += 1
                
            # Cool down
            if it % 100 == 0:
                T *= 0.95
                
        # End local search
        if current_max_book < best_max_book:
            best_max_book = current_max_book
            # Reconstruct edge list from adj
            best_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        best_edges.append([u, v])
                        
            if best_max_book <= 7:
                # Theoretical lower bound is 7 (40/6 = 6.66, so integer >= 7)
                # If we hit 7, it's optimal.
                return best_edges

    # Return best found
    if best_edges:
        return best_edges
    
    # Fallback
    random.shuffle(all_edges)
    return [list(e) for e in all_edges[:target_edges]]
