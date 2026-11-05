import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    n = 40
    m = 401
    target_degree = 20.05
    
    # Precompute all possible edges for initialization
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def compute_max_book(adj):
        max_book = 0
        # adj is n x n boolean matrix
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    common = np.sum(adj[i] & adj[j])
                    if common > max_book:
                        max_book = common
        return max_book

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges

    def degrees(adj):
        return np.sum(adj, axis=1)

    def degree_penalty(adj):
        degs = degrees(adj)
        # Penalize deviation from target degree
        return np.sum((degs - target_degree) ** 2)

    # Fast incremental update for book sizes
    # book[u][v] = number of common neighbors of u and v (if edge exists)
    # We can recompute book sizes for affected edges only.
    # Affected edges: (u, k) for k in N(u), (v, k) for k in N(v), and (u, v) itself.
    
    def update_books(adj, book, u, v):
        # Remove edge (u, v) from consideration for book size? 
        # Actually, book size is defined for edges that exist.
        # If we flip (u, v), we need to update book sizes for edges incident to u or v.
        
        # Get neighbors of u and v
        nu = np.where(adj[u])[0]
        nv = np.where(adj[v])[0]
        
        # Edges to update: all (x, y) where x in {u, v} U (N(u) U N(v))
        # Actually, simpler: just recompute book sizes for all edges incident to u or v.
        # And the edge (u, v) itself if it becomes an edge.
        
        # 1. Update book[u][v] and book[v][u]
        if adj[u][v]:
            book[u][v] = np.sum(adj[u] & adj[v])
            book[v][u] = book[u][v]
        else:
            book[u][v] = 0
            book[v][u] = 0
            
        # 2. Update books for edges (u, w) where w is neighbor of u
        for w in nu:
            if w == v: continue
            if adj[u][w]:
                book[u][w] = np.sum(adj[u] & adj[w])
                book[w][u] = book[u][w]
            else:
                book[u][w] = 0
                book[w][u] = 0
                
        # 3. Update books for edges (v, w) where w is neighbor of v
        for w in nv:
            if w == u: continue
            if adj[v][w]:
                book[v][w] = np.sum(adj[v] & adj[w])
                book[w][v] = book[v][w]
            else:
                book[v][w] = 0
                book[w][v] = 0

    def compute_books(adj):
        book = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    b = np.sum(adj[i] & adj[j])
                    book[i][j] = b
                    book[j][i] = b
        return book

    def get_max_book(book):
        return np.max(book)

    # Initialization: Random graph with 401 edges, biased towards regularity
    # We can start with a random 20-regular graph if possible, or just random.
    # Let's try a few different initializations.
    
    best_adj = None
    best_score = float('inf')
    best_penalty = float('inf')
    
    # Number of restarts
    num_restarts = 0
    time_per_restart = 1.0 # approx
    
    while time.time() < deadline - 5:
        num_restarts += 1
        
        # Initialize
        adj = np.zeros((n, n), dtype=bool)
        
        # Strategy: Start with K_{20,20} + 1 edge, then perturb?
        # Or random?
        # Let's try random regular-like.
        # Shuffle vertices and create a circulant?
        # Circulants are structured. Random is more general.
        
        # Let's use a random graph with p=0.5, then adjust edge count.
        # To ensure 401 edges:
        # 1. Generate all edges.
        # 2. Shuffle.
        # 3. Take first 401.
        
        random_edges = random.sample(all_edges, m)
        for u, v in random_edges:
            adj[u][v] = True
            adj[v][u] = True
            
        # Compute initial books
        book = compute_books(adj)
        max_book = get_max_book(book)
        penalty = degree_penalty(adj)
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.999
        current_score = max_book
        current_penalty = penalty
        current_total = current_score + 0.1 * current_penalty # Weighted objective
        
        best_local_adj = adj.copy()
        best_local_score = max_book
        best_local_penalty = penalty
        best_local_total = current_total
        
        steps = 0
        max_steps = 100000 # per restart, but time limited
        
        while time.time() < deadline - 2 and T > T_min:
            steps += 1
            
            # Pick a random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Check if flip changes edge count? No, flip keeps count same.
            # Wait, flipping an edge changes the graph.
            # If adj[u][v] is True, we remove it. If False, we add it.
            # This changes the edge count!
            # We need to keep edge count exactly 401.
            # So we must do a SWAP: remove one edge, add another.
            
            # Let's pick a second edge (x, y) to swap with.
            # If we remove (u, v) and add (x, y), count stays same.
            
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
                
            # Ensure (x, y) is not (u, v)
            if (u, v) == (x, y):
                continue
                
            # Ensure (u, v) is an edge to remove, and (x, y) is NOT an edge to add.
            if not adj[u][v]:
                # (u, v) is not an edge, so we can't remove it.
                # Swap roles: remove (x, y), add (u, v).
                if not adj[x][y]:
                    # Neither is an edge? Then we can't swap.
                    continue
                else:
                    # (x, y) is an edge, (u, v) is not.
                    # Remove (x, y), add (u, v).
                    u, v = x, y
                    x, y = u, v # This logic is getting messy.
                    # Let's just pick a random edge to remove and a random non-edge to add.
                    pass
            
            # Better approach:
            # 1. Pick a random edge (u, v) from the graph.
            # 2. Pick a random non-edge (x, y) from the graph.
            # 3. Swap them.
            
            # To pick random edge:
            # We can maintain a list of edges, but that's expensive to update.
            # Or just sample until we find one.
            
            found_remove = False
            found_add = False
            ru = rv = rx = ry = -1
            
            # Sample remove edge
            while not found_remove:
                ru = random.randint(0, n-1)
                rv = random.randint(0, n-1)
                if ru != rv and adj[ru][rv]:
                    if ru > rv: ru, rv = rv, ru
                    found_remove = True
                    
            # Sample add edge
            while not found_add:
                rx = random.randint(0, n-1)
                ry = random.randint(0, n-1)
                if rx != ry and not adj[rx][ry]:
                    if rx > ry: rx, ry = ry, rx
                    if (ru, rv) == (rx, ry): # Should not happen if one is edge and other isn't
                        continue
                    found_add = True
            
            # Perform swap: remove (ru, rv), add (rx, ry)
            adj[ru][rv] = False
            adj[rv][ru] = False
            adj[rx][ry] = True
            adj[ry][rx] = True
            
            # Update books incrementally
            # This is complex. Let's just recompute books for affected edges.
            # Or, for simplicity and given n=40, recompute full book matrix?
            # 40x40 matrix, 800 entries. np.sum on 40-bit arrays is fast.
            # But we need to be fast.
            
            # Let's try the incremental update function defined earlier.
            # It updates books for edges incident to ru, rv, rx, ry.
            
            # Actually, the incremental update function `update_books` I wrote above
            # assumes a single edge flip. Here we have two flips.
            # We can call it twice?
            # 1. Remove (ru, rv)
            # 2. Add (rx, ry)
            
            # But the adjacency matrix `adj` has already been modified.
            # So `update_books` needs to know the OLD state to compute changes?
            # No, it computes the NEW book size based on the NEW adj matrix.
            # But it only updates specific entries.
            # If we call it for (ru, rv) first, it updates books for edges incident to ru, rv.
            # Then we call it for (rx, ry), it updates books for edges incident to rx, ry.
            # This should cover all affected edges.
            
            update_books(adj, book, ru, rv)
            update_books(adj, book, rx, ry)
            
            new_max_book = get_max_book(book)
            new_penalty = degree_penalty(adj)
            new_total = new_max_book + 0.1 * new_penalty
            
            delta = new_total - current_total
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_score = new_max_book
                current_penalty = new_penalty
                current_total = new_total
                
                if new_total < best_local_total:
                    best_local_total = new_total
                    best_local_score = new_max_book
                    best_local_penalty = new_penalty
                    best_local_adj = adj.copy()
            else:
                # Reject: revert changes
                adj[ru][rv] = True
                adj[rv][ru] = True
                adj[rx][ry] = False
                adj[ry][rx] = False
                
                # Revert book updates
                # This is tricky. We need to restore the book values.
                # Easiest: recompute books for affected edges from the reverted adj.
                # But `update_books` overwrites.
                # So we just call `update_books` again with the reverted adj?
                # No, `update_books` computes from `adj`.
                # Since `adj` is reverted, calling `update_books` again will restore correct values.
                
                update_books(adj, book, ru, rv)
                update_books(adj, book, rx, ry)
                
            T *= cooling_rate
            
            # Early stop if we hit theoretical min?
            if best_local_score <= 7:
                break
                
        # End of restart
        if best_local_total < best_score:
            best_score = best_local_total
            best_adj = best_local_adj.copy()
            
        if best_score <= 7 + 0.1 * degree_penalty(best_adj):
            # We found a very good solution.
            # Note: score is max_book. The evaluator measures raw value (max_book).
            # So we should track best_max_book, not best_total.
            pass

    # Final check: return the graph with the lowest max_book
    # We tracked best_total which includes penalty.
    # We should find the graph with the lowest max_book among the best candidates.
    # Or just return best_adj and let the evaluator measure it.
    # But we want to minimize max_book.
    
    # Let's re-evaluate the best_adj to be sure.
    if best_adj is None:
        # Fallback
        adj = np.zeros((n, n), dtype=bool)
        for i in range(20):
            for j in range(20, 40):
                adj[i][j] = True
                adj[j][i] = True
        adj[0][1] = True
        adj[1][0] = True
        best_adj = adj

    return get_edge_list(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
