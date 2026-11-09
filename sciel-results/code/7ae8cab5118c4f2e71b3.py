import random
import time
import math

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 130  # 130s deadline (10s margin before 150s kill)
    
    def count_books(adj):
        """Compute book size for each edge. Returns dict (u,v)->count for u<v."""
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    # Count common neighbors
                    cnt = 0
                    for w in range(n):
                        if adj[u][w] and adj[v][w]:
                            cnt += 1
                    books[(u,v)] = cnt
        return books
    
    def get_max_book(adj):
        books = count_books(adj)
        if not books:
            return 0
        return max(books.values())
    
    # Start from K_{20,20} + 1 extra edge
    # Vertices 0-19 in part A, 20-39 in part B
    adj = [[0]*n for _ in range(n)]
    for a in range(20):
        for b in range(20, 40):
            adj[a][b] = 1
            adj[b][a] = 1
    # Add one extra edge within part A
    adj[0][1] = 1
    adj[1][0] = 1
    
    current_edges = sum(adj[i][j] for i in range(n) for j in range(i+1, n))
    # Should be 400 + 1 = 401
    
    best_adj = [row[:] for row in adj]
    best_max = get_max_book(adj)
    
    # Hill climbing with simulated annealing
    T = 0.5
    T_min = 0.01
    cooling = 0.9999
    
    iterations = 0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        iterations += 1
        
        if no_improve > max_no_improve:
            # Restart from best
            adj = [row[:] for row in best_adj]
            no_improve = 0
            T = max(T, 0.1)
            max_no_improve = int(max_no_improve * 1.2)
        
        # Random move: remove one edge, add another
        # Pick a random existing edge to remove
        existing_edges = [(i,j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
        if not existing_edges:
            break
        
        rem_u, rem_v = random.choice(existing_edges)
        
        # Pick a random non-edge to add
        non_edges = [(i,j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        if not non_edges:
            break
        
        add_u, add_v = random.choice(non_edges)
        
        if rem_u == add_u and rem_v == add_v:
            continue
        
        # Compute delta in max book size
        # This is expensive; let's do a quick estimate first
        # Remove (rem_u, rem_v) and add (add_u, add_v)
        
        # Save old state
        old_rem = adj[rem_u][rem_v]
        old_add = adj[add_u][add_v]
        
        # Remove edge
        adj[rem_u][rem_v] = 0
        adj[rem_v][rem_u] = 0
        # Add edge
        adj[add_u][add_v] = 1
        adj[add_v][add_u] = 1
        
        # Compute new max book - this is O(n^3) which is 64000 ops, acceptable
        new_books = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    cnt = 0
                    for w in range(n):
                        if adj[u][w] and adj[v][w]:
                            cnt += 1
                    new_books[(u,v)] = cnt
                    if cnt > 0:
                        pass
        
        new_max = max(new_books.values()) if new_books else 0
        
        # Acceptance
        if new_max <= best_max or (T > T_min and random.random() < math.exp(-(new_max - best_max) / T)):
            if new_max < best_max:
                best_max = new_max
                best_adj = [row[:] for row in adj]
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Revert
            adj[rem_u][rem_v] = old_rem
            adj[rem_v][rem_u] = old_rem
            adj[add_u][add_v] = old_add
            adj[add_v][add_u] = old_add
            no_improve += 1
        
        T *= cooling
    
    # Return best found
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                result.append([i, j])
    
    return result
