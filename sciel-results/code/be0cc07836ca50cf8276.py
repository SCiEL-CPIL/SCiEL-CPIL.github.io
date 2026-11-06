import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    deadline = time.time() + 115
    
    # Initialize
    adj = np.zeros((n, n), dtype=np.int8)
    edges = set()
    while len(edges) < m:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v and (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))
            edges.add((v, u))
            adj[u, v] = 1
            adj[v, u] = 1
    
    # Convert to list for random.choice
    edge_list = list(edges)
    
    def calc_books():
        books = np.zeros((n, n), dtype=np.int32)
        for u, v in edge_list:
            if u < v:
                common = np.sum(adj[u] & adj[v])
                books[u, v] = common
                books[v, u] = common
        return books

    books = calc_books()
    current_max = np.max(books)
    
    best_adj = adj.copy()
    best_edge_set = set(edge_list)
    best_max = current_max
    
    def update_remove(u, v):
        for w in range(n):
            if w != u and w != v:
                if adj[u, w] and adj[v, w]:
                    books[u, w] -= 1
                    books[v, w] -= 1
        books[u, v] = 0
        books[v, u] = 0
        adj[u, v] = 0
        adj[v, u] = 0
        edge_list.remove((u, v))
        edge_list.remove((v, u))
        edges.discard((u, v))
        edges.discard((v, u))

    def update_add(x, y):
        common = 0
        for w in range(n):
            if w != x and w != y:
                if adj[x, w] and adj[y, w]:
                    books[x, w] += 1
                    books[y, w] += 1
                    common += 1
        books[x, y] = common
        books[y, x] = common
        adj[x, y] = 1
        adj[y, x] = 1
        edge_list.append((x, y))
        edge_list.append((y, x))
        edges.add((x, y))
        edges.add((y, x))

    T = 1.0
    # Simple SA
    while time.time() < deadline:
        u, v = random.choice(edge_list)
        
        # Try 3 swaps
        for _ in range(3):
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y or adj[x, y] == 1:
                continue
            
            # Swap
            update_remove(u, v)
            update_add(x, y)
            
            new_max = np.max(books)
            
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_adj = adj.copy()
                    best_edge_set = set(edge_list)
                break
            else:
                # Revert
                update_remove(x, y)
                update_add(u, v)
        
        # Cool down
        if T > 0.1:
            T *= 0.99

    return [[u, v] for u, v in best_edge_set if u < v]
