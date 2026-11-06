import time
import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115.0
    
    best_max = 999
    best_adj = None
    
    def compute_books(adj):
        """Compute book size for each edge. Returns (books_dict, max_book)."""
        A = adj.astype(np.int32)
        A2 = A @ A
        books = {}
        max_book = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    b = int(A2[i, j])
                    books[(i, j)] = b
                    if b > max_book:
                        max_book = b
        return books, max_book
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append((i, j))
        return edges
    
    def evaluate_swap(adj, books, rem_u, rem_v, new_u, new_v):
        """Hypothetically remove (rem_u,rem_v) and add (new_u,new_v).
        Returns (new_books, new_max) or None if invalid. Does not modify state."""
        if adj[new_u, new_v]:
            return None
        if (new_u, new_v) == (rem_u, rem_v):
            return None
        
        common_add = np.where(adj[new_u] & adj[new_v])[0]
        new_book = len(common_add)
        
        common_rm = np.where(adj[rem_u] & adj[rem_v])[0]
        
        new_books = dict(books)
        
        old_key = (min(rem_u, rem_v), max(rem_u, rem_v))
        if old_key not in new_books:
            return None
        del new_books[old_key]
        
        for w in common_rm:
            if w == rem_u or w == rem_v:
                continue
            k1 = (min(rem_u, w), max(rem_u, w))
            if k1 in new_books:
                new_books[k1] -= 1
            k2 = (min(rem_v, w), max(rem_v, w))
            if k2 in new_books:
                new_books[k2] -= 1
        
        for z in common_add:
            if z == new_u or z == new_v:
                continue
            k1 = (min(new_u, z), max(new_u, z))
            if k1 in new_books:
                new_books[k1] += 1
            k2 = (min(new_v, z), max(new_v, z))
            if k2 in new_books:
                new_books[k2] += 1
        
        new_key = (min(new_u, new_v), max(new_u, new_v))
        new_books[new_key] = new_book
        
        new_max = max(new_books.values()) if new_books else 0
        return new_books, new_max
    
    # Generate initial configurations
    configs = []
    
    # Random 20-regular graphs via configuration model
    for seed in range(30):
        random.seed(seed)
        adj = np.zeros((n, n), dtype=bool)
        degree = 20
        stubs = []
        for v in range(n):
            for _ in range(degree):
                stubs.append(v)
        random.shuffle(stubs)
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u != v and not adj[u, v]:
                adj[u, v] = True
                adj[v, u] = True
        edge_count = np.sum(adj) // 2
        if edge_count == 400:
            non_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i, j]:
                        non_edges.append((i, j))
            if non_edges:
                u, v = random.choice(non_edges)
                adj[u, v] = True
                adj[v, u] = True
            configs.append(adj.copy())
    
    # Near-bipartite configurations
    for seed in range(15):
        random.seed(200 + seed)
        adj = np.zeros((n, n), dtype=bool)
        for a in range(20):
            for b in range(20, 40):
                adj[a, b] = True
                adj[b, a] = True
        k = random.randint(5, 15)
        bip_edges = [(a, b) for a in range(20) for b in range(20, 40)]
        random.shuffle(bip_edges)
        for i in range(k):
            u, v = bip_edges[i]
            adj[u, v] = False
            adj[v, u] = False
        
        intra_a = [(a, b) for a in range(20) for b in range(a+1, 20)]
        intra_b = [(a, b) for a in range(20, 40) for b in range(a+1, 40)]
        all_intra = intra_a + intra_b
        random.shuffle(all_intra)
        for i in range(k + 1):
            u, v = all_intra[i]
            adj[u, v] = True
            adj[v, u] = True
        
        if np.sum(adj) // 2 == 401:
            configs.append(adj.copy())
    
    # Hill climb each configuration
    for init_adj in configs:
        if time.time() > deadline:
            break
        
        adj = init_adj.copy()
        books, current_max = compute_books(adj)
        
        if current_max < best_max:
            best_max = current_max
            best_adj = adj.copy()
        
        if current_max <= 7:
            continue
        
        edges = get_edges(adj)
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < deadline and no_improve < max_no_improve:
            improved = False
            
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                rem_u, rem_v = random.choice(edges)
                new_u = random.randint(0, n-1)
                new_v = random.randint(0, n-1)
                if new_u == new_v:
                    continue
                if new_u > new_v:
                    new_u, new_v = new_v, new_u
                if adj[new_u, new_v]:
                    continue
                if (new_u, new_v) == (rem_u, rem_v):
                    continue
                
                result = evaluate_swap(adj, books, rem_u, rem_v, new_u, new_v)
                if result is None:
                    continue
                
                new_books, new_max = result
                
                if new_max < current_max:
                    adj[rem_u, rem_v] = False
                    adj[rem_v, rem_u] = False
                    adj[new_u, new_v] = True
                    adj[new_v, new_u] = True
                    current_max = new_max
                    books = new_books
                    edges.remove((rem_u, rem_v))
                    eu = min(new_u, new_v)
                    ev = max(new_u, new_v)
                    edges.append((eu, ev))
                    no_improve = 0
                    improved = True
                    if current_max < best_max:
                        best_max = current_max
                        best_adj = adj.copy()
                    break
            
            if not improved:
                no_improve += 1
            
            if current_max <= 7:
                break
    
    if best_adj is None:
        adj = np.zeros((n, n), dtype=bool)
        for a in range(20):
            for b in range(20, 40):
                adj[a, b] = True
                adj[b, a] = True
        adj[0, 1] = True
        adj[1, 0] = True
        best_adj = adj
    
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                result.append([i, j])
    
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    
    return result
# EVOLVE-BLOCK-END
