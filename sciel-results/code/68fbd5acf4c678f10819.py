import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 40
    m = 401
    random.seed(42)
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Helper to compute books
    def compute_books(adj_mask):
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                mask ^= lsb
                if u < v:
                    books[(u, v)] = (adj_mask[u] & adj_mask[v]).bit_count()
        return books

    # Helper to get max and sum of squares
    def get_metrics(books):
        if not books:
            return 0, 0
        max_book = max(books.values())
        sum_sq = sum(b * b for b in books.values())
        return max_book, sum_sq

    # Generate a regular-like initial graph
    def generate_regular_like_graph(n, m, seed):
        rng = random.Random(seed)
        target_deg = 2 * m / n  # ~20.05
        adj_mask = [0] * n
        edge_set = set()
        
        # Start with a random regular-ish graph
        # Use configuration model or simple balancing
        remaining = m
        for _ in range(remaining):
            # Pick edge that minimizes degree imbalance
            best_edge = None
            best_imbalance = float('inf')
            for _ in range(100):  # Sample 100 random edges
                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v or (u, v) in edge_set or (v, u) in edge_set:
                    continue
                du = bin(adj_mask[u]).count('1')
                dv = bin(adj_mask[v]).count('1')
                imbalance = abs(du - target_deg) + abs(dv - target_deg)
                if imbalance < best_imbalance:
                    best_imbalance = imbalance
                    best_edge = (min(u, v), max(u, v))
            if best_edge:
                u, v = best_edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edge_set.add(best_edge)
        
        # Fix if we didn't get enough edges
        while len(edge_set) < m:
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v or (u, v) in edge_set or (v, u) in edge_set:
                continue
            e = (min(u, v), max(u, v))
            adj_mask[e[0]] |= (1 << e[1])
            adj_mask[e[1]] |= (1 << e[0])
            edge_set.add(e)
        
        return adj_mask, edge_set

    # Generate initial graph
    adj_mask, edge_set = generate_regular_like_graph(n, m, 42)
    
    books = compute_books(adj_mask)
    current_max_book, current_sum_sq = get_metrics(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    T = 5.0
    T_min = 0.005
    cooling = 0.9995
    
    # Maintain non_edges list
    non_edges = [e for e in all_edges if e not in edge_set]
    
    # Track edges by book size for efficient selection
    def get_top_edges(books, k):
        sorted_books = sorted(books.items(), key=lambda x: -x[1])
        return [e for e, b in sorted_books[:k]]
    
    step_count = 0
    while time.time() < deadline and T > T_min:
        step_count += 1
        
        # Select edge to remove
        # Bias towards high book sizes
        if random.random() < 0.7:
            top_k = max(1, len(books) // 8)
            top_edges = get_top_edges(books, top_k)
            e_remove = random.choice(top_edges)
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        
        # Select edge to add
        # Sample candidates and pick best
        if not non_edges:
            break
            
        # Sample more candidates for better selection
        sample_size = min(80, len(non_edges))
        candidates = random.sample(non_edges, sample_size)
        
        best_add = None
        best_add_score = float('inf')
        
        for e in candidates:
            a, b = e
            # Score: number of common neighbors (creates triangles)
            score = (adj_mask[a] & adj_mask[b]).bit_count()
            # Also penalize if it would create a very high book
            if score < best_add_score:
                best_add_score = score
                best_add = e
                
        e_add = best_add
        
        if e_add == e_remove or e_add in edge_set:
            T *= cooling
            continue
            
        a, b = e_add
        
        # Compute delta in books more efficiently
        # When we remove (u,v) and add (a,b):
        # Books affected: all edges (u,x) for x in N(u)\{v}, (v,x) for x in N(v)\{u}
        # And all edges (a,x) for x in N(a), (b,x) for x in N(b)
        
        # For now, just recompute (faster than complex delta tracking)
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add(e_add)
        
        # Update non_edges
        non_edges.remove(e_add)
        non_edges.append(e_remove)
        
        # Recompute books
        new_books = compute_books(adj_mask)
        new_max_book, new_sum_sq = get_metrics(new_books)
        
        # Acceptance Criteria
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            if new_sum_sq < current_sum_sq:
                accept = True
            else:
                delta_sq = new_sum_sq - current_sum_sq
                if delta_sq > 0:
                    prob = math.exp(-delta_sq / (T * 50))
                    if random.random() < prob:
                        accept = True
                else:
                    accept = True
        else:
            delta_max = new_max_book - current_max_book
            prob = math.exp(-delta_max / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            current_max_book = new_max_book
            current_sum_sq = new_sum_sq
            books = new_books
            
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
        else:
            # Revert
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            
            edge_set.remove(e_add)
            edge_set.add(e_remove)
            
            # Revert non_edges
            non_edges.remove(e_remove)
            non_edges.append(e_add)
            
        T *= cooling
        
    return [list(e) for e in best_edge_set]
