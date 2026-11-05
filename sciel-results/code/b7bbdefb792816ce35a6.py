import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit
    
    n = 40
    m = 401
    random.seed(42)
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
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
    
    def get_metrics(books):
        if not books:
            return 0, 0
        max_book = max(books.values())
        sum_sq = sum(b * b for b in books.values())
        return max_book, sum_sq
    
    def generate_regular_graph(n, d, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        edge_set = set()
        
        # Start with random graph, then balance degrees
        all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_possible)
        
        # Take first n*d/2 edges
        target_edges = n * d // 2
        for u, v in all_possible[:target_edges]:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edge_set.add((u, v))
        
        # Balance degrees via edge moves
        degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
        
        for _ in range(500):
            max_idx = max(range(n), key=lambda i: degrees[i])
            min_idx = min(range(n), key=lambda i: degrees[i])
            
            if degrees[max_idx] - degrees[min_idx] < 2:
                break
            
            # Try to move an edge from max_idx to min_idx
            moved = False
            neigh_max = [j for j in range(n) if (adj_mask[max_idx] >> j) & 1]
            rng.shuffle(neigh_max)
            
            for u in neigh_max:
                if u == min_idx:
                    continue
                if not ((adj_mask[min_idx] >> u) & 1):
                    # Remove (max_idx, u), add (min_idx, u)
                    adj_mask[max_idx] &= ~(1 << u)
                    adj_mask[u] &= ~(1 << max_idx)
                    edge_set.discard((min(max_idx, u), max(max_idx, u)))
                    
                    adj_mask[min_idx] |= (1 << u)
                    adj_mask[u] |= (1 << min_idx)
                    edge_set.add((min(min_idx, u), max(min_idx, u)))
                    
                    degrees[max_idx] -= 1
                    degrees[min_idx] += 1
                    moved = True
                    break
            
            if not moved:
                break
        
        return adj_mask, edge_set
    
    # Initialize with a 20-regular graph (400 edges), then add 1 more
    adj_mask, edge_set = generate_regular_graph(n, 20, 42)
    
    # Add one more edge to reach 401
    missing = [e for e in all_edges if e not in edge_set]
    if missing:
        rng = random.Random(43)
        e_add = rng.choice(missing)
        u, v = e_add
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edge_set.add(e_add)
    
    books = compute_books(adj_mask)
    current_max_book, current_sum_sq = get_metrics(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    # Simulated annealing
    T = 3.0
    T_min = 0.01
    cooling = 0.9997
    
    non_edges = [e for e in all_edges if e not in edge_set]
    
    def get_top_edges(books, k):
        if k >= len(books):
            return list(books.keys())
        sorted_books = sorted(books.items(), key=lambda x: -x[1])
        return [e for e, b in sorted_books[:k]]
    
    while time.time() < deadline and T > T_min:
        # Select edge to remove - bias towards high book sizes
        if random.random() < 0.75:
            top_k = max(1, len(books) // 6)
            top_edges = get_top_edges(books, top_k)
            e_remove = random.choice(top_edges)
        else:
            e_remove = random.choice(list(edge_set))
        
        u, v = e_remove
        
        if not non_edges:
            break
        
        # Select edge to add - prefer edges with low common neighbors
        sample_size = min(60, len(non_edges))
        candidates = random.sample(non_edges, sample_size)
        
        best_add = None
        best_add_score = float('inf')
        
        for e in candidates:
            a, b = e
            score = (adj_mask[a] & adj_mask[b]).bit_count()
            if score < best_add_score:
                best_add_score = score
                best_add = e
        
        e_add = best_add
        
        if e_add is None or e_add == e_remove or e_add in edge_set:
            T *= cooling
            continue
        
        a, b = e_add
        
        # Apply the move
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add(e_add)
        
        non_edges.remove(e_add)
        non_edges.append(e_remove)
        
        new_books = compute_books(adj_mask)
        new_max_book, new_sum_sq = get_metrics(new_books)
        
        # Acceptance criteria
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            if new_sum_sq <= current_sum_sq:
                accept = True
            else:
                delta_sq = new_sum_sq - current_sum_sq
                prob = math.exp(-delta_sq / (T * 30))
                if random.random() < prob:
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
            
            non_edges.remove(e_remove)
            non_edges.append(e_add)
        
        T *= cooling
    
    # Final polish pass: targeted local search on edges with max book
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        adj_mask = list(best_adj_mask)
        edge_set = set(best_edge_set)
        books = compute_books(adj_mask)
        current_max_book, current_sum_sq = get_metrics(books)
        
        polish_T = 0.5
        polish_cooling = 0.999
        
        while time.time() < polish_deadline and polish_T > 0.005:
            # Find all edges with book size >= current_max_book - 1
            target_edges = [e for e, b in books.items() if b >= current_max_book - 1]
            
            if not target_edges:
                break
            
            e_remove = random.choice(target_edges)
            u, v = e_remove
            
            non_edges = [e for e in all_edges if e not in edge_set]
            if not non_edges:
                break
            
            sample_size = min(40, len(non_edges))
            candidates = random.sample(non_edges, sample_size)
            
            best_add = None
            best_add_score = float('inf')
            
            for e in candidates:
                a, b = e
                score = (adj_mask[a] & adj_mask[b]).bit_count()
                if score < best_add_score:
                    best_add_score = score
                    best_add = e
            
            if best_add is None or best_add == e_remove or best_add in edge_set:
                polish_T *= polish_cooling
                continue
            
            a, b = best_add
            
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            
            edge_set.remove(e_remove)
            edge_set.add(best_add)
            
            new_books = compute_books(adj_mask)
            new_max_book, new_sum_sq = get_metrics(new_books)
            
            accept = False
            if new_max_book < current_max_book:
                accept = True
            elif new_max_book == current_max_book:
                if new_sum_sq <= current_sum_sq:
                    accept = True
                else:
                    delta_sq = new_sum_sq - current_sum_sq
                    prob = math.exp(-delta_sq / (polish_T * 20))
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
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                adj_mask[a] &= ~(1 << b)
                adj_mask[b] &= ~(1 << a)
                
                edge_set.remove(best_add)
                edge_set.add(e_remove)
            
            polish_T *= polish_cooling
    
    return [list(e) for e in best_edge_set]
