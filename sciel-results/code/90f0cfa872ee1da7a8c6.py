import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 40
    m = 401
    random.seed(42)
    
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

    # Initialization: Random graph with 401 edges
    # To be closer to regular, we can try to balance degrees initially
    # But random is often good enough for SA to fix.
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_edges)
    initial_edges = all_edges[:m]
    
    edge_set = set(initial_edges)
    adj_mask = [0] * n
    for u, v in initial_edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    books = compute_books(adj_mask)
    current_max_book, current_sum_sq = get_metrics(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute list of all edges for random selection
    # We need a list of edges currently in the graph for removal
    # And a list of edges NOT in the graph for addition
    
    # To speed up move selection, we can maintain lists
    # But re-building lists every step is O(m).
    # list(edge_set) is O(m).
    # [e for e in all_edges if e not in edge_set] is O(E_total).
    # E_total is 780. This is fine.
    
    while time.time() < deadline and T > T_min:
        # Select edge to remove
        # Bias towards high book sizes
        if random.random() < 0.8:
            # Get top 10% edges by book size
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            top_k = max(1, len(sorted_books) // 10)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        
        # Select edge to add
        # Bias towards low common neighbors (low triangle creation)
        # We want to add an edge (a, b) that minimizes common neighbors
        # But we can't check all 780-401=379 non-edges every time if it's slow.
        # Checking common neighbors is O(1) with bitmask.
        # So we can sample a few candidates and pick the best.
        
        # Sample 50 random non-edges
        non_edges = [e for e in all_edges if e not in edge_set]
        if not non_edges:
            break
            
        candidates = random.sample(non_edges, min(50, len(non_edges)))
        
        best_add = None
        best_add_score = float('inf')
        
        for e in candidates:
            a, b = e
            # Score is number of common neighbors
            score = (adj_mask[a] & adj_mask[b]).bit_count()
            if score < best_add_score:
                best_add_score = score
                best_add = e
                
        e_add = best_add
        
        if e_add == e_remove or e_add in edge_set:
            T *= cooling
            continue
            
        a, b = e_add
        
        # Apply move tentatively
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add(e_add)
        
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
                # Accept worse sum of squares with probability
                delta_sq = new_sum_sq - current_sum_sq
                if delta_sq > 0:
                    prob = math.exp(-delta_sq / (T * 100)) # Scale T
                    if random.random() < prob:
                        accept = True
                else:
                    accept = True
        else:
            # New max book is worse
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
            # books reverts implicitly because we didn't update 'books' variable
            # Wait, 'books' is the old books. We computed 'new_books' but didn't assign to 'books' unless accepted.
            # So 'books' is still correct for the reverted state.
            
        T *= cooling
        
    return [list(e) for e in best_edge_set]
