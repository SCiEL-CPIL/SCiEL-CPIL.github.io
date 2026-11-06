import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 40
    m = 401
    
    # We will try to find a graph with max book size 6.
    # If not found, we return the best found (likely 7).
    
    def compute_books(adj_mask):
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    common = adj_mask[u] & adj_mask[v]
                    books[(u, v)] = common.bit_count()
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    def get_sum_sq_books(books):
        return sum(b*b for b in books.values())

    def balanced_bipartite_init(seed_val):
        random.seed(seed_val)
        adj_mask = [0] * n
        edge_set = set()
        set_a = list(range(20))
        set_b = list(range(20, 40))
        for u in set_a:
            for v in set_b:
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        # Add 1 random edge within a set
        if random.random() < 0.5:
            u, v = random.sample(set_a, 2)
        else:
            u, v = random.sample(set_b, 2)
        u, v = min(u, v), max(u, v)
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        return edge_set, adj_mask

    def perturbed_bipartite_init(seed_val, perturb_count=5):
        edge_set, adj_mask = balanced_bipartite_init(seed_val)
        # Perform some random swaps to break local minima
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        for _ in range(perturb_count):
            # Remove a random edge
            e_remove = random.choice(list(edge_set))
            u, v = e_remove
            # Add a random non-edge
            while True:
                a, b = random.choice(all_possible)
                a, b = min(a, b), max(a, b)
                if (a, b) not in edge_set:
                    break
            edge_set.remove(e_remove)
            edge_set.add((a, b))
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
        return edge_set, adj_mask

    def random_regular_init(seed_val):
        random.seed(seed_val)
        adj_mask = [0] * n
        edge_set = set()
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible[:m]:
            edge_set.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        # Balance degrees
        degrees = [adj_mask[i].bit_count() for i in range(n)]
        for _ in range(200):
            max_idx = max(range(n), key=lambda i: degrees[i])
            min_idx = min(range(n), key=lambda i: degrees[i])
            if degrees[max_idx] - degrees[min_idx] < 2:
                break
            moved = False
            neigh_max = [j for j in range(n) if (adj_mask[max_idx] >> j) & 1]
            random.shuffle(neigh_max)
            for u in neigh_max:
                if u == min_idx:
                    continue
                if not ((adj_mask[min_idx] >> u) & 1):
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
        return edge_set, adj_mask

    def simulated_annealing(edge_set, adj_mask, target_max_book=6, max_steps=10000):
        books = compute_books(adj_mask)
        current_max_book = get_max_book(books)
        current_sum_sq = get_sum_sq_books(books)
        
        if current_max_book <= target_max_book:
            return edge_set, adj_mask, current_max_book

        T = 5.0
        T_min = 0.1
        cooling = 0.999
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        steps = 0
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            
            # Bias towards high book edges
            if random.random() < 0.8:
                top_k = max(1, len(sorted_books) // 5)
                e_remove = random.choice(sorted_books[:top_k])[0]
            else:
                e_remove = random.choice(list(edge_set))
                
            u, v = e_remove
            u, v = min(u, v), max(u, v)
            
            # Find candidate to add
            best_cand = None
            best_common = 100
            for _ in range(50):
                a, b = random.choice(all_possible)
                a, b = min(a, b), max(a, b)
                if (a, b) not in edge_set and (a, b) != e_remove:
                    common = (adj_mask[a] & adj_mask[b]).bit_count()
                    if common < best_common:
                        best_common = common
                        best_cand = (a, b)
            
            if not best_cand:
                continue
                
            a, b = best_cand
            affected_verts = {u, v, a, b}
            
            old_books = {e: books[e] for e in edge_set if e[0] in affected_verts or e[1] in affected_verts}
            
            # Apply move
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            edge_set.remove(e_remove)
            edge_set.add((a, b))
            
            # Recompute affected books
            new_books = {}
            for e in edge_set:
                if e[0] in affected_verts or e[1] in affected_verts:
                    new_books[e] = (adj_mask[e[0]] & adj_mask[e[1]]).bit_count()
            
            for e in old_books:
                if e in new_books:
                    books[e] = new_books[e]
                else:
                    del books[e]
            for e in new_books:
                if e not in old_books:
                    books[e] = new_books[e]
                    
            new_max_book = get_max_book(books)
            new_sum_sq = get_sum_sq_books(books)
            
            accept = False
            if new_max_book < current_max_book:
                accept = True
            elif new_max_book == current_max_book:
                if new_sum_sq < current_sum_sq:
                    accept = True
                else:
                    if random.random() < math.exp(-(new_sum_sq - current_sum_sq) / (T * 100)):
                        accept = True
            else:
                if random.random() < math.exp(-(new_max_book - current_max_book) / T):
                    accept = True
                        
            if accept:
                current_max_book = new_max_book
                current_sum_sq = new_sum_sq
                if current_max_book <= target_max_book:
                    return edge_set, adj_mask, current_max_book
            else:
                # Revert
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                adj_mask[a] &= ~(1 << b)
                adj_mask[b] &= ~(1 << a)
                edge_set.remove((a, b))
                edge_set.add(e_remove)
                for e in new_books:
                    if e in old_books:
                        books[e] = old_books[e]
                    else:
                        del books[e]
                for e in old_books:
                    if e not in new_books:
                        books[e] = old_books[e]
                        
            T *= cooling
            if T < T_min:
                T = T_min
                
        return edge_set, adj_mask, current_max_book

    best_edge_set = None
    best_adj_mask = None
    best_max_book = float('inf')
    
    # Try multiple initializations and SA runs
    # We will run SA on several different starts
    starts = []
    for i in range(5):
        starts.append(balanced_bipartite_init(100 + i))
    for i in range(5):
        starts.append(perturbed_bipartite_init(200 + i, perturb_count=10))
    starts.append(random_regular_init(300))
    
    for edge_set, adj_mask in starts:
        if time.time() > deadline - 10:
            break
        es, am, max_book = simulated_annealing(edge_set, adj_mask, target_max_book=6, max_steps=5000)
        if max_book < best_max_book:
            best_max_book = max_book
            best_edge_set = es
            best_adj_mask = am
            if best_max_book == 6:
                break
        
    if best_edge_set is None:
        # Fallback to first start
        best_edge_set, best_adj_mask = starts[0]
        best_max_book = get_max_book(compute_books(best_adj_mask))
        
    return [list(e) for e in best_edge_set]
