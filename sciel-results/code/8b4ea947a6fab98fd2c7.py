import time
import random
import math

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115

    def build_books(adj, edges):
        books = {}
        for u, v in edges:
            books[(u, v)] = (adj[u] & adj[v]).bit_count()
        return books

    def get_max_sum(books):
        if not books:
            return 0, 0
        mx = max(books.values())
        s = sum(books.values())
        return mx, s

    def make_random_init(seed):
        """Start from a random graph with 401 edges."""
        rng = random.Random(seed)
        adj = [0] * n
        edges = set()
        all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_possible)
        for e in all_possible[:target_edges]:
            u, v = e
            edges.add(e)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj, edges

    def local_search(adj, edges, seed, time_limit):
        rng = random.Random(seed)
        books = build_books(adj, edges)
        cur_max, cur_sum = get_max_sum(books)
        best_max, best_sum = cur_max, cur_sum
        best_edges = edges.copy()
        best_adj = adj[:]
        
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        stall = 0
        max_stall = 1000
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if T <= min_T:
                break
            
            # Pick edge to remove: bias toward high-book edges
            edge_list = list(books.keys())
            if rng.random() < 0.7:
                # Remove from top 20% by book size
                sorted_edges = sorted(edge_list, key=lambda e: books[e], reverse=True)
                top_k = max(1, len(sorted_edges) // 5)
                e_rem = rng.choice(sorted_edges[:top_k])
            else:
                e_rem = rng.choice(edge_list)
            
            u1, v1 = e_rem
            
            # Save
            adj_bak = adj[:]
            books_bak = books.copy()
            edges_bak = edges.copy()
            
            # Remove edge
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            edges.discard(e_rem)
            del books[e_rem]
            
            # Update books for affected edges
            for w in range(n):
                if w == u1 or w == v1:
                    continue
                if adj[u1] & (1 << w):
                    k = (min(u1, w), max(u1, w))
                    if k in books and (adj[v1] & (1 << w)):
                        books[k] -= 1
                if adj[v1] & (1 << w):
                    k = (min(v1, w), max(v1, w))
                    if k in books and (adj[u1] & (1 << w)):
                        books[k] -= 1
            
            # Find best add: sample many non-edges
            best_add = None
            best_pot = 999
            for _ in range(100):
                u2 = rng.randint(0, n - 1)
                v2 = rng.randint(0, n - 1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2, v2), max(u2, v2)
                if (u2, v2) in edges:
                    continue
                pot = (adj[u2] & adj[v2]).bit_count()
                if pot < best_pot:
                    best_pot = pot
                    best_add = (u2, v2)
            
            if best_add is None:
                adj = adj_bak
                books = books_bak
                edges = edges_bak
                T *= cooling
                continue
            
            u2, v2 = best_add
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            edges.add(best_add)
            books[best_add] = best_pot
            
            for w in range(n):
                if w == u2 or w == v2:
                    continue
                if adj[u2] & (1 << w):
                    k = (min(u2, w), max(u2, w))
                    if k in books and (adj[v2] & (1 << w)):
                        books[k] += 1
                if adj[v2] & (1 << w):
                    k = (min(v2, w), max(v2, w))
                    if k in books and (adj[u2] & (1 << w)):
                        books[k] += 1
            
            new_max, new_sum = get_max_sum(books)
            
            # Accept/reject
            accept = False
            if new_max < cur_max:
                accept = True
                stall = 0
            elif new_max == cur_max and new_sum < cur_sum:
                accept = True
                stall = 0
            else:
                delta = (new_max - cur_max) * 10 + (new_sum - cur_sum)
                if delta < 0:
                    accept = True
                    stall = 0
                else:
                    if rng.random() < math.exp(-delta / T):
                        accept = True
                    else:
                        stall += 1
                        if stall > max_stall:
                            T = max(T * 0.5, min_T)
                            stall = 0
            
            if accept:
                cur_max, cur_sum = new_max, new_sum
                if cur_max < best_max or (cur_max == best_max and cur_sum < best_sum):
                    best_max, best_sum = cur_max, cur_sum
                    best_edges = edges.copy()
                    best_adj = adj[:]
            else:
                adj = adj_bak
                books = books_bak
                edges = edges_bak
            
            T *= cooling
            
            if best_max <= 7:
                break
        
        return best_max, best_sum, best_edges, best_adj

    global_best = None
    global_best_max = 999
    global_best_sum = 999
    
    # Phase 1: Random initializations with local search
    # Use more seeds and shorter time per seed to explore more
    for seed in range(50):
        if time.time() > deadline - 15:
            break
        adj, edges = make_random_init(seed)
        tm = max(0.5, (deadline - 15 - time.time()) / 50)
        bm, bs, be, ba = local_search(adj, edges, seed, tm)
        if bm < global_best_max or (bm == global_best_max and bs < global_best_sum):
            global_best_max, global_best_sum = bm, bs
            global_best = be
        if global_best_max <= 7:
            break
    
    # Phase 2: If we haven't reached 7, try bipartite init (maybe it converges faster from there?)
    if global_best_max > 7:
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            rng = random.Random(seed)
            adj = [0] * n
            edges = set()
            for a in range(20):
                for b in range(20, 40):
                    edges.add((a, b))
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            part = rng.choice([list(range(20)), list(range(20, 40))])
            u, v = rng.sample(part, 2)
            u, v = min(u, v), max(u, v)
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            tm = max(1.0, (deadline - 10 - time.time()) / 20)
            bm, bs, be, ba = local_search(adj, edges, seed + 1000, tm)
            if bm < global_best_max or (bm == global_best_max and bs < global_best_sum):
                global_best_max, global_best_sum = bm, bs
                global_best = be
            if global_best_max <= 7:
                break
    
    if global_best is None:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return [[u, v] for (u, v) in sorted(global_best)]
