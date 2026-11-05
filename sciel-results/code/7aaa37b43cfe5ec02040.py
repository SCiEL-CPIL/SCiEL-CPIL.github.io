import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n = 40
    m = 401

    def build_bitmasks(edge_set):
        adj = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def book(u, v, adj):
        return bin(adj[u] & adj[v]).count('1')

    def max_book(books):
        return max(books.values()) if books else 0

    def all_books(edge_set, adj):
        return {(u, v): book(u, v, adj) for (u, v) in edge_set}

    def circulant_init():
        """Highly symmetric, regular, non-bipartite start (measured to help)."""
        edge_set = set()
        for i in range(n):
            for d in (1, 2, 3, 4, 5):
                j = (i + d) % n
                if i < j:
                    edge_set.add((i, j))
        edges = list(edge_set)
        random.shuffle(edges)
        for e in edges:
            if len(edge_set) >= m:
                break
            edge_set.add(e)
        # top up deterministically if short (shouldn't happen: circulant r=5 has 100 edges)
        while len(edge_set) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u < v:
                edge_set.add((u, v))
        return edge_set

    def near_regular_init():
        """
        Initialize a near-regular graph using greedy degree balancing.
        This targets a flatter degree distribution, which is structurally distinct
        from the circulant start and aims to avoid concentrated high-degree hubs
        that often drive up book sizes.
        """
        edge_set = set()
        deg = [0] * n
        
        # Start with a cycle to ensure connectivity
        for i in range(n):
            j = (i + 1) % n
            if i < j:
                edge_set.add((i, j))
                deg[i] += 1
                deg[j] += 1
        
        # Add remaining edges greedily, preferring pairs with lower degrees
        remaining = m - len(edge_set)
        attempts = 0
        max_attempts = 100000
        
        while len(edge_set) < m and attempts < max_attempts:
            attempts += 1
            # Select vertices with lowest degrees
            # To make this efficient and random-ish, we can pick a random pair
            # but bias towards low degree.
            # A simple heuristic: pick u randomly, then pick v from low-degree nodes.
            # Or just random and check. Given n=40, we can be a bit smarter.
            
            # Let's try a weighted random selection based on (max_deg - deg[v])
            weights = [max(deg) - d + 1 for d in deg]
            total_w = sum(weights)
            
            # Pick u
            r = random.uniform(0, total_w)
            cumulative = 0
            u = 0
            for i in range(n):
                cumulative += weights[i]
                if r <= cumulative:
                    u = i
                    break
            
            # Pick v (different from u)
            r = random.uniform(0, total_w)
            cumulative = 0
            v = 0
            for i in range(n):
                if i == u:
                    continue
                cumulative += weights[i]
                if r <= cumulative:
                    v = i
                    break
            
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                continue
                
            edge_set.add((u, v))
            deg[u] += 1
            deg[v] += 1
            
        # Fallback if greedy failed to reach m
        while len(edge_set) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                if u > v:
                    u, v = v, u
                edge_set.add((u, v))
                
        return edge_set

    def local_search(edge_set, adj, books, current_score, max_steps=4000, stuck_limit=60):
        stuck = 0
        for step in range(max_steps):
            if time.time() > deadline - 2:
                return books, current_score

            # worst edge
            worst_edge = None
            worst_val = -1
            for e, val in books.items():
                if val > worst_val:
                    worst_val = val
                    worst_edge = e

            if worst_val <= 7:
                break

            u, v = worst_edge
            swapped = False

            # candidate new edges: prefer low-triangle, and (for structure) cross the 20/20 split
            cands = []
            seen = set()
            for _ in range(400):
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                if (ru, rv) in edge_set or (ru, rv) == (u, v):
                    continue
                if (ru, rv) in seen:
                    continue
                seen.add((ru, rv))
                tc = book(ru, rv, adj)
                cands.append((tc, ru, rv))

            cands.sort(key=lambda x: x[0])
            # bias: put cross-partition candidates first among equal triangle counts
            cands.sort(key=lambda x: (x[0], not (x[1] < 20 < x[2] or x[2] < 20 < x[1])))

            for tc, ru, rv in cands[:40]:
                # apply swap
                edge_set.remove((u, v))
                edge_set.add((ru, rv))
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)

                affected = {u, v, ru, rv}
                new_books = {}
                for e, val in books.items():
                    eu, ev = e
                    if eu in affected or ev in affected:
                        new_books[e] = book(eu, ev, adj)
                    else:
                        new_books[e] = val
                new_books[(ru, rv)] = book(ru, rv, adj)

                new_max = max_book(new_books)
                if new_max < current_score:
                    current_score = new_max
                    books = new_books
                    swapped = True
                    stuck = 0
                    break
                else:
                    # revert
                    edge_set.remove((ru, rv))
                    edge_set.add((u, v))
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            if not swapped:
                stuck += 1
            else:
                stuck = 0

            if stuck > stuck_limit:
                break

        return books, current_score

    best_score = 999
    best_edge_set = None

    # Combine initializations: alternate between circulant (symmetric) and near-regular (degree-balanced)
    inits = [circulant_init, near_regular_init]
    
    for restart in range(40):
        if time.time() > deadline - 5:
            break
        init_fn = inits[restart % len(inits)]
        edge_set = init_fn()
        adj = build_bitmasks(edge_set)
        books = all_books(edge_set, adj)
        current_score = max_book(books)

        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)

        if current_score <= 7:
            break

        books, current_score = local_search(edge_set, adj, books, current_score)

        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)

        if current_score <= 7:
            break

    if best_edge_set is None:
        best_edge_set = circulant_init()

    return [[u, v] for (u, v) in best_edge_set]
