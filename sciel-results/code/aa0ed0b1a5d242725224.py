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

    def bipartite_clique_init():
        """
        Start with K_{20,20} (400 edges) + 1 internal edge to reach 401.
        This is the extremal structure from the Edwards-Khadzhiivanov-Nikiforov theorem.
        The single internal edge creates a small book; we then use local search to
        redistribute edges to minimize the maximum book.
        """
        edge_set = set()
        # Add all bipartite edges between {0..19} and {20..39}
        for i in range(20):
            for j in range(20, 40):
                edge_set.add((i, j))
        # Add 1 internal edge to reach 401
        # Choose an internal edge that minimizes initial book size
        # An internal edge within one part: its book = common neighbors in the other part
        # For edge (u,v) with u,v in [0,19]: book = |N(u) ∩ N(v)| = 20 (all of part 2)
        # So any internal edge has book 20 initially. Let's pick one.
        edge_set.add((0, 1))
        return edge_set

    def near_regular_init():
        """
        Initialize a near-regular graph using greedy degree balancing.
        """
        edge_set = set()
        deg = [0] * n
        
        for i in range(n):
            j = (i + 1) % n
            if i < j:
                edge_set.add((i, j))
                deg[i] += 1
                deg[j] += 1
        
        remaining = m - len(edge_set)
        attempts = 0
        max_attempts = 100000
        
        while len(edge_set) < m and attempts < max_attempts:
            attempts += 1
            weights = [max(deg) - d + 1 for d in deg]
            total_w = sum(weights)
            
            r = random.uniform(0, total_w)
            cumulative = 0
            u = 0
            for i in range(n):
                cumulative += weights[i]
                if r <= cumulative:
                    u = i
                    break
            
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
            
        while len(edge_set) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                if u > v:
                    u, v = v, u
                edge_set.add((u, v))
                
        return edge_set

    def circulant_init():
        """Highly symmetric, regular, non-bipartite start."""
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
        while len(edge_set) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u < v:
                edge_set.add((u, v))
        return edge_set

    def local_search_pair_swap(edge_set, adj, books, current_score, max_steps=2000, stuck_limit=40):
        """
        Pair-swap local search: remove two edges and add two different edges.
        This allows more global restructuring than single-edge swaps.
        """
        stuck = 0
        for step in range(max_steps):
            if time.time() > deadline - 2:
                return books, current_score

            # Find the worst edge
            worst_edge = None
            worst_val = -1
            for e, val in books.items():
                if val > worst_val:
                    worst_val = val
                    worst_edge = e

            if worst_val <= 7:
                break

            e1 = worst_edge
            u1, v1 = e1

            # Find a second edge to swap with (prefer another high-book edge)
            # Sample some edges and pick one with high book
            edges_list = list(books.keys())
            random.shuffle(edges_list)
            
            found_swap = False
            for _ in range(50):  # try up to 50 candidate second edges
                if time.time() > deadline - 2:
                    return books, current_score
                
                e2 = random.choice([e for e in edges_list if e != e1])
                u2, v2 = e2
                
                # Generate candidate replacement edges
                cands = []
                seen = set()
                for _ in range(200):
                    ru = random.randint(0, n - 1)
                    rv = random.randint(0, n - 1)
                    if ru == rv:
                        continue
                    if ru > rv:
                        ru, rv = rv, ru
                    if (ru, rv) in edge_set or (ru, rv) == e1 or (ru, rv) == e2:
                        continue
                    if (ru, rv) in seen:
                        continue
                    seen.add((ru, rv))
                    tc = book(ru, rv, adj)
                    cands.append((tc, ru, rv))

                if len(cands) < 2:
                    continue
                
                cands.sort(key=lambda x: x[0])
                
                # Pick the two best candidates
                for c1_idx in range(min(10, len(cands))):
                    for c2_idx in range(c1_idx + 1, min(10, len(cands))):
                        tc_a, ra, rb = cands[c1_idx]
                        tc_b, rc, rd = cands[c2_idx]
                        
                        # Check no conflicts
                        new_edges = [(ra, rb), (rc, rd)]
                        if len(set(new_edges)) != 2:
                            continue
                        
                        # Apply swap
                        edge_set.remove(e1)
                        edge_set.remove(e2)
                        edge_set.add((ra, rb))
                        edge_set.add((rc, rd))
                        
                        adj[u1] &= ~(1 << v1)
                        adj[v1] &= ~(1 << u1)
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                        adj[ra] |= (1 << rb)
                        adj[rb] |= (1 << ra)
                        adj[rc] |= (1 << rd)
                        adj[rd] |= (1 << rc)

                        # Recompute books for affected edges
                        affected_vertices = {u1, v1, u2, v2, ra, rb, rc, rd}
                        new_books = {}
                        for e, val in books.items():
                            eu, ev = e
                            if eu in affected_vertices or ev in affected_vertices:
                                new_books[e] = book(eu, ev, adj)
                            else:
                                new_books[e] = val
                        new_books[(ra, rb)] = book(ra, rb, adj)
                        new_books[(rc, rd)] = book(rc, rd, adj)

                        new_max = max_book(new_books)
                        if new_max < current_score:
                            current_score = new_max
                            books = new_books
                            found_swap = True
                            stuck = 0
                            break
                        else:
                            # Revert
                            edge_set.remove((ra, rb))
                            edge_set.remove((rc, rd))
                            edge_set.add(e1)
                            edge_set.add(e2)
                            adj[ra] &= ~(1 << rb)
                            adj[rb] &= ~(1 << ra)
                            adj[rc] &= ~(1 << rd)
                            adj[rd] &= ~(1 << rc)
                            adj[u1] |= (1 << v1)
                            adj[v1] |= (1 << u1)
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                    
                    if found_swap:
                        break
                
                if found_swap:
                    break

            if not found_swap:
                stuck += 1
            else:
                stuck = 0

            if stuck > stuck_limit:
                break

        return books, current_score

    def local_search_single_swap(edge_set, adj, books, current_score, max_steps=3000, stuck_limit=50):
        """
        Single-edge swap local search (backup method).
        """
        stuck = 0
        for step in range(max_steps):
            if time.time() > deadline - 2:
                return books, current_score

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

            cands = []
            seen = set()
            for _ in range(300):
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

            for tc, ru, rv in cands[:30]:
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

    # Use bipartite_clique_init as primary, with fallbacks
    inits = [bipartite_clique_init, near_regular_init, circulant_init]
    
    for restart in range(30):
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

        # Try pair-swap first
        books, current_score = local_search_pair_swap(edge_set, adj, books, current_score)

        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)

        if current_score <= 7:
            break

        # If still high, try single-swap
        if current_score > 8:
            books, current_score = local_search_single_swap(edge_set, adj, books, current_score)

            if current_score < best_score:
                best_score = current_score
                best_edge_set = set(edge_set)

            if current_score <= 7:
                break

    if best_edge_set is None:
        best_edge_set = bipartite_clique_init()

    return [[u, v] for (u, v) in best_edge_set]
