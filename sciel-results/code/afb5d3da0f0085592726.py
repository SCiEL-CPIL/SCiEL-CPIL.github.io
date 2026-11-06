import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        """Extract edge list from bitmask adjacency."""
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def check_valid(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg."""
        for i in range(n):
            if adj[i].bit_count() < min_deg:
                return False
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common = ai & adj[j]
                if common & (common - 1):
                    return False
        return True

    def polarity_graph(n):
        """
        Construct a polarity graph over GF(q) where n = q+1.
        Vertices are points of PG(1,q) plus a special point.
        Edge (i,j) iff the bilinear form B(i,j) = 0 (orthogonal).
        This gives a C4-free graph.
        """
        # For small n, use known polarity constructions
        # Actually, let's use a simpler approach: finite field construction
        # For n = q+1, use GF(q)
        
        # Find prime power q such that q+1 = n
        q = n - 1
        # Check if q is a prime power
        def is_prime_power(m):
            if m < 2:
                return False
            for p in range(2, m + 1):
                if m % p == 0:
                    # Check if m = p^k for some k
                    temp = m
                    while temp % p == 0:
                        temp //= p
                    if temp == 1:
                        return True
            return False
        
        if not is_prime_power(q):
            return None
        
        # For simplicity, only handle prime q (not prime powers)
        def is_prime(m):
            if m < 2:
                return False
            for i in range(2, int(m**0.5) + 1):
                if m % i == 0:
                    return False
            return True
        
        if not is_prime(q):
            return None
        
        # Build polarity graph over GF(q)
        # Vertices: 0, 1, ..., q-1 (elements of GF(q))
        # Plus vertex q (the "infinity" point)
        # Edge (i,j) for i,j in GF(q): i*j = 1 (or some non-degenerate condition)
        # Actually, standard construction:
        # Points of PG(1,q) = GF(q) ∪ {∞}
        # Polarity: (x, y) -> x*y + x + y = 0 (or similar)
        
        # Let me use a known C4-free construction:
        # For prime p, the graph with vertices Z_p, edge (i,j) iff i-j is a quadratic residue
        # This is the Paley graph, but it may contain C4s.
        
        # Better: use the construction from the literature on R(C4, K_{1,t})
        # For t=39, we need n up to 46 or 47.
        
        # Let me try a different approach: start with a known good structure
        # and then optimize.
        
        return None

    def greedy_build(n, target_deg):
        """
        Greedily construct a C4-free graph on n vertices aiming for min degree >= target_deg.
        Returns adj list of bitmasks or None if failed.
        """
        adj = [0] * n
        deg = [0] * n
        
        order = list(range(n))
        random.shuffle(order)
        
        for u in order:
            candidates = []
            for v in range(n):
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            # Sort by degree ascending to help low-degree vertices
            candidates.sort(key=lambda x: deg[x])
            
            for v in candidates:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
        
        if min(deg) >= target_deg:
            return adj
        return None

    def local_search(adj, n, target_deg, max_time=10.0):
        """
        Improve min degree using edge additions and edge swaps.
        Returns (adj, min_deg).
        """
        deg = [0] * n
        for i in range(n):
            deg[i] = adj[i].bit_count()

        best_min_deg = min(deg)
        best_adj = adj[:]

        start = time.time()
        no_improve = 0

        def try_add_edge(u, v):
            if u == v:
                return False
            if adj[u] & (1 << v):
                return False
            common = adj[u] & adj[v]
            if common & (common - 1):
                return False
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            return True

        while time.time() - start < max_time:
            if no_improve > 1000:
                break

            current_min = min(deg)
            if current_min >= target_deg:
                if check_valid(adj, n, target_deg):
                    return adj, best_min_deg
                break

            # Pick a low-degree vertex
            u = min(range(n), key=lambda i: deg[i])

            # 1) Try to add an edge incident to u
            added = False
            for _ in range(10):
                v = random.randrange(n)
                if v == u:
                    continue
                if adj[u] & (1 << v):
                    continue
                if try_add_edge(u, v):
                    added = True
                    break

            if added:
                new_min = min(deg)
                if new_min > best_min_deg:
                    best_min_deg = new_min
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
                continue

            # 2) Try an edge swap
            swapped = False
            for _ in range(30):
                a = random.randrange(n)
                mask = adj[a]
                if mask == 0:
                    continue
                b = (mask & -mask).bit_length() - 1
                if deg[a] <= target_deg or deg[b] <= target_deg:
                    continue
                
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                deg[a] -= 1
                deg[b] -= 1

                low_verts = [i for i in range(n) if deg[i] < target_deg]
                if not low_verts:
                    low_verts = [i for i in range(n) if deg[i] < best_min_deg + 2]
                if not low_verts:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    deg[a] += 1
                    deg[b] += 1
                    continue

                c = random.choice(low_verts)
                added_edge = False
                for _ in range(10):
                    d = random.randrange(n)
                    if d == c:
                        continue
                    if adj[c] & (1 << d):
                        continue
                    common = adj[c] & adj[d]
                    if common & (common - 1):
                        continue
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    deg[c] += 1
                    deg[d] += 1
                    added_edge = True
                    break

                if added_edge:
                    new_min = min(deg)
                    if new_min > best_min_deg:
                        best_min_deg = new_min
                        best_adj = adj[:]
                        no_improve = 0
                        swapped = True
                    else:
                        adj[c] &= ~(1 << d)
                        adj[d] &= ~(1 << c)
                        deg[c] -= 1
                        deg[d] -= 1
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        deg[a] += 1
                        deg[b] += 1
                        no_improve += 1
                else:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    deg[a] += 1
                    deg[b] += 1
                    no_improve += 1

            if not swapped and not added:
                no_improve += 1

        return best_adj, best_min_deg

    def structured_build(n, target_deg):
        """
        Build a structured C4-free graph using a block design approach.
        Partition vertices into groups and use a regular pattern.
        """
        adj = [0] * n
        deg = [0] * n
        
        # For n=46, target_deg=7. For n=47, target_deg=8.
        # Use a cyclic construction or similar.
        
        # Try a circulant-like graph
        # For each vertex i, connect to i+1, i+2, ..., i+k mod n for some k
        # But this may create C4s.
        
        # Better: use a random regular-ish graph and repair
        return None

    best_result = None
    best_n = 0

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        found = False
        
        # Strategy 1: Multiple greedy builds with different seeds
        for attempt in range(300):
            if time.time() > deadline:
                break

            random.seed(attempt * 123 + 456 + n)
            adj = greedy_build(n, min_deg)

            if adj is not None:
                if check_valid(adj, n, min_deg):
                    edges = get_edges(adj, n)
                    if n > best_n:
                        best_n = n
                        best_result = {'n': n, 'edges': edges}
                    found = True
                    break

                # Try local search
                remaining = deadline - time.time()
                ls_time = min(5.0, remaining * 0.2)
                if ls_time < 0.5:
                    break
                adj_ls, min_deg_ls = local_search(adj, n, min_deg, max_time=ls_time)
                if min_deg_ls >= min_deg and check_valid(adj_ls, n, min_deg):
                    edges = get_edges(adj_ls, n)
                    if n > best_n:
                        best_n = n
                        best_result = {'n': n, 'edges': edges}
                    found = True
                    break
        
        if found:
            continue
        
        # Strategy 2: Random Initialization + Repair
        for attempt in range(150):
            if time.time() > deadline:
                break

            random.seed(attempt * 7919 + 13 + n)

            p = min(min_deg / n * 1.5, 0.6)
            adj = [0] * n

            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # Repair phase
            max_iter = 200
            for _ in range(max_iter):
                if time.time() > deadline:
                    break

                violating_pair = None
                for i in range(n):
                    ai = adj[i]
                    for j in range(i + 1, n):
                        common = ai & adj[j]
                        if common & (common - 1):
                            violating_pair = (i, j)
                            break
                    if violating_pair:
                        break

                if violating_pair:
                    i, j = violating_pair
                    common = adj[i] & adj[j]
                    common_list = []
                    mask = common
                    k = 0
                    while mask:
                        if mask & 1:
                            common_list.append(k)
                        mask >>= 1
                        k += 1

                    deg_i = adj[i].bit_count()
                    deg_j = adj[j].bit_count()

                    for w in common_list[1:]:
                        if deg_i > deg_j:
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                            deg_i -= 1
                        elif deg_j > deg_i:
                            adj[j] &= ~(1 << w)
                            adj[w] &= ~(1 << j)
                            deg_j -= 1
                        else:
                            if random.random() < 0.5:
                                adj[i] &= ~(1 << w)
                                adj[w] &= ~(1 << i)
                                deg_i -= 1
                            else:
                                adj[j] &= ~(1 << w)
                                adj[w] &= ~(1 << j)
                                deg_j -= 1
                else:
                    min_deg_actual = min(adj[i].bit_count() for i in range(n))
                    if min_deg_actual >= min_deg:
                        if check_valid(adj, n, min_deg):
                            edges = get_edges(adj, n)
                            if n > best_n:
                                best_n = n
                                best_result = {'n': n, 'edges': edges}
                            found = True
                            break
                        break
                    else:
                        low_verts = [i for i in range(n) if adj[i].bit_count() < min_deg]
                        if not low_verts:
                            break
                        u = random.choice(low_verts)
                        candidates = []
                        for v in range(n):
                            if v == u:
                                continue
                            if not (adj[u] & (1 << v)):
                                common = adj[u] & adj[v]
                                if common == 0:
                                    candidates.append(v)

                        if candidates:
                            v = random.choice(candidates)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                        else:
                            break

        if found:
            continue

    # Fallback
    if best_result is not None:
        return best_result

    n = 40
    min_deg = 1
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    if check_valid(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
