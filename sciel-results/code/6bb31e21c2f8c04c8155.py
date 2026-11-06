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
        # Check min degree
        for i in range(n):
            if adj[i].bit_count() < min_deg:
                return False
        # Check C4: any two vertices share at most 1 common neighbor
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common = ai & adj[j]
                # common.bit_count() > 1  <=>  (common & (common - 1)) != 0
                if common & (common - 1):
                    return False
        return True

    def greedy_build(n, target_deg):
        """
        Greedily construct a C4-free graph on n vertices aiming for min degree >= target_deg.
        Returns adj list of bitmasks or None if failed.
        """
        adj = [0] * n
        deg = [0] * n
        
        # We want to add edges such that we don't create C4.
        # A C4 is created if we add edge (u,v) and there exists w such that 
        # u-w-v is a path (i.e., w is common neighbor of u and v).
        # So, we can only add (u,v) if common_neighbors(u,v) == 0.
        
        # To maximize min degree, we should prioritize connecting low-degree vertices.
        # However, we must maintain the property.
        
        # Randomized greedy:
        # 1. Create a list of all potential edges.
        # 2. Shuffle them? No, that's slow.
        # 3. Iterate vertices, try to connect to others.
        
        # Better approach: 
        # Maintain a list of "available" edges.
        # Start with empty graph.
        # While there are low-degree vertices:
        #   Pick a vertex u with low degree.
        #   Find all v such that (u,v) not in graph and common_neighbors(u,v)==0.
        #   Add one such edge.
        
        # To make it fast, we can just iterate u from 0 to n-1 and try to add edges to v > u.
        # But this might leave some vertices isolated if we're not careful.
        
        # Let's try a multi-pass approach.
        # Pass 1: Connect each vertex to as many as possible.
        
        # We'll use a randomized order to break symmetries.
        order = list(range(n))
        random.shuffle(order)
        
        # We want to add edges.
        # For each u in order:
        #   Find candidates v (v != u) such that:
        #     1. Not adjacent
        #     2. No common neighbor
        #   If found, add edge.
        
        # This is O(n^3) if done naively.
        # For n=47, n^3 ~ 100k. This is fast.
        
        # We want to ensure min degree is high.
        # So, we should prioritize adding edges to vertices with currently lowest degree.
        
        # Let's simulate:
        # 1. Initialize adj, deg.
        # 2. Loop while we can add edges:
        #    Find u with min degree.
        #    Find v such that (u,v) can be added.
        #    Add (u,v).
        #    If no such v for min-degree u, break? Or pick next min?
        
        # To avoid getting stuck, we can allow some randomness.
        
        # Let's try a simpler, robust heuristic:
        # Iterate u from 0 to n-1.
        # For each u, try to connect to v (v > u) if possible.
        # To boost min degree, we might need to go back and connect earlier vertices to later ones?
        # Actually, if we process u in order, and try to connect to ALL possible v, we get a maximal C4-free graph.
        # Maximal C4-free graphs often have high min degree.
        
        # Let's try:
        for u in range(n):
            # Candidates for u
            # We want to connect u to v such that common_neighbors(u,v) == 0.
            # Note: if we add (u,v), it might block future edges for other vertices.
            # But we just want A valid graph with high min degree.
            
            # To maximize min degree, we should connect u to vertices that are currently "starving".
            # However, checking "starving" dynamically is complex.
            
            # Let's just try to add as many edges as possible for u.
            # Sort candidates by degree of v? Connect to low degree v first?
            
            candidates = []
            for v in range(n):
                if u == v:
                    continue
                if adj[u] & (1 << v):
                    continue
                # Check common neighbors
                # common = adj[u] & adj[v]
                # if common.bit_count() == 0:
                #    candidates.append(v)
                
                # Optimization: if adj[u] is dense, common is likely non-zero.
                # But n is small.
                
                common = adj[u] & adj[v]
                if common == 0:
                    candidates.append(v)
            
            # Sort candidates by degree (ascending) to help low-degree vertices?
            # Or random?
            # Let's sort by degree to balance.
            candidates.sort(key=lambda x: deg[x])
            
            # Add all?
            # If we add all, we might create C4s with future vertices?
            # No, C4 is local to the pair (u,v) and their common neighbors.
            # Adding (u,v) only creates C4 if there was a common neighbor.
            # So adding all valid candidates is safe for C4-freeness *at this step*.
            # However, it might reduce the options for other vertices.
            
            # To maximize min degree, adding all is good.
            for v in candidates:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
        
        # Check if min degree is sufficient
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
            """Return True if edge (u,v) can be added; if so, add it."""
            if u == v:
                return False
            if adj[u] & (1 << v):
                return False
            common = adj[u] & adj[v]
            if common & (common - 1):  # > 1 common neighbor -> C4
                return False
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            return True

        def get_low_vertices(threshold):
            """Get vertices with degree below threshold."""
            return [i for i in range(n) if deg[i] < threshold]

        def try_add_edge_to_low(u, max_trials=10):
            """Try to add an edge to low-degree vertex u."""
            for _ in range(max_trials):
                v = random.randrange(n)
                if v == u:
                    continue
                if adj[u] & (1 << v):
                    continue
                if try_add_edge(u, v):
                    return True
            return False

        def try_swap_improve():
            """Try an edge swap to improve min degree."""
            # Find low-degree vertices
            low_verts = get_low_vertices(target_deg)
            if not low_verts:
                low_verts = get_low_vertices(best_min_deg + 2)
            if not low_verts:
                return False

            c = random.choice(low_verts)
            
            # Find a candidate edge (a,b) to remove
            # Prefer edges where both endpoints have high degree
            high_deg_edges = []
            for a in range(n):
                if deg[a] <= target_deg:
                    continue
                mask = adj[a]
                if mask == 0:
                    continue
                # Get a neighbor b with high degree
                while mask:
                    b = (mask & -mask).bit_length() - 1
                    if deg[b] > target_deg:
                        high_deg_edges.append((a, b))
                        break
                    mask &= mask - 1
            
            if not high_deg_edges:
                # Try any edge
                for a in range(n):
                    if deg[a] <= 2:
                        continue
                    mask = adj[a]
                    if mask == 0:
                        continue
                    b = (mask & -mask).bit_length() - 1
                    high_deg_edges.append((a, b))
                    break
            
            if not high_deg_edges:
                return False

            for _ in range(15):
                if not high_deg_edges:
                    break
                a, b = random.choice(high_deg_edges)
                
                # Temporarily remove (a,b)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                deg[a] -= 1
                deg[b] -= 1

                # Try to add an edge to low-degree vertex c
                added_edge = False
                for _ in range(8):
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
                        return True
                    else:
                        # Revert
                        adj[c] &= ~(1 << d)
                        adj[d] &= ~(1 << c)
                        deg[c] -= 1
                        deg[d] -= 1
                # Re-add (a,b)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                deg[a] += 1
                deg[b] += 1

            return False

        while time.time() - start < max_time:
            if no_improve > 1500:
                break

            current_min = min(deg)
            if current_min >= target_deg:
                if check_valid(adj, n, target_deg):
                    return adj, best_min_deg
                break

            # Pick the lowest-degree vertex
            u = min(range(n), key=lambda i: deg[i])

            # 1) Try to add an edge incident to u
            added = try_add_edge_to_low(u, max_trials=12)

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
            swapped = try_swap_improve()

            if swapped:
                new_min = min(deg)
                if new_min > best_min_deg:
                    best_min_deg = new_min
                    best_adj = adj[:]
                    no_improve = 0
                else:
                    no_improve += 1
                continue

            no_improve += 1

        return best_adj, best_min_deg

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        found = False
        
        # Strategy 1: Greedy Build with more attempts
        for attempt in range(300):
            if time.time() > deadline:
                break

            random.seed(attempt * 123 + 456)
            adj = greedy_build(n, min_deg)

            if adj is not None:
                if check_valid(adj, n, min_deg):
                    edges = get_edges(adj, n)
                    return {'n': n, 'edges': edges}

                # If greedy build failed min_deg, try local search to fix it
                remaining = deadline - time.time()
                ls_time = min(10.0, remaining * 0.3)
                if ls_time < 0.5:
                    break
                adj_ls, min_deg_ls = local_search(adj, n, min_deg, max_time=ls_time)
                if min_deg_ls >= min_deg and check_valid(adj_ls, n, min_deg):
                    edges = get_edges(adj_ls, n)
                    return {'n': n, 'edges': edges}
        
        # Strategy 2: Structured initialization based on polarity graph / incidence graph
        # Use a near-complete graph minus a matching or sparse structure
        for attempt in range(150):
            if time.time() > deadline:
                break

            random.seed(attempt * 7919 + 13)
            adj = [0] * n

            # Start with a random permutation-based structure
            perm = list(range(n))
            random.shuffle(perm)
            
            # Create a "block" structure: divide vertices into groups
            # and connect within groups sparsely, between groups densely
            # This mimics known C4-free constructions
            
            # For now, use a randomized dense initialization
            p = min(min_deg / n * 1.5, 0.8)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # Repair phase with improved C4 resolution
            max_iter = 200
            for _ in range(max_iter):
                if time.time() > deadline:
                    break

                # Check for C4s
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

                    # Remove edges to reduce common neighbors to 1
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
                    # No C4s, check min degree
                    min_deg_actual = min(adj[i].bit_count() for i in range(n))
                    if min_deg_actual >= min_deg:
                        if check_valid(adj, n, min_deg):
                            edges = get_edges(adj, n)
                            return {'n': n, 'edges': edges}
                        break
                    else:
                        # Need to add edges - use local search
                        remaining = deadline - time.time()
                        ls_time = min(8.0, remaining * 0.2)
                        if ls_time < 0.3:
                            break
                        adj_ls, min_deg_ls = local_search(adj, n, min_deg, max_time=ls_time)
                        if min_deg_ls >= min_deg and check_valid(adj_ls, n, min_deg):
                            edges = get_edges(adj_ls, n)
                            return {'n': n, 'edges': edges}

    # Fallback: Return a valid small graph
    # Use a cycle which is C4-free (for n != 4)
    n = 40
    min_deg = 2  # Cycle has degree 2
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
        adj[i] |= (1 << ((i - 1) % n))
        adj[(i - 1) % n] |= (1 << i)
    
    if check_valid(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
