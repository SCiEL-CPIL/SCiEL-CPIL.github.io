import time
import random
import numpy as np

def construct(k, g):
    deadline = time.time() + 115  # Leave 5s margin
    
    # 1. Get baseline from LUW
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        n_luw, edges_luw = luw_result
        if _verify_graph(edges_luw, n_luw, k, g):
            best_n = n_luw
            best_edges = edges_luw
        else:
            best_n = float('inf')
            best_edges = None
    else:
        best_n = float('inf')
        best_edges = None

    # 2. Try to find a smaller graph using local search on bipartite graphs
    # The (k,g)-cage for k=4, g=10 is bipartite.
    # We search for n starting from Moore bound up to best_n - 1.
    # Since we know 320 is possible, we prioritize small n.
    
    if k % 2 != 0:
        if best_edges is not None:
            return best_n, best_edges
        return _luw_construct(k, g)
        
    # Moore bound
    moore_bound = _moore_bound(k, g)
    
    # Search range
    # We know 320 is a candidate.
    # Let's try n from moore_bound to min(best_n - 1, 400)
    # But to save time, if we suspect 320, we should try 320 first if it's >= moore_bound.
    # Actually, let's just iterate.
    
    # Optimization: If best_n is large (e.g. 2000), we should definitely try small n first.
    # The local search is more likely to succeed on smaller n if a solution exists.
    
    # We will try n values in increasing order.
    # But checking every n is expensive.
    # Let's try a few specific candidates: 320, 322, ... up to best_n.
    # Or just a range.
    
    # To be safe and fast, let's try n = 320 first if it's >= moore_bound.
    # If not found, try 322, 324...
    
    candidates = []
    start_n = max(moore_bound, 320) # We know 320 is the known best
    if start_n < best_n:
        # Try 320, 322, 324 ... up to best_n - 1
        # But don't try too many.
        for n in range(start_n, min(best_n, start_n + 100), 2):
            candidates.append(n)
            
    # Also try some random n in between if time permits?
    # For now, just the sequential small n.
    
    for n in candidates:
        if time.time() > deadline - 5:
            break
        result = _local_search_bipartite(k, g, n, deadline)
        if result is not None:
            n_found, edges_found = result
            if n_found < best_n:
                best_n = n_found
                best_edges = edges_found
                # If we found 320, we can stop? 
                # The problem says "smallest known... 320". 
                # If we find 320, it's likely optimal.
                # But to be sure, we could keep searching for smaller?
                # Moore bound is 243. So 244..318 are possible.
                # But local search on smaller n is harder (tighter constraints).
                # If we find 320, it's a huge improvement over LUW.
                # Let's stop if we find 320 to ensure we return a valid good result quickly.
                if n_found == 320:
                    return best_n, best_edges

    if best_edges is not None:
        return best_n, best_edges
        
    return _luw_construct(k, g)

def _moore_bound(k, g):
    if g % 2 == 0:
        m = g // 2
        s = 0
        for i in range(m):
            s += (k - 1) ** i
        return 2 * s
    else:
        m = (g - 1) // 2
        s = 0
        for i in range(m):
            s += (k - 1) ** i
        return 1 + k * s

def _verify_graph(edges, n, k, g):
    if n <= 0 or len(edges) == 0:
        return False
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        adj[u].append(v)
        adj[v].append(u)
    
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != k:
            return False
            
    # Girth check
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        max_depth = (g - 2) // 2
        
        for d in range(1, max_depth + 1):
            new_queue = []
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = d
                        parent[v] = u
                        new_queue.append(v)
                    else:
                        if dist[v] < d - 1:
                            cycle_len = d + dist[v] + 1
                            if cycle_len < g:
                                return False
                        elif dist[v] == d - 1 and parent[u] != v:
                            cycle_len = 2 * d
                            if cycle_len < g:
                                return False
            queue = new_queue
    return True

def _luw_construct(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _local_search_bipartite(k, g, n, deadline):
    M = n // 2
    # Adjacency: left[i] is a set of right neighbors
    left_adj = [set() for _ in range(M)]
    right_adj = [set() for _ in range(M)]
    
    def init_random():
        nonlocal left_adj, right_adj
        left_adj = [set() for _ in range(M)]
        right_adj = [set() for _ in range(M)]
        # Configuration model
        left_stubs = [(i, s) for i in range(M) for s in range(k)]
        right_stubs = [(i, s) for i in range(M) for s in range(k)]
        random.shuffle(right_stubs)
        
        for (li, ls), (ri, rs) in zip(left_stubs, right_stubs):
            if ri not in left_adj[li]:
                left_adj[li].add(ri)
                right_adj[ri].add(li)
            else:
                # Multi-edge, need to fix or retry init
                return False
        return True
        
    if not init_random():
        return None
        
    # Count short cycles
    def count_short_cycles():
        # Count 4-cycles
        c4 = 0
        for l in range(M):
            # For each pair of neighbors of l, count common neighbors in right?
            # No, 4-cycle is l1-r1-l2-r2-l1.
            # l1 and l2 share r1 and r2.
            # So for each pair of left nodes (l1, l2), count common right neighbors.
            # Sum over all pairs.
            pass # Too slow for Python loops if M=160? 160^2 * 160 = 4M ops. OK.
            
        # Better: 
        # 4-cycles: 
        # For each right node r, let d_r be its degree.
        # Number of 4-cycles through r is C(d_r, 2) * (number of other right nodes sharing neighbors?)
        # Actually, just sum over all pairs of left nodes (l1, l2) the number of common neighbors squared?
        # No. 
        # Let's just use BFS for girth check, it's simpler and M is small.
        
        # But for local search, we need a fast metric.
        # Let's just check if girth >= g.
        
        # Build full adj for BFS
        adj = [None] * n
        for i in range(M):
            adj[i] = list(left_adj[i])
        for i in range(M):
            adj[M+i] = list(right_adj[i])
            
        # Check girth
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            max_depth = (g - 2) // 2
            found_short = False
            for d in range(1, max_depth + 1):
                new_queue = []
                for u in queue:
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = d
                            parent[v] = u
                            new_queue.append(v)
                        else:
                            if dist[v] < d - 1:
                                cycle_len = d + dist[v] + 1
                                if cycle_len < g:
                                    return False, cycle_len, (start, u, v)
                            elif dist[v] == d - 1 and parent[u] != v:
                                cycle_len = 2 * d
                                if cycle_len < g:
                                    return False, cycle_len, (start, u, v)
                queue = new_queue
        return True, g, None

    # Initial check
    is_valid, girth, cycle_info = count_short_cycles()
    if is_valid:
        edges = []
        for l in range(M):
            for r in left_adj[l]:
                edges.append((l, M+r))
        return n, edges

    # Local search
    # We need to find a cycle to break.
    # The BFS above returns one short cycle.
    # We can try to swap edges to break it.
    
    for _ in range(10000):
        if time.time() > deadline:
            return None
            
        is_valid, girth, cycle_info = count_short_cycles()
        if is_valid:
            edges = []
            for l in range(M):
                for r in left_adj[l]:
                    edges.append((l, M+r))
            return n, edges
            
        # Get a short cycle
        # cycle_info is (start, u, v) where u-v is a back edge causing cycle.
        # This is not enough to identify the full cycle for swapping easily.
        # Let's just pick a random edge and try to swap it with another random edge.
        
        # Pick random edge (l, r)
        l = random.randint(0, M-1)
        if not left_adj[l]:
            continue
        r = random.choice(list(left_adj[l]))
        
        # Pick another random edge (l2, r2)
        l2 = random.randint(0, M-1)
        if not left_adj[l2]:
            continue
        r2 = random.choice(list(left_adj[l2]))
        
        # Swap (l, r) and (l2, r2) to (l, r2) and (l2, r)
        # Check if (l, r2) and (l2, r) are valid (not existing)
        if r2 in left_adj[l] or r in left_adj[l2]:
            continue
        if l in right_adj[r2] or l2 in right_adj[r]:
            continue
            
        # Perform swap
        left_adj[l].remove(r)
        right_adj[r].remove(l)
        left_adj[l2].remove(r2)
        right_adj[r2].remove(l2)
        
        left_adj[l].add(r2)
        right_adj[r2].add(l)
        left_adj[l2].add(r)
        right_adj[r].add(l2)
        
        # Check if this improved things
        # Simple acceptance: always accept? Or check girth?
        # Checking girth every step is expensive.
        # Let's accept and check every 10 steps.
        
        if _ % 10 == 0:
            is_valid, girth, _ = count_short_cycles()
            if is_valid:
                edges = []
                for l in range(M):
                    for r in left_adj[l]:
                        edges.append((l, M+r))
                return n, edges

    return None
