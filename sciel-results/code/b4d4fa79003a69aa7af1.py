import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Moore bound for k-regular graph with girth g
    def moore_bound(k, g):
        if k == 2:
            return g
        if g % 2 == 1:
            terms = (g - 1) // 2
            if k == 2:
                return 1 + 2 * terms
            else:
                return 1 + k * (((k-1)**terms - 1) // (k - 2))
        else:
            terms = g // 2 - 1
            if k == 2:
                return 2 * terms
            else:
                return 2 * (((k-1)**terms - 1) // (k - 2))

    n_min = moore_bound(k, g)
    
    # Generate candidates
    # We know for (5,9) n is around 1116.
    # n_min is 426.
    # We will try a range of n values.
    candidates = []
    # Add a dense range from n_min to n_min * 3
    # Step size 10
    end_range = min(n_min * 3, 2000) # Cap at 2000 for speed
    for n in range(n_min, end_range, 10):
        if (n * k) % 2 == 0:
            candidates.append(n)
            
    # Add some larger jumps
    for mult in [3.5, 4.0, 5.0, 6.0, 8.0, 10.0]:
        n_val = int(n_min * mult)
        if (n_val * k) % 2 == 0:
            candidates.append(n_val)
            
    candidates = sorted(list(set(candidates)))

    def generate_random_regular_graph(n, k):
        """
        Generates a simple k-regular graph on n vertices.
        Returns adjacency list or None.
        """
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
            
        for _ in range(50): # Max retries
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            
            edges = set()
            adj = [[] for _ in range(n)]
            valid = True
            
            for i in range(0, len(stubs), 2):
                u = stubs[i]
                v = stubs[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((u, v))
                adj[u].append(v)
                adj[v].append(u)
            
            if valid:
                return adj
        return None

    def find_short_cycle_fast(adj, n, g, sample_size=None):
        """
        Finds a cycle of length < g.
        If sample_size is None, checks all nodes.
        Otherwise, checks a random sample of nodes.
        """
        if sample_size is None:
            sample_size = n
        
        nodes_to_check = random.sample(range(n), min(sample_size, n))
        
        for start in nodes_to_check:
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            cycle = []
                            curr = u
                            while curr != -1:
                                cycle.append(curr)
                                curr = parent[curr]
                            curr = v
                            while curr != u:
                                cycle.append(curr)
                                curr = parent[curr]
                            cycle.append(u)
                            return cycle
        return None

    def local_search(adj, n, k, g, time_limit):
        """
        Attempts to increase girth to g.
        """
        adj_sets = [set(neigh) for neigh in adj]
        
        # Precompute edges list for fast access
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        max_iters = 20000
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                return None
            
            # Use fast check (sample 10% of nodes)
            cycle = find_short_cycle_fast(adj, n, g, sample_size=max(1, n // 10))
            
            if cycle is None:
                # No short cycle found in sample.
                # Do a full check every 100 iterations or if we are near time limit
                if it % 100 == 0 or time.time() - start_time > time_limit - 10:
                    full_cycle = find_short_cycle_fast(adj, n, g)
                    if full_cycle is None:
                        return adj
                    else:
                        cycle = full_cycle
                else:
                    # Assume it's okay for now, continue searching
                    # But to be safe, we should probably do a full check more often
                    # Let's do a full check every 20 iterations
                    if it % 20 == 0:
                         full_cycle = find_short_cycle_fast(adj, n, g)
                         if full_cycle is None:
                             return adj
                         else:
                             cycle = full_cycle
                    else:
                         continue
            else:
                # Found a short cycle, try to break it
                pass
            
            if cycle is None:
                continue

            # Pick a random edge from the cycle
            idx = random.randint(0, len(cycle) - 1)
            u = cycle[idx]
            v = cycle[(idx + 1) % len(cycle)]
            
            # Find candidate edges to swap with
            candidates = []
            for e in edges:
                a, b = e
                if a in (u, v) or b in (u, v):
                    continue
                candidates.append(e)
                
            if not candidates:
                continue
                
            # Try a few random swaps
            for _ in range(5):
                if not candidates:
                    break
                x, y = random.choice(candidates)
                
                # Try swap 1: (u,x), (v,y)
                if x != u and y != v and (u, x) not in adj_sets[u] and (v, y) not in adj_sets[v]:
                    # Apply swap
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)
                    
                    adj_sets[u].discard(v)
                    adj_sets[v].discard(u)
                    adj_sets[x].discard(y)
                    adj_sets[y].discard(x)
                    
                    adj_sets[u].add(x)
                    adj_sets[x].add(u)
                    adj_sets[v].add(y)
                    adj_sets[y].add(v)
                    
                    edges.remove((u, v))
                    edges.remove((x, y))
                    edges.append((u, x) if u < x else (x, u))
                    edges.append((v, y) if v < y else (y, v))
                    break
                # Try swap 2: (u,y), (v,x)
                if y != u and x != v and (u, y) not in adj_sets[u] and (v, x) not in adj_sets[v]:
                    # Apply swap
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(y)
                    adj[y].append(u)
                    adj[v].append(x)
                    adj[x].append(v)
                    
                    adj_sets[u].discard(v)
                    adj_sets[v].discard(u)
                    adj_sets[x].discard(y)
                    adj_sets[y].discard(x)
                    
                    adj_sets[u].add(y)
                    adj_sets[y].add(u)
                    adj_sets[v].add(x)
                    adj_sets[x].add(v)
                    
                    edges.remove((u, v))
                    edges.remove((x, y))
                    edges.append((u, y) if u < y else (y, u))
                    edges.append((v, x) if v < x else (x, v))
                    break

        return None

    # Main Loop
    for n in candidates:
        if time.time() - start_time > time_limit:
            break
            
        # Ensure n*k is even
        if (n * k) % 2 != 0:
            continue
            
        # Try to generate and solve for this n
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            if time.time() - start_time > time_limit:
                break
                
            adj = generate_random_regular_graph(n, k)
            if not adj:
                attempts += 1
                continue
                
            result_adj = local_search(adj, n, k, g, time_limit)
            if result_adj:
                # Final verification
                # Check degree
                valid = True
                for i in range(n):
                    if len(result_adj[i]) != k:
                        valid = False
                        break
                if valid:
                    # Check girth exactly
                    if find_short_cycle_fast(result_adj, n, g) is None:
                        edges = []
                        for u in range(n):
                            for v in result_adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        return n, edges
            attempts += 1

    # Fallback: LUW construction
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
