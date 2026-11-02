"""Construct a 5-regular graph with girth at least 9 using algebraic + local search."""
import time
import random
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) with 0 <= u, v < n.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    # For this specific task: k=5, g=9
    # Use algebraic construction as starting point, then try local search to reduce n
    # But first, let's try to construct a small graph directly.
    
    # Moore bound for (5, 9):
    # For odd girth g=2r+1=9, r=4:
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i = 1 + 5*(1 + 4 + 16 + 64) = 1 + 5*85 = 426
    # So lower bound is 426.
    
    # Strategy:
    # 1. Try to construct a random 5-regular graph on small n (starting from ~450)
    #    and use local search (edge swaps) to eliminate short cycles.
    # 2. If we find a valid graph with girth >= 9, return it.
    # 3. Fall back to algebraic construction if local search fails.
    
    def moore_bound_lower(k, g):
        if g % 2 == 1:  # odd girth
            r = (g - 1) // 2
            n = 1 + k * sum((k - 1)**i for i in range(r))
        else:  # even girth
            r = g // 2
            n = 2 * sum((k - 1)**i for i in range(r))
        return n
    
    lb = moore_bound_lower(k, g)
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar construction - known valid (k=5, g>=10, n=15625)"""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        else:
            add, sub, mul = (lambda x,y: (x+y)%q), (lambda x,y: (x-y)%q), (lambda x,y: x*y%q)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    # Try local search approach for smaller graphs
    def compute_girth(adj, n):
        """Compute girth using BFS from each vertex"""
        min_cycle = float('inf')
        for src in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                if dist[u] * 2 >= min_cycle:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
        return min_cycle
    
    def find_short_cycle(adj, n, target_girth):
        """Find a cycle shorter than target_girth. Returns list of vertices or None."""
        for src in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                u = queue.popleft()
                if dist[u] * 2 >= target_girth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u > v:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            # Reconstruct cycle
                            cycle = [u]
                            while u != src and parent[u] != -1:
                                u = parent[u]
                                cycle.append(u)
                            cycle.reverse()
                            cycle2 = [v]
                            while v != src and parent[v] != -1:
                                v = parent[v]
                                cycle2.append(v)
                            cycle2.reverse()
                            return cycle + cycle2
        return None
    
    def random_k_regular(n, k):
        """Generate random k-regular graph using configuration model with cleanup"""
        if n * k % 2 != 0:
            return None
        for _ in range(50):  # try multiple times
            # Create stubs
            stubs = list(range(n * k))
            random.shuffle(stubs)
            # Pair them up
            edges_set = set()
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * k, 2):
                u = stubs[i] // k
                v = stubs[i+1] // k
                if u == v:
                    valid = False
                    break
                if (min(u,v), max(u,v)) in edges_set:
                    valid = False
                    break
                edges_set.add((min(u,v), max(u,v)))
                adj[u].append(v)
                adj[v].append(u)
            if not valid:
                continue
            # Check degrees
            if all(len(adj[i]) == k for i in range(n)):
                return adj
        return None
    
    def local_search(adj, n, k, target_girth, time_limit=10.0):
        """Try to increase girth via edge swaps"""
        start = time.time()
        girth = compute_girth(adj, n)
        if girth >= target_girth:
            return adj, girth
        
        best_girth = girth
        best_adj = [lst[:] for lst in adj]
        
        for iteration in range(10000):
            if time.time() - start > time_limit:
                break
            
            # Find a short cycle
            cycle = find_short_cycle(adj, n, target_girth)
            if cycle is None:
                girth = compute_girth(adj, n)
                if girth >= target_girth:
                    return adj, girth
                break
            
            # Pick two edges in the cycle to swap
            if len(cycle) < 4:
                continue
            
            # Try random edge swaps
            for attempt in range(20):
                # Pick random edge in cycle
                i = random.randint(0, len(cycle)-1)
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                
                # Pick random other edge
                x = random.randint(0, n-1)
                while not adj[x] or all(adj[x][j] == u or adj[x][j] == v for j in range(len(adj[x]))):
                    x = random.randint(0, n-1)
                y_candidates = [y for y in adj[x] if y not in [u, v] and y != x]
                if not y_candidates:
                    continue
                y = random.choice(y_candidates)
                
                # Check if swap is valid
                if (min(u,y), max(u,y)) in [(min(a,b), max(a,b)) for a,b in zip(adj[x], [0]*len(adj[x]))]:
                    continue
                    
                # Perform swap: remove (u,v) and (x,y), add (u,x) and (v,y)
                # or (u,y) and (v,x)
                for swap_type in [0, 1]:
                    if swap_type == 0:
                        new_edges = [(u,x), (v,y)]
                    else:
                        new_edges = [(u,y), (v,x)]
                    
                    # Check no loops or multi-edges
                    valid = True
                    for a, b in new_edges:
                        if a == b:
                            valid = False
                            break
                        if b in adj[a]:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Check that removing edges doesn't break regularity
                    # (it preserves degree since we remove 2 edges and add 2)
                    
                    # Apply swap
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    adj[u].append(new_edges[0][1])
                    adj[new_edges[0][1]].append(u)
                    adj[v].append(new_edges[1][1])
                    adj[new_edges[1][1]].append(v)
                    
                    # Compute new girth
                    new_girth = compute_girth(adj, n)
                    
                    if new_girth > best_girth:
                        best_girth = new_girth
                        best_adj = [lst[:] for lst in adj]
                    
                    if new_girth >= target_girth:
                        return adj, new_girth
                    
                    if new_girth < girth:
                        # Revert
                        adj[u].remove(new_edges[0][1])
                        adj[new_edges[0][1]].remove(u)
                        adj[v].remove(new_edges[1][1])
                        adj[new_edges[1][1]].remove(v)
                        adj[u].append(v)
                        adj[v].append(u)
                        adj[x].append(y)
                        adj[y].append(x)
                    else:
                        # Accept, update
                        girth = new_girth
                        break
        
        # Restore best if current is worse
        if best_girth > girth:
            return best_adj, best_girth
        return adj, girth
    
    # Try different n values
    best_result = None
    best_n = float('inf')
    
    # First try a few small n values with local search
    for trial_n in [450, 500, 600, 800, 1000, 1200, 1500, 2000, 3000]:
        if time.time() > deadline - 30:
            break
        if trial_n < lb:
            continue
        if trial_n % 2 != 0 and k % 2 != 0:
            continue  # n*k must be even
        
        adj = random_k_regular(trial_n, k)
        if adj is None:
            continue
        
        # Quick girth check
        girth = compute_girth(adj, trial_n)
        if girth >= g:
            edges = []
            for i in range(trial_n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            if best_result is None or trial_n < best_n:
                best_n = trial_n
                best_result = (trial_n, edges)
            return trial_n, edges
        
        # Try local search with limited time
        remaining = deadline - 25 - time.time()
        if remaining < 5:
            break
        time_limit = min(remaining * 0.15, 8.0)
        adj, new_girth = local_search(adj, trial_n, k, g, time_limit)
        
        if new_girth >= g:
            edges = []
            for i in range(trial_n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            if best_result is None or trial_n < best_n:
                best_n = trial_n
                best_result = (trial_n, edges)
            return trial_n, edges
    
    # Fall back to algebraic construction
    n, edges = luw_construct(k, g)
    return n, edges

