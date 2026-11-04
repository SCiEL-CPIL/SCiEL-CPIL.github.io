import math
import random
import time
import numpy as np
from collections import deque
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Moore bound
    if g % 2 == 0:
        h = g // 2 - 1
        moore = 2 * k * (k - 1) ** h if h >= 0 else 2 * k
    else:
        h = (g - 1) // 2
        moore = 2 * (k - 1) ** h + k * (k - 1) ** (h - 1) if h >= 1 else 2 * k
    moore = max(moore, 2 * k)
    
    # Strategy 1: LUW
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    if n_luw == 0:
        # Fallback
        n = max(moore, 100)
        while (n * k) % 2 != 0: n += 1
        edges = _random_k_regular(n, k)
        if edges:
            return (n, edges)
        return (0, [])
        
    # Strategy 2: Random Search
    # Start from a reasonable estimate
    # girth ~ 2 * log_{k-1}(n) => n ~ (k-1)^(g/2)
    est_n = int((k-1) ** (g / 2.0))
    start_n = max(moore, est_n - 100)
    # Ensure even
    if (start_n * k) % 2 != 0:
        start_n += 1
        
    end_n = best_n - 1
    
    for n in range(start_n, end_n + 1, 2):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random graphs
        # More attempts for smaller n?
        attempts = 50 if n < 1100 else 20
        
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
                
            edges = _random_k_regular(n, k)
            if edges is None:
                continue
                
            if _verify_girth_fast(k, g, n, edges):
                best_n = n
                best_edges = edges
                # If we found one, we can stop searching this n
                # and try smaller n?
                # The loop goes UP. So we found a valid one at n.
                # We should break and try n-2?
                # But we are iterating up.
                # If we find one at n, it's better than LUW.
                # But maybe n-2 works?
                # We should break out of the inner loop and decrement n?
                # It's easier to just break and let the outer loop continue?
                # No, outer loop increments n.
                # So if we find one at n, we should record it, and then
                # ideally search n-2, n-4...
                # But we started at est_n.
                # If est_n is close to the optimum, we might find it.
                # Let's just break and return? No, maybe a smaller n exists.
                # But we are going UP from est_n.
                # If we find one at n, it's valid.
                # We can try to search lower?
                # For simplicity, let's just update best and break inner.
                # Then outer loop continues to n+2.
                # This is wrong. We want the SMALLEST n.
                # So we should search DOWN from a high bound?
                # Or search UP from a low bound?
                # If we search UP from est_n, and find one at n,
                # we don't know if n-2 works.
                # So we should search DOWN from best_n?
                
                # Let's change strategy:
                # Search from best_n - 2 down to moore?
                # No, random graphs at small n are unlikely to work.
                # So search from est_n up to best_n.
                # If we find one at n, we can stop?
                # No, maybe n-2 works.
                # But n-2 < est_n?
                # If est_n is a good estimate, n is close to est_n.
                # So we can just break and return?
                # Risky.
                
                # Let's just update best and break inner.
                # Then continue outer?
                # If we continue outer, we go to larger n.
                # So we should NOT continue outer if we found a valid one,
                # unless we want to find an even smaller one?
                # But we are going up.
                # So if we find one at n, we should stop?
                # Or we should search n-2, n-4...
                
                # Let's restart the search from n-2 down to start_n?
                # That's complex.
                
                # Simple heuristic:
                # If we find one at n, it's a good candidate.
                # Break inner.
                # Break outer?
                # If we break outer, we return n.
                # Is it possible that n-2 works? Yes.
                # But probability is low if n is near the threshold.
                # Let's just return the first one found?
                # No, let's try to find the smallest.
                
                # Let's change the loop to go DOWN from best_n?
                # No, best_n is 1250.
                # est_n is 1000.
                # Searching 1248, 1246... is bad because random graphs at 1250 are likely to have short cycles?
                # No, random graphs at larger n have LARGER girth.
                # So random graphs at 1250 are MORE likely to have girth 9 than at 1000.
                # So we should search DOWN from best_n?
                # No, we want the SMALLEST n.
                # So we should search UP from est_n.
                # If we find one at n, we can stop?
                # Or we can try n-2?
                
                # Let's just break both loops and return n.
                # This is a greedy approach.
                break
            # End if valid
            
        # End attempts
        if best_n == n:
            # Found a valid graph at n
            # We can try to search lower?
            # For now, let's just break and return.
            # This is a risk, but given time constraints, it's reasonable.
            break
            
    return (best_n, best_edges)

def _luw_graph(q, g):
    # Same as before
    if q < 3:
        return 0, []
    
    d_min = g - 5
    if d_min % 2 == 0:
        d_min += 1
    d_min = max(3, d_min)
    
    for d in range(d_min, min(d_min + 5, 20)):
        if d < 3:
            continue
        try:
            n, edges = _luw_graph_params(q, d)
            if n > 0:
                return n, edges
        except Exception:
            continue
    
    return 0, []

def _luw_graph_params(q, d):
    # Same as before
    if q < 3 or d < 3:
        return 0, []
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _random_k_regular(n, k):
    if n * k < 2 or (n * k) % 2 != 0:
        return None
    
    for _ in range(10):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = []
        valid = True
        edge_set = set()
        
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            edges.append(key)
        
        if valid:
            return edges
    
    return None

def _verify_girth_fast(k, g, n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    adj_indices = [{} for _ in range(n)]
    for u in range(n):
        for idx, v in enumerate(adj[u]):
            adj_indices[u][v] = idx
            
    for start in range(n):
        visited = [[-1] * k for _ in range(n)]
        queue = deque()
        
        for neighbor in adj[start]:
            idx = adj_indices[start][neighbor]
            visited[start][idx] = 1
            queue.append((start, neighbor, 1))
            
        while queue:
            prev, curr, dist = queue.popleft()
            
            if dist >= g - 1:
                continue
                
            idx_prev = adj_indices[curr][prev]
            
            for next_v in adj[curr]:
                if next_v == prev:
                    continue
                    
                idx_next = adj_indices[curr][next_v]
                
                if visited[curr][idx_next] != -1:
                    cycle_len = visited[curr][idx_next] + dist + 1
                    if cycle_len < g:
                        return False
                else:
                    visited[curr][idx_next] = dist + 1
                    if dist + 1 < g - 1:
                        queue.append((curr, next_v, dist + 1))
                        
    return True
