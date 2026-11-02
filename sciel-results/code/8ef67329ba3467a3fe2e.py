import time
import math
import random
import numpy as np
from collections import deque

def construct(k, g):
    deadline = time.time() + 115
    
    if k != 3:
        return _luw_fallback(k, g)
        
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Phase 1: Circulant Search
    # Use NumPy check for speed
    max_circ_n = 2000
    for n in range(moore_bound, max_circ_n + 1, 2):
        if time.time() > deadline - 5: # Reserve time for random phase
            break
        h = n // 2
        for a in range(1, h):
            if math.gcd(h, a) != 1:
                continue
            if _check_girth_circulant_numpy(n, a, g):
                edges = []
                for u in range(n):
                    for s in (a, n - a, h):
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
    # Phase 2: Random Graph Search
    # Start from moore_bound, go up
    # We want to find the smallest n
    # If we fail at n=512, try n=514, etc.
    # But we must be careful with time.
    # Let's try a few n values with a few random graphs each.
    
    # Heuristic: The probability of girth >= 16 is low for n=512.
    # It increases with n.
    # Let's try n from 512 to 1000.
    # For each n, try ~10-20 random graphs.
    
    n = moore_bound
    while n < 2000:
        if time.time() > deadline - 10:
            break
            
        # Try a few random graphs for this n
        # Limit attempts to keep time per n low
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts:
            if time.time() > deadline - 10:
                break
                
            adj = _generate_random_3_reg(n)
            if adj is None:
                attempts += 1
                continue
                
            if _check_girth_general(adj, g):
                edges = _get_edges_from_adj(adj)
                return n, edges
                
            attempts += 1
            
        n += 2
        
    return _luw_fallback(k, g)

def _check_girth_circulant_numpy(n, a, g):
    h = n // 2
    S = np.array([a, n - a, h])
    neg_idx = [1, 0, 2]
    
    # R[i, v] is True if we can reach v from 0 with i steps of type S[i] 
    # Wait, the previous numpy check was:
    # R[0, a] = True
    # R_new[j] |= roll(R[i], S[j])
    # This tracks reachable vertices at depth d with last step type j.
    # If 0 is in R_new, we have a cycle.
    
    R = np.zeros((3, n), dtype=bool)
    R[0, a % n] = True
    R[1, (n - a) % n] = True
    R[2, h % n] = True
    
    for d in range(2, g):
        R_new = np.zeros((3, n), dtype=bool)
        for i in range(3):
            for j in range(3):
                if j == neg_idx[i]:
                    continue
                R_new[j] |= np.roll(R[i], S[j])
        R = R_new
        if R[:, 0].any():
            return False
            
    return True

def _generate_random_3_reg(n):
    # Configuration model
    # Create list of 3*n stubs
    # Pair them randomly
    # Check for loops and multi-edges
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
        
    random.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    
    for i in range(0, 3*n, 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        if v in adj[u]:
            return None # Multi-edge
        adj[u].add(v)
        adj[v].add(u)
        
    # Verify degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
            
    return adj

def _check_girth_general(adj, g):
    n = len(adj)
    # BFS from each vertex
    # To speed up, we can just check for cycles < g
    # Since g=16, we check up to depth 7 or 8.
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        q = deque([start])
        dist[start] = 0
        
        while q:
            u = q.popleft()
            du = dist[u]
            
            if du >= g // 2: # We only need to find cycles < g
                # Actually, if we go deeper, we might find longer cycles, 
                # but we stop if we find a short one.
                # If we reach depth g//2 without finding a cycle, 
                # we don't guarantee no cycle of length g-1 (odd)?
                # Girth is min cycle length.
                # If no cycle < g, girth >= g.
                # A cycle of length L involves nodes with dist summing to L-1.
                # If L < g, then L-1 < g-1.
                # In BFS tree, an edge (u,v) where v is visited and v != parent[u]
                # creates a cycle of length dist[u] + dist[v] + 1.
                # If this length < g, return False.
                
                # We can prune BFS at depth g//2?
                # If dist[u] + dist[v] + 1 < g, and dist[u] <= dist[v],
                # then dist[u] < g/2.
                # So we only need to explore up to depth g//2.
                pass
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    # Found a visited node
                    # Check if it's the parent (back edge)
                    if v == parent[u]:
                        continue
                    
                    # Cycle length
                    # The cycle is start -> ... -> u -> v -> ... -> start?
                    # No, the cycle is in the BFS tree of start?
                    # Not necessarily. The edge (u,v) closes a cycle.
                    # The cycle length is dist[u] + dist[v] + 1.
                    # BUT this is only valid if u and v are in the same BFS tree 
                    # and the path from start to u and start to v are disjoint 
                    # except for start?
                    # Actually, the standard BFS girth check:
                    # For each edge (u,v) where dist[u] + dist[v] + 1 < g, 
                    # and u,v are not parent-child in a way that implies 
                    # the path is just the tree edge...
                    # Wait.
                    # If we run BFS from `start`, and we find an edge (u,v) 
                    # where both are visited, the cycle length is 
                    # dist[u] + dist[v] + 1 ONLY IF the paths from start 
                    # to u and start to v are disjoint except at start?
                    # No. The cycle is formed by the path start->u, edge u-v, 
                    # and path v->start.
                    # The length is dist[u] + 1 + dist[v].
                    # This assumes the paths don't share nodes other than start.
                    # In a BFS tree, the path from start to u is unique.
                    # The path from start to v is unique.
                    # If they share a node w (not start), then w is the LCA.
                    # The cycle length is (dist[u] - dist[w]) + 1 + (dist[v] - dist[w]).
                    # This is <= dist[u] + dist[v] + 1.
                    # So if dist[u] + dist[v] + 1 < g, then the actual cycle 
                    # length is < g.
                    # So checking dist[u] + dist[v] + 1 < g is a sufficient 
                    # condition for a short cycle.
                    
                    if dist[u] + dist[v] + 1 < g:
                        return False
                        
    return True

def _get_edges_from_adj(adj):
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _luw_fallback(k, g):
    # Same as before
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
