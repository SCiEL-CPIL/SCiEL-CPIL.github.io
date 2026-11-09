import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if not (k == 5 and g == 9):
        return _seed_construct(k, g)

    # Try to find a graph with n < 1250
    # We will try local search on n=1116, 1120, 1150
    # If we find a valid graph, return it.
    # Otherwise, return seed.
    
    n_candidates = [1116, 1120, 1150]
    
    for n in n_candidates:
        if time.time() > deadline - 10:
            break
        result = _attempt_local_search(n, k, g, deadline - 5)
        if result is not None:
            return result

    return _seed_construct(k, g)

def _attempt_local_search(n, k, g, deadline):
    # Generate random k-regular graph
    # Try to increase girth via 2-switches targeting short cycles
    
    for restart in range(3):
        if time.time() > deadline:
            return None
        
        adj, is_simple = generate_random_regular_graph(n, k)
        if not is_simple:
            continue
            
        # Local search
        adj = _local_search_girth(adj, n, k, g, deadline)
        
        # Check final girth
        girth = compute_girth(adj, n)
        if girth >= g:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges

    return None

def _local_search_girth(adj, n, k, g, deadline):
    # Loop until time or girth >= g
    # We use a "cycle breaking" strategy.
    # Find a short cycle, pick an edge in it, try to swap it with an edge outside.
    
    while time.time() < deadline:
        # Find a cycle of length < g
        cycle = find_short_cycle(adj, n, g)
        if cycle is None:
            return adj # Girth >= g
        
        # cycle is a list of vertices [v0, v1, ..., v_{L-1}]
        # Pick a random edge in the cycle: (u, v)
        # Let u = cycle[i], v = cycle[(i+1)%L]
        
        # We want to replace (u, v) with (u, w) and (v, x) ? No, 2-switch replaces (u,v) and (w,x) with (u,w) and (v,x).
        # To break the cycle, we need to remove an edge from it.
        # Let's pick a random edge (u, v) from the cycle.
        # We need another edge (w, x) NOT in the cycle (or at least not forming a shorter cycle) to swap with.
        # Actually, any edge (w, x) such that the swap is valid (simple graph) and doesn't create a shorter cycle immediately.
        
        # To be efficient, we just pick random edges (w, x) and try to swap.
        # If the swap is valid and the resulting graph has girth >= current_girth (or just doesn't create a 3-cycle), accept.
        # To ensure progress, we might require that the specific cycle is broken.
        
        # Let's try:
        # 1. Pick edge (u, v) in cycle.
        # 2. Pick random edge (w, x) from graph.
        # 3. Check if swap (u,v),(w,x) -> (u,w),(v,x) is valid (no loops, no multi-edges).
        # 4. Check if (u,w) and (v,x) are not in the graph.
        # 5. Perform swap.
        # 6. Check if girth increased or stayed same? 
        #    Checking girth is expensive.
        #    Instead, check if the specific cycle is broken.
        #    If the cycle is broken, accept. If not, reject.
        #    This might get stuck if breaking one cycle creates another.
        #    But it's a start.

        # Optimization: We can pre-compute the set of edges in the cycle.
        cycle_edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1)%len(cycle)]
            cycle_edges.add((min(u,v), max(u,v)))
        
        # Pick a random edge from the cycle to break
        u, v = random.choice(list(cycle_edges))
        
        # Find a candidate edge (w, x) to swap with
        # To avoid creating a short cycle, w and x should be far from u and v?
        # For now, just random.
        
        found_swap = False
        for _ in range(100): # Try 100 random swaps
            # Pick random edge (w, x)
            # We can iterate over all edges or pick random vertices.
            # Picking random vertices is faster.
            w = random.randint(0, n-1)
            x = random.randint(0, n-1)
            if w == x: continue
            if x not in adj[w]: continue # (w,x) must be an edge
            
            # Ensure (w,x) is not (u,v) or (v,u)
            if (w == u and x == v) or (w == v and x == u): continue
            
            # Check validity of swap (u,v), (w,x) -> (u,w), (v,x)
            # 1. u != w, u != x, v != w, v != x (to avoid loops in new edges? No, u-w is loop if u==w)
            if u == w or u == x or v == w or v == x: continue
            
            # 2. (u, w) not in adj[u]
            # 3. (v, x) not in adj[v]
            if w in adj[u] or x in adj[v]: continue
            
            # 4. (u, x) and (v, w) are the other swap? No, standard 2-switch is (u,v) and (w,x) -> (u,w) and (v,x) OR (u,x) and (v,w).
            # Let's try (u,w) and (v,x).
            
            # Check if new edges create triangles
            common_uw = (set(adj[u]) - {v}) & (set(adj[w]) - {x})
            common_vx = (set(adj[v]) - {u}) & (set(adj[x]) - {w})
            
            if common_uw or common_vx:
                continue

            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj[u].append(w)
            adj[w].append(u)
            adj[v].append(x)
            adj[x].append(v)
            
            # Check if the specific cycle is broken.
            # The cycle contained (u,v). Since (u,v) is removed, the cycle is broken.
            # But did we create a new shorter cycle?
            # We can't check fully. But we accept this move.
            found_swap = True
            break
            
        if not found_swap:
            # If we couldn't find a swap, break to avoid infinite loop
            break

    return adj

def find_short_cycle(adj, n, g):
    # BFS from each node to find shortest cycle
    # Return the cycle path if length < g
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        # Reconstruct cycle
                        path_u = [u]
                        while path_u[-1] != start:
                            path_u.append(parent[path_u[-1]])
                        path_u.reverse()
                        
                        path_v = [v]
                        while path_v[-1] != start:
                            path_v.append(parent[path_v[-1]])
                        path_v.reverse()
                        
                        # Cycle is path_u + path_v (excluding start duplicate?)
                        # path_u: start ... u
                        # path_v: start ... v
                        # Cycle: u ... start ... v ... u ?
                        # So list is path_u (reversed to u->start) + path_v (start->v)
                        # Wait, path_u is start->...->u. Reversed is u->...->start.
                        # path_v is start->...->v.
                        # So cycle is path_u_reversed + path_v[1:]
                        # Let's verify.
                        # u -> ... -> start -> ... -> v -> u (edge v-u exists? No, v and u are neighbors in BFS tree? No, they are connected by an edge in the graph, but not in the tree necessarily. The edge (u,v) closes the cycle.)
                        # So the cycle is: u, ..., start, ..., v, u.
                        cycle = list(reversed(path_u)) + path_v[1:]
                        return cycle
    return None

def generate_random_regular_graph(n, k):
    if (n * k) % 2 != 0:
        return None, False
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = set()
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None, False
        if v in adj[u]:
            return None, False
        adj[u].add(v)
        adj[v].add(u)
        edges.add((min(u, v), max(u, v)))
    adj_list = [list(s) for s in adj]
    return adj_list, True

def compute_girth(adj_list, n):
    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth == 3:
                        return 3
    return min_girth if min_girth != float('inf') else n

def _seed_construct(k, g):
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
