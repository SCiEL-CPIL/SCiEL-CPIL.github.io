import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def luw_construct(k, g):
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

    if k != 3:
        return luw_construct(k, g)

    t_start = time.time()
    t_deadline = t_start + 110  # 10s margin

    # Try to find a small valid graph
    # Known best is 936. We try to find one.
    # We will try n values. If we find a valid graph, we return it.
    # To minimize n, we could try smaller n first, but they are harder.
    # Let's try n around 936.
    
    best_n = float('inf')
    best_edges = None
    fallback_n, fallback_edges = luw_construct(3, g)
    
    # List of n to try
    # We prioritize finding *any* valid graph first, then minimize.
    # But since 936 is the known best, trying 936 is a good bet.
    # If we fail, we try larger n to ensure we return *something* valid.
    
    n_candidates = [936, 940, 950, 1000, 1100, 1200, 1500, 2000]
    
    for n in n_candidates:
        if time.time() > t_deadline:
            break
            
        # Try a few restarts for this n
        found = False
        for _ in range(3):
            if time.time() > t_deadline:
                break
                
            # 1. Generate random 3-regular graph
            edges = random_3reg(n)
            if edges is None:
                continue
                
            # 2. Local search to increase girth
            edges = local_search(edges, n, g, t_deadline)
            
            # 3. Check girth
            girth_val = calculate_girth(edges, n)
            
            if girth_val >= g:
                if n < best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break # Stop restarts for this n if found
        
        if found:
            # If we found a valid graph, we can try to reduce n?
            # Given time constraints, we might not have time to search smaller n thoroughly.
            # However, if we found one at 936, that's the known best.
            # We can break here if we are confident.
            # Let's break to save time, unless we have lots of time left.
            if time.time() < t_deadline - 20:
                # Try to see if we can find a smaller one?
                # This is risky. Let's just return the found one.
                pass
            break

    if best_edges is not None:
        return best_n, best_edges
    else:
        return fallback_n, fallback_edges

def random_3reg(n):
    """Generate a random 3-regular graph on n vertices."""
    if n < 4 or n % 2 != 0:
        return None
        
    # Configuration model
    half_edges = n * 3 // 2
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
        
    random.shuffle(stubs)
    
    edges = []
    seen = set()
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            continue
        if (u, v) in seen or (v, u) in seen:
            continue
        edges.append((u, v))
        seen.add((u, v))
        seen.add((v, u))
        
    if len(edges) != n * 3 // 2:
        return None
        
    # Verify degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 3 for d in deg):
        return None
        
    return edges

def calculate_girth(edges, n):
    """Calculate girth of a graph using BFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    min_girth = float('inf')
    
    for start in range(n):
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
                elif parent[u] != v and parent[v] != u:
                    # Cycle found
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth == 3:
                            return 3
                            
    return min_girth

def find_short_cycle(edges, n, max_len):
    """Find a cycle of length <= max_len. Returns list of nodes in cycle."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
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
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len <= max_len:
                        # Reconstruct cycle
                        cycle = []
                        curr = u
                        while curr != v:
                            cycle.append(curr)
                            curr = parent[curr]
                        cycle.append(v)
                        # Add path from v to start? No, the cycle is u->...->v->u
                        # Wait, BFS tree path is u->...->start and v->...->start.
                        # The cycle is u->...->lca->...->v->u.
                        # My reconstruction above is wrong.
                        
                        # Correct reconstruction:
                        path_u = []
                        curr = u
                        while curr != -1:
                            path_u.append(curr)
                            curr = parent[curr]
                        path_v = []
                        curr = v
                        while curr != -1:
                            path_v.append(curr)
                            curr = parent[curr]
                        
                        # Find LCA
                        set_u = set(path_u)
                        lca = None
                        for node in path_v:
                            if node in set_u:
                                lca = node
                                break
                                
                        if lca is not None:
                            # Cycle is u->...->lca->...->v->u
                            cycle_u = []
                            curr = u
                            while curr != lca:
                                cycle_u.append(curr)
                                curr = parent[curr]
                            cycle_u.append(lca)
                            
                            cycle_v = []
                            curr = v
                            while curr != lca:
                                cycle_v.append(curr)
                                curr = parent[curr]
                            cycle_v.append(lca)
                            
                            cycle = cycle_u + cycle_v[:-1][::-1]
                            return cycle
                            
    return None

def local_search(edges, n, g, deadline):
    """Improve girth using local search."""
    # Convert to set for fast lookup
    edge_set = set()
    for u, v in edges:
        if u > v: u, v = v, u
        edge_set.add((u, v))
        
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        girth_val = calculate_girth(edges, n)
        if girth_val >= g:
            break
            
        # Find a short cycle
        cycle = find_short_cycle(edges, n, girth_val)
        if not cycle:
            break
            
        # Try to break the cycle
        success = False
        for _ in range(20):
            if time.time() > deadline:
                break
                
            # Pick a random edge in the cycle
            idx = random.randint(0, len(cycle) - 1)
            u = cycle[idx]
            v = cycle[(idx + 1) % len(cycle)]
            
            # Find another edge (w, x) to swap with
            # We need to pick w, x such that swapping (u,v) and (w,x) with (u,w) and (v,x) is valid
            # Let's pick a random edge (w, x) not in the cycle
            if len(edges) < 2:
                break
            w, x = random.choice(edges)
            if (w, x) == (u, v) or (w, x) == (v, u):
                continue
            if w == u or w == v or x == u or x == v:
                continue
                
            # Check if swap is valid
            # New edges: (u, w) and (v, x)
            # Must not exist
            if (u, w) in edge_set or (w, u) in edge_set or (v, x) in edge_set or (x, v) in edge_set:
                continue
            if u == w or v == x:
                continue
                
            # Perform swap
            edge_set.remove((u, v) if u < v else (v, u))
            edge_set.remove((w, x) if w < x else (x, w))
            edge_set.add((u, w) if u < w else (w, u))
            edge_set.add((v, x) if v < x else (x, v))
            
            # Update edges list
            edges = list(edge_set)
            
            # Update adj
            # Rebuild adj for simplicity (slow but n is small)
            adj = [[] for _ in range(n)]
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
                
            success = True
            break
            
        if not success:
            # Try a different strategy? Or just restart local search?
            # For now, break and let outer loop handle it
            break
            
    return edges
# EVOLVE-BLOCK-END
