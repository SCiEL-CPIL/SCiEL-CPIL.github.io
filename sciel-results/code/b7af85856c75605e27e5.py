import time
import random
from collections import deque

def _bfs_girth_fast(adj, target_girth):
    """Compute girth via non-backtracking BFS. Returns girth (or 0 if none found < target)."""
    n = len(adj)
    best = target_girth + 1
    for s in range(n):
        visited = {}
        dq = deque()
        for v in adj[s]:
            visited[(s, v)] = 1
            dq.append((s, v))
        while dq:
            u, prev = dq.popleft()
            d = visited[(u, prev)]
            if d >= best:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                if (v, u) in visited:
                    cycle = d + 1
                    if cycle < best:
                        best = cycle
                        if best < target_girth:
                            return best
                else:
                    visited[(v, u)] = d + 1
                    if d + 1 < best:
                        dq.append((v, u))
        if best < target_girth:
            return best
    return 0 # Girth >= target

def _find_shortest_cycle(adj):
    """Returns the list of vertices in the shortest cycle, or None."""
    n = len(adj)
    best_len = float('inf')
    best_cycle = None
    for s in range(n):
        # BFS storing parent
        parent = {}
        dist = {s: 0}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            if dist[u] >= best_len:
                continue
            for v in adj[u]:
                if v == s and dist[u] > 0:
                    # Found cycle back to start
                    cycle_len = dist[u] + 1
                    if cycle_len < best_len:
                        # Reconstruct cycle
                        path = [u]
                        curr = u
                        while curr != s:
                            curr = parent[curr]
                            path.append(curr)
                        path.reverse()
                        best_cycle = path
                        best_len = cycle_len
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    dq.append(v)
                else:
                    # Back edge or cross edge
                    # In undirected graph, if v is visited and v != parent[u], 
                    # we have a cycle.
                    # But BFS tree edges are only one way.
                    # If v is in dist and v != parent[u] (and u != parent[v]), 
                    # then cycle length = dist[u] + dist[v] + 1
                    if v != parent.get(u, -1) and u != parent.get(v, -1):
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < best_len:
                            # Reconstruct
                            path1 = []
                            curr = u
                            while curr != s:
                                path1.append(curr)
                                curr = parent[curr]
                            path1.append(s)
                            path1.reverse()
                            
                            path2 = []
                            curr = v
                            while curr != s:
                                path2.append(curr)
                                curr = parent[curr]
                            path2.append(s)
                            path2.reverse()
                            
                            # Merge
                            # Find LCA? No, both go to s.
                            # Cycle is path1 + path2[-1:-1:-1] (excluding s twice)
                            # Actually, just concatenate and remove duplicates
                            cycle = path1 + path2[-1:-1:-1]
                            # Remove duplicate s
                            if cycle[0] == cycle[-1]:
                                cycle.pop()
                            best_cycle = cycle
                            best_len = cycle_len
    return best_cycle

def _generate_random_4_reg_bipartite(n):
    """Generates a random 4-regular bipartite graph on n vertices (n even)."""
    n_half = n // 2
    left = list(range(n_half))
    right = list(range(n_half, n))
    
    # Stubs
    left_stubs = [i for i in left for _ in range(4)]
    right_stubs = [i for i in right for _ in range(4)]
    
    # Random matching
    random.shuffle(right_stubs)
    edges = set()
    valid = True
    for u, v in zip(left_stubs, right_stubs):
        if u == v: # Loop (impossible in bipartite)
            valid = False
            break
        if (u, v) in edges: # Multi-edge
            valid = False
            break
        edges.add((u, v))
    
    if not valid:
        return None
        
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _construct_4_10_cage(deadline):
    """Construct the (4,10)-cage using local search."""
    n = 320
    best_adj = None
    best_girth = 0
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        adj = _generate_random_4_reg_bipartite(n)
        if not adj:
            continue
            
        # Local search
        for step in range(20000):
            if time.time() > deadline - 5:
                break
                
            girth = _bfs_girth_fast(adj, 10)
            if girth == 0:
                return n, _adj_to_edges(adj)
            
            # girth < 10
            if girth > best_girth:
                best_girth = girth
                best_adj = [lst[:] for lst in adj]
                
            # Try to improve girth
            cycle = _find_shortest_cycle(adj)
            if not cycle:
                break
                
            # Pick two edges on the cycle to switch
            # We want to break the cycle
            # Randomly pick two non-adjacent edges in the cycle
            L = len(cycle)
            if L < 4:
                continue
                
            # Get edges in cycle
            cycle_edges = []
            for i in range(L):
                u = cycle[i]
                v = cycle[(i+1)%L]
                cycle_edges.append((u, v))
                
            # Try a few random 2-switches
            for _ in range(10):
                # Pick two edges
                e1_idx, e2_idx = random.sample(range(L), 2)
                # Ensure they are not adjacent in the cycle (to avoid creating triangles easily)
                if abs(e1_idx - e2_idx) % L in [1, L-1]:
                    continue
                    
                u1, v1 = cycle_edges[e1_idx]
                u2, v2 = cycle_edges[e2_idx]
                
                # Switch: remove (u1,v1), (u2,v2), add (u1,v2), (u2,v1)
                # Check if new edges exist
                if (u1, v2) in set(tuple(sorted(e)) for e in adj[u1]):
                    continue
                if (u2, v1) in set(tuple(sorted(e)) for e in adj[u2]):
                    continue
                    
                # Apply switch
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                adj[u1].append(v2)
                adj[v2].append(u1)
                adj[u2].append(v1)
                adj[v1].append(u2)
                
                # Check if girth improved (or at least didn't get worse)
                new_girth = _bfs_girth_fast(adj, 10)
                if new_girth == 0:
                    return n, _adj_to_edges(adj)
                if new_girth > girth:
                    break # Accept and restart loop
                elif new_girth == girth:
                    # Keep it? Maybe random walk
                    if random.random() < 0.1:
                        break
                else:
                    # Revert
                    adj[u1].remove(v2)
                    adj[v2].remove(u1)
                    adj[u2].remove(v1)
                    adj[v1].remove(u2)
                    
                    adj[u1].append(v1)
                    adj[v1].append(u1)
                    adj[u2].append(v2)
                    adj[v2].append(u2)
                    
    # Return best found or None
    if best_adj:
        return n, _adj_to_edges(best_adj)
    return None

def _adj_to_edges(adj):
    edges = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction (from seed)."""
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130
    
    if k == 4 and g == 10:
        try:
            result = _construct_4_10_cage(deadline)
            if result:
                n, edges = result
                # Verify
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                for i in range(n):
                    if len(adj[i]) != k:
                        raise ValueError
                girth = _bfs_girth_fast(adj, g)
                if girth == 0:
                    return n, edges
        except Exception:
            pass
        
        # Fallback
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)
