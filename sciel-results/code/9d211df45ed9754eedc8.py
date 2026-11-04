import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # --- Fast girth verification via non-backtracking BFS ---
    def _check_girth(adj, n, g_limit):
        """Check that no cycle shorter than g_limit exists."""
        max_depth = (g_limit - 1) // 2
        for start in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] < max_depth:
                            queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g_limit:
                            return False
        return True

    def _find_short_cycle(adj, n, g_limit):
        """Find a cycle shorter than g_limit. Returns list of vertices or None."""
        for start in range(n):
            parent = np.full(n, -1, dtype=np.int32)
            dist = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g_limit // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a back edge
                        # Reconstruct cycle
                        u_cur = u
                        v_cur = v
                        cycle_u = []
                        cycle_v = []
                        while u_cur != start:
                            cycle_u.append(u_cur)
                            u_cur = parent[u_cur]
                        cycle_u.append(start)
                        
                        while v_cur != start:
                            cycle_v.append(v_cur)
                            v_cur = parent[v_cur]
                        cycle_v.append(start)
                        
                        # The cycle is cycle_u + [edge u-v] + reversed(cycle_v)
                        # But we just need to return the vertices involved
                        return cycle_u + cycle_v[::-1]
        return None

    def _generate_random_3_regular(n):
        """Generate a random 3-regular graph using configuration model."""
        half_edges = []
        for i in range(n):
            for _ in range(3):
                half_edges.append(i)
        
        while True:
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, n * 3, 2):
                u = half_edges[i]
                v = half_edges[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj

    # --- LUW Graph Construction (Fallback) ---
    def _luw_graph(k, g):
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

    # --- Main logic ---
    if k == 3 and g == 16:
        # Try random local search for a small graph
        target_n = 1000
        adj = _generate_random_3_regular(target_n)
        
        # Local search to increase girth
        while time.time() < deadline - 10:
            if _check_girth(adj, target_n, g):
                # Convert adj to edge list
                edges = []
                for u in range(target_n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return target_n, edges
            
            # Find a short cycle to break
            cycle = _find_short_cycle(adj, target_n, g)
            if cycle is None:
                # Should not happen if check failed, but safety
                break
            
            # Pick two random edges from the cycle to swap
            # A 2-switch involves removing (u,v) and (w,x) and adding (u,w) and (v,x)
            # We need to pick edges that are in the cycle.
            # The cycle is a list of vertices. Edges are (cycle[i], cycle[i+1]).
            # We pick two distinct edges from the cycle.
            if len(cycle) < 4:
                break
                
            # Get unique edges in the cycle
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if u < v:
                    cycle_edges.append((u, v))
                else:
                    cycle_edges.append((v, u))
            
            if len(cycle_edges) < 2:
                continue
                
            # Pick two edges to swap
            e1, e2 = random.sample(cycle_edges, 2)
            u1, v1 = e1
            u2, v2 = e2
            
            # New edges: (u1, u2) and (v1, v2) or (u1, v2) and (v1, u2)
            # Try both, pick one that doesn't create multi-edges or loops
            new_edges_options = [
                [(u1, u2), (v1, v2)],
                [(u1, v2), (v1, u2)]
            ]
            
            swapped = False
            for new_e1, new_e2 in new_edges_options:
                # Check validity
                if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
                    continue
                if new_e2[0] in adj[new_e1[0]] or new_e1[0] in adj[new_e2[0]]:
                    continue
                    
                # Perform swap
                # Remove old edges
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                # Add new edges
                adj[new_e1[0]].append(new_e1[1])
                adj[new_e1[1]].append(new_e1[0])
                adj[new_e2[0]].append(new_e2[1])
                adj[new_e2[1]].append(new_e2[0])
                
                swapped = True
                break
            
            if not swapped:
                # Try another cycle or random swap
                continue

        # Fallback to LUW if search failed
        n_luw, edges_luw = _luw_graph(k, g)
        return n_luw, edges_luw

    # General fallback
    n_luw, edges_luw = _luw_graph(k, g)
    return n_luw, edges_luw
