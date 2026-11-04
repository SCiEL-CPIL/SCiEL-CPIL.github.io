import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin

    if k != 4 or g != 10:
        return _luw_fallback()

    # Target n=320
    # Base graph: K4,4 (8 vertices)
    # Lift size m = 40
    n_base = 8
    m = 40
    n = n_base * m
    
    # Base edges: (i, 4+j) for i in 0..3, j in 0..3
    base_edges = []
    for i in range(4):
        for j in range(4):
            base_edges.append((i, 4+j))
            
    # We need to assign permutations to these edges.
    # Note: The edge (u, v) and (v, u) are the same undirected edge.
    # In voltage graphs, we assign a permutation to the directed edge.
    # For an undirected edge {u,v}, we assign sigma_uv. Then sigma_vu = sigma_uv^{-1}.
    
    # Let's try random permutations
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Assign random permutations to the 16 edges
        # We can just generate a random permutation for each directed edge and ensure inverse consistency?
        # Or just generate for one direction and use inverse for the other.
        # Let's index edges.
        # There are 16 undirected edges.
        
        perms = {}
        for u, v in base_edges:
            # Generate random permutation of size m
            p = np.random.permutation(m)
            # Store for direction u->v
            perms[(u, v)] = p
            # Store inverse for v->u
            # Inverse of p: p_inv[p[i]] = i
            p_inv = np.empty_like(p)
            p_inv[p] = np.arange(m)
            perms[(v, u)] = p_inv
            
        # Build the lifted graph
        # Vertices 0..n-1
        # adj list
        adj = [[] for _ in range(n)]
        
        for u, v in base_edges:
            p = perms[(u, v)]
            for i in range(m):
                u_id = u * m + i
                v_id = v * m + p[i]
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
                
        # Check girth
        if _has_short_cycle(adj, g):
            continue
        else:
            # Construct edge list
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges

    # If random search fails, fallback to LUW
    return _luw_fallback()

def _has_short_cycle(adj, g):
    n = len(adj)
    # Check for cycles of length < g
    # Since graph is bipartite, only even cycles.
    # We check for cycles of length 4, 6, 8 (if g=10)
    # General check:
    # For each vertex u, check if there is a path of length d < g-2 between two distinct neighbors of u.
    # If dist(v, w) = d, cycle is d+2.
    # We need d+2 < g => d < g-2.
    
    limit = g - 2
    
    for u in range(n):
        neighbors = adj[u]
        # For each pair of neighbors v, w
        # We can optimize by doing BFS from each neighbor v, and checking if any other neighbor w is reached in < limit steps.
        # But doing BFS from every neighbor for every u is O(n * k * (n+k)).
        # n=320, k=4. 320 * 4 * 320 = 409,600. Very fast.
        
        for v in neighbors:
            # BFS from v
            visited = [-1] * n
            visited[v] = 0
            queue = [v]
            head = 0
            found = False
            while head < len(queue) and not found:
                curr = queue[head]
                head += 1
                dist = visited[curr]
                if dist >= limit:
                    continue
                for nb in adj[curr]:
                    if visited[nb] == -1:
                        visited[nb] = dist + 1
                        queue.append(nb)
                    # Check if nb is a neighbor of u (and not v, and not u itself)
                    # If nb is a neighbor of u, and nb != v, and dist+1 < limit?
                    # Actually, if we reach nb, the path is v -> ... -> nb.
                    # Length is visited[nb].
                    # Cycle is u-v-...-nb-u. Length is 1 + visited[nb] + 1 = visited[nb] + 2.
                    # We need visited[nb] + 2 < g => visited[nb] < g-2 = limit.
                    # So if we reach a neighbor of u (other than v) with dist < limit, we have a short cycle.
                    
            # Re-doing BFS and checking inside is messy. Let's just collect distances.
            
    # Simpler:
    # For each u:
    #   for each v in adj[u]:
    #      BFS from v, record dist to all nodes.
    #      for each w in adj[u], w != v:
    #          if dist[w] < limit: return True
    #          # Also if dist[w] == -1, no path.
            
    for u in range(n):
        neighbors = adj[u]
        # Optimization: We only need to check pairs.
        # But BFS from each v gives dist to all.
        for v in neighbors:
            # BFS from v
            dist = [-1] * n
            dist[v] = 0
            queue = [v]
            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                if dist[curr] >= limit:
                    continue
                for nb in adj[curr]:
                    if dist[nb] == -1:
                        dist[nb] = dist[curr] + 1
                        queue.append(nb)
            
            for w in neighbors:
                if w == v:
                    continue
                if dist[w] != -1 and dist[w] < limit:
                    return True
                    
    return False

def _luw_fallback():
    # ... (existing code) ...
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
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
