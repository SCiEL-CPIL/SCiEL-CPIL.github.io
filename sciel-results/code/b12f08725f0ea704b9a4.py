"""De Bruijn-type finite-state construction for regular graphs of large girth."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; keep margin before 150s kill

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

    # Helper: Check girth using non-backtracking BFS from every vertex
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each vertex with non-backtracking
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # De Bruijn-type deterministic construction
    def de_bruijn_construct(k, g):
        """
        Construct a k-regular graph using a De Bruijn-type finite-state machine.
        Vertices are words of length m over alphabet of size k.
        Edges: word w = a1...am connects to shift(w) with new last symbol.
        For undirected k-regular graph, we use a different approach:
        
        Actually, for a proper undirected k-regular graph with high girth,
        let's use the following:
        - Vertices: Z_m x Z_m (m*m vertices)
        - Each vertex (u,v) has neighbors:
          (u+1, v + f1(u)), (u-1, v + f2(u)), (u, v + s)
        where f1, f2: Z_m -> Z_m and s in Z_m are chosen to maximize girth.
        
        For k=3, g=16: Moore bound is 512. Try m=24 (n=576), m=26 (n=676), etc.
        """
        if k != 3:
            return None
        
        best_result = None
        best_n = float('inf')
        
        # Try different m values
        for m in range(24, 33, 2):
            if time.time() - t0 > deadline - 20:
                break
            n = m * m
            if n < 512:
                continue
            
            # Try several random initializations with simulated annealing
            for trial in range(5):
                if time.time() - t0 > deadline - 20:
                    break
                
                f1 = [random.randint(0, m-1) for _ in range(m)]
                f2 = [random.randint(0, m-1) for _ in range(m)]
                s = random.randint(1, m-1)
                
                def compute_girth_fast(f1, f2, s, m, target_girth):
                    """Compute girth by checking for short cycles."""
                    n = m * m
                    adj = [[] for _ in range(n)]
                    
                    for u in range(m):
                        for v in range(m):
                            idx = u * m + v
                            # Neighbor 1
                            u1 = (u + 1) % m
                            v1 = (v + f1[u]) % m
                            adj[idx].append(u1 * m + v1)
                            # Neighbor 2
                            u2 = (u - 1) % m
                            v2 = (v + f2[u]) % m
                            adj[idx].append(u2 * m + v2)
                            # Neighbor 3
                            v3 = (v + s) % m
                            adj[idx].append(u * m + v3)
                    
                    # Check girth by BFS from each vertex
                    for start in range(n):
                        dist = [-1] * n
                        parent = [-1] * n
                        dist[start] = 0
                        queue = [start]
                        qi = 0
                        while qi < len(queue):
                            u = queue[qi]
                            qi += 1
                            if dist[u] >= target_girth // 2:
                                continue
                            for v in adj[u]:
                                if v == parent[u]:
                                    continue
                                if dist[v] == -1:
                                    dist[v] = dist[u] + 1
                                    parent[v] = u
                                    queue.append(v)
                                else:
                                    cycle_len = dist[u] + dist[v] + 1
                                    if cycle_len < target_girth:
                                        return cycle_len
                    return target_girth
                
                current_girth = compute_girth_fast(f1, f2, s, m, g)
                best_girth = current_girth
                best_f1 = f1[:]
                best_f2 = f2[:]
                best_s = s
                
                # Simulated annealing
                T = 500.0
                cooling = 0.97
                max_iters = 200
                
                for it in range(max_iters):
                    if time.time() - t0 > deadline - 20:
                        break
                    
                    # Propose a move
                    move_type = random.random()
                    if move_type < 0.4:
                        idx = random.randint(0, m-1)
                        f1[idx] = (f1[idx] + random.randint(-1, 1)) % m
                    elif move_type < 0.8:
                        idx = random.randint(0, m-1)
                        f2[idx] = (f2[idx] + random.randint(-1, 1)) % m
                    else:
                        s = (s + random.randint(-1, 1)) % m
                        if s == 0:
                            s = 1
                    
                    new_girth = compute_girth_fast(f1, f2, s, m, g)
                    
                    if new_girth >= current_girth:
                        current_girth = new_girth
                        if new_girth > best_girth:
                            best_girth = new_girth
                            best_f1 = f1[:]
                            best_f2 = f2[:]
                            best_s = s
                    else:
                        if random.random() < pow(2.71828, (new_girth - current_girth) / T):
                            current_girth = new_girth
                    
                    T *= cooling
                    
                    if best_girth >= g:
                        break
                
                if best_girth >= g:
                    # Construct the edge list
                    f1 = best_f1
                    f2 = best_f2
                    s = best_s
                    edges = []
                    seen = set()
                    for u in range(m):
                        for v in range(m):
                            idx = u * m + v
                            # Neighbor 1
                            u1 = (u + 1) % m
                            v1 = (v + f1[u]) % m
                            nidx = u1 * m + v1
                            key = (min(idx, nidx), max(idx, nidx))
                            if key not in seen:
                                seen.add(key)
                                edges.append((idx, nidx))
                            # Neighbor 2
                            u2 = (u - 1) % m
                            v2 = (v + f2[u]) % m
                            nidx = u2 * m + v2
                            key = (min(idx, nidx), max(idx, nidx))
                            if key not in seen:
                                seen.add(key)
                                edges.append((idx, nidx))
                            # Neighbor 3
                            v3 = (v + s) % m
                            nidx = u * m + v3
                            key = (min(idx, nidx), max(idx, nidx))
                            if key not in seen:
                                seen.add(key)
                                edges.append((idx, nidx))
                    
                    # Verify it's 3-regular and simple
                    deg = [0] * n
                    for a, b in edges:
                        deg[a] += 1
                        deg[b] += 1
                    if all(d == 3 for d in deg) and len(edges) == n * 3 // 2:
                        if n < best_n:
                            best_n = n
                            best_result = (n, edges)
                            return best_result  # Return immediately for smallest n
        
        return best_result

    # Main logic
    if k == 3 and g == 16:
        # Try De Bruijn-type construction first
        db_result = de_bruijn_construct(k, g)
        if db_result is not None:
            return db_result
        
        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
# EVOLVE-BLOCK-END
