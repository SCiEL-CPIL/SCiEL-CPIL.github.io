"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    q = k
    
    # Field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    def get_luw_graph(d):
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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

    def check_girth(n, edges, min_girth):
        if n == 0: return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def benson_psl2(q_val):
        """Benson construction: coset graph of PSL(2,q) with specific subgroup.
        Returns (n, edges) for a (q+1)-regular graph."""
        if q_val == 4:
            # GF(4) = {0,1,2,3} with 1+1=0, 2+2=0, 3+3=0, 1+2=3, etc.
            # PSL(2,4) ≅ A5, |PSL(2,4)| = 60
            # Generate PSL(2,4) and find a suitable subgroup
            # For degree 5, use the standard generators
            # This is complex; return None for now
            return None
        else:
            return None

    best_n = float('inf')
    best_edges = []
    
    # Try LUW construction first
    for d in range(1, 10):
        if time.time() > deadline - 20:
            break
            
        if d % 2 == 0:
            exp_girth = 2 * d + 2
        else:
            exp_girth = d + 5
            
        if exp_girth < g:
            continue
            
        try:
            n, edges = get_luw_graph(d)
            if n < best_n:
                if check_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
                    break
        except Exception:
            pass

    # If LUW gives large n, try greedy construction for smaller n
    if best_n > 200 and time.time() < deadline - 30:
        def greedy_construct(target_n, k_val, g_val, max_tries=100):
            """Try to build a k-regular graph with girth >= g on target_n vertices."""
            for attempt in range(max_tries):
                if time.time() > deadline - 10:
                    return None, None
                
                # Start with random regular graph
                if target_n * k_val % 2 != 0:
                    continue
                    
                adj = [set() for _ in range(target_n)]
                # Build random k-regular graph
                half_edges = target_n * k_val // 2
                edges_list = []
                deg = [0] * target_n
                remaining = list(range(target_n))
                random.shuffle(remaining)
                
                # Simple random regular graph construction
                for _ in range(half_edges):
                    # Pick two vertices with remaining degree
                    candidates = [v for v in range(target_n) if deg[v] < k_val and v not in adj[v]]
                    if len(candidates) < 2:
                        break
                    u = random.choice(candidates)
                    candidates.remove(u)
                    candidates = [v for v in candidates if v not in adj[u]]
                    if not candidates:
                        break
                    v = random.choice(candidates)
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    edges_list.append((u, v))
                
                if len(edges_list) != half_edges:
                    continue
                if any(deg[v] != k_val for v in range(target_n)):
                    continue
                
                # Check girth
                if check_girth(target_n, edges_list, g_val):
                    return target_n, edges_list
            return None, None
        
        # Try various target sizes
        for target_n in range(100, 200, 10):
            if time.time() > deadline - 15:
                break
            n, edges = greedy_construct(target_n, k, g, max_tries=20)
            if n is not None and n < best_n:
                best_n = n
                best_edges = edges
                break

    if best_n == float('inf'):
        d = 6
        try:
            n, edges = get_luw_graph(d)
            best_n = n
            best_edges = edges
        except:
            pass

    return best_n, best_edges
# EVOLVE-BLOCK-END
