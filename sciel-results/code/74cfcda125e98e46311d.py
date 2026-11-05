"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    # Keep a safety margin of 10 seconds
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
        # Rule definition from seed
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
        
        # BFS from each vertex to find shortest cycle
        # We can optimize by stopping BFS if depth exceeds min_girth//2
        # but for small n, full BFS is fine.
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                # If we have already found a cycle shorter than min_girth, we return False
                # But we check inside the loop.
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def check_girth_fast(adj, n, g):
        """Check if graph has girth >= g using BFS from each vertex up to depth g//2."""
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    elif dist[v] != dist[u] + 1:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def count_short_cycles_fast(adj, n):
        """Count triangles and 4-cycles as fast penalty."""
        count = 0
        adj_set = [set(a) for a in adj]
        for u in range(n):
            for i in range(len(adj[u])):
                for j in range(i + 1, len(adj[u])):
                    if adj[u][j] in adj_set[adj[u][i]]:
                        count += 1
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    common = adj_set[u] & adj_set[v]
                    count += len(common)
        return count

    def gen_random_k_reg(n, k):
        for attempt in range(500):
            stubs = [i for i in range(n) for _ in range(k)]
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid and all(len(a) == k for a in adj):
                return adj
        return None

    def optimize(n, k, g, time_limit):
        adj = gen_random_k_reg(n, k)
        if adj is None:
            return None

        # Initial check
        if not check_girth_fast(adj, n, g):
            return None

        # Try to improve by local search
        # This is a placeholder for more sophisticated local search
        # For now, just return the random graph if it's valid
        return adj

    best_n = float('inf')
    best_edges = []
    
    # Try d from 1 upwards
    # For q=5, d=4 is expected to be the first valid for g=9
    # d=1,2,3 will be rejected by girth check or expected girth
    for d in range(1, 10):
        if time.time() > deadline - 20:
            break
            
        # Quick filter: expected girth
        if d % 2 == 0:
            exp_girth = 2 * d + 2
        else:
            exp_girth = d + 5
            
        if exp_girth < g:
            continue
            
        try:
            n, edges = get_luw_graph(d)
            # Only check girth if n is smaller than current best
            if n < best_n:
                if check_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
                    break
        except Exception:
            pass

    if best_n == float('inf'):
        # Fallback: use a large d to ensure validity, even if n is large
        # d=10 for q=5 is 2*5^10 ~ 2M vertices. Too big.
        # d=6 is 31250.
        d = 6
        try:
            n, edges = get_luw_graph(d)
            best_n = n
            best_edges = edges
        except:
            # Last resort: return empty or minimal valid?
            # Must return valid.
            # If construction fails, return a trivial graph? No, must be k-regular.
            # For k=5, minimum n is 5? K5 has girth 3.
            # If we can't construct, we might fail.
            # But LUW construction is robust for primes.
            pass

    return best_n, best_edges
# EVOLVE-BLOCK-END
