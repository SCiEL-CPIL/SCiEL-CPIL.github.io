import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # Fallback: LUW graph (current implementation)
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

    # Try to find a smaller cyclic graph
    if k == 5 and g == 9:
        # Search for cyclic graph Z_n with generators {+-a, +-b, n/2}
        # n must be even.
        # Girth check: BFS from 0 to depth 4.
        
        def check_girth_cyclic(n, a, b):
            # Generators
            c = n // 2
            S = [a, -a % n, b, -b % n, c]
            
            # Check for multi-edges/loops
            # S must have 5 distinct elements
            if len(set(S)) != 5:
                return False
            # Check for loops (0 in S)
            if 0 in S:
                return False
                
            # BFS from 0
            # dist[v] = distance from 0
            # parent[v] = previous node
            dist = [-1] * n
            parent = [-1] * n
            dist[0] = 0
            queue = [0]
            
            # We only care about cycles of length < 9.
            # Max depth to explore is 4 (since 2*4+1 = 9 is the first allowed odd cycle? 
            # Actually, if we find a cycle of length L, it must be < 9.
            # If we explore up to depth 4, we can detect cycles of length up to 2*4+1=9?
            # If we find an edge (u,v) where dist[u] + dist[v] + 1 < 9, we have a short cycle.
            # Also if we return to 0 with dist < 9.
            
            for u in queue:
                if dist[u] >= 4:
                    continue
                for s in S:
                    v = (u + s) % n
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # If v is already visited, check if this edge creates a short cycle
                        # Cycle length = dist[u] + dist[v] + 1
                        # We must ensure it's a simple cycle. 
                        # In BFS, if dist[u] == dist[v], it's a cross edge (even cycle).
                        # If dist[u] != dist[v], it's a back edge (odd cycle).
                        # In both cases, the cycle length is dist[u] + dist[v] + 1.
                        # However, we must ensure that the path from 0 to u and 0 to v are disjoint except for 0?
                        # Actually, if they share other nodes, the cycle might be shorter.
                        # But BFS gives shortest paths.
                        # If dist[u] + dist[v] + 1 < 9, we definitely have a cycle of length < 9.
                        # Is it possible that the shortest cycle is < 9 but this specific edge doesn't reveal it?
                        # No, if a short cycle exists, BFS will find an edge connecting two nodes in the BFS tree (or cross) that forms it.
                        # Specifically, for any cycle, there is an edge (u,v) such that the cycle length is dist[u] + dist[v] + 1 or similar.
                        # Actually, the standard check is: if we find an edge (u,v) where u and v are in the BFS tree, 
                        # the cycle length is dist[u] + dist[v] + 1 - 2 * dist[lca(u,v)].
                        # Since we don't track LCA, we use the conservative check:
                        # If dist[u] + dist[v] + 1 < 9, it's a short cycle.
                        # This might miss some short cycles if LCA is deep?
                        # No, if LCA is deep, dist[u] and dist[v] are larger, so sum is larger.
                        # The shortest cycle will be detected by an edge where the sum is minimized.
                        # So if we find ANY edge with dist[u] + dist[v] + 1 < 9, we have a short cycle.
                        # If we find NO such edge, do we guarantee girth >= 9?
                        # Yes, because any cycle of length L has an edge (u,v) such that L <= dist[u] + dist[v] + 1?
                        # Actually, L = dist[u] + dist[v] + 1 - 2*dist[lca].
                        # So dist[u] + dist[v] + 1 >= L.
                        # So if dist[u] + dist[v] + 1 >= 9 for all edges, then L >= 9?
                        # Wait. If L < 9, then there exists an edge in the cycle such that dist[u] + dist[v] + 1 - 2*dist[lca] < 9.
                        # This implies dist[u] + dist[v] + 1 < 9 + 2*dist[lca].
                        # This doesn't directly imply dist[u] + dist[v] + 1 < 9.
                        # Example: Cycle of length 5. u, v are neighbors in cycle.
                        # dist[u]=2, dist[v]=2. lca=0. 2+2+1=5 < 9. Detected.
                        # Cycle of length 7. u, v neighbors. dist[u]=3, dist[v]=3? No.
                        # In a BFS tree, for a cycle of length 7, the "chord" or closing edge will connect nodes with small distances.
                        # Generally, checking dist[u] + dist[v] + 1 < 9 is a sufficient condition for REJECTION.
                        # Is it necessary?
                        # If the girth is < 9, there is a cycle of length L < 9.
                        # In the BFS tree, the nodes of the cycle have distances.
                        # There must be an edge (u,v) in the cycle such that the path from 0 to u and 0 to v in the BFS tree, combined with (u,v), forms a closed walk of length <= L + 2*dist[lca]?
                        # Actually, a simpler check:
                        # Just do BFS up to depth 4.
                        # If we reach 0 again, we have a cycle.
                        # If we find an edge to a node visited at depth d, and current depth is d',
                        # and d+d'+1 < 9, we have a short cycle.
                        # This should catch all cycles of length < 9.
                        
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < 9:
                            return False
                            
            # Also check if we returned to 0
            # The loop above handles edges. If v==0, dist[v]=0.
            # cycle_len = dist[u] + 0 + 1 = dist[u] + 1.
            # If dist[u] < 8, then cycle_len < 9.
            # So it's covered.
            
            return True

        # Search n
        for n in range(428, 1250, 2):
            if time.time() > deadline - 5:
                break
            
            # Try random a, b
            # Limit tries per n
            max_tries = 50
            for _ in range(max_tries):
                a = random.randint(1, n//2 - 1)
                b = random.randint(1, n//2 - 1)
                if a == b:
                    continue
                if (a + b) % n == 0:
                    continue
                
                if check_girth_cyclic(n, a, b):
                    # Construct edges
                    c = n // 2
                    S = [a, -a % n, b, -b % n, c]
                    edges = []
                    seen = set()
                    for u in range(n):
                        for s in S:
                            v = (u + s) % n
                            if u < v:
                                if (u, v) not in seen:
                                    edges.append((u, v))
                                    seen.add((u, v))
                    return n, edges
                    
    # Fallback to LUW
    return luw_construct(k, g)
