import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Fallback: LUW Construction (from original code)
    def _lazebnik_construct(k, g):
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

    # New Strategy: Random Regular Graph + Girth Repair
    if k == 5 and g == 9:
        # Try to find a smaller graph than LUW (6250)
        # Best known is 1116. Moore bound 428.
        # We will try to construct one in the range 800-1200.
        
        def generate_random_regular(n, k):
            # Configuration model
            while True:
                stubs = [i for i in range(n) for _ in range(k)]
                random.shuffle(stubs)
                edges = set()
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if (min(u, v), max(u, v)) in edges:
                        valid = False
                        break
                    edges.add((min(u, v), max(u, v)))
                if valid:
                    return list(edges)

        def find_short_cycle(edges, n, max_len):
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # BFS from each node
            for start in range(n):
                visited = {start: -1} # node: parent
                queue = deque([(start, -1, 0)])
                while queue:
                    node, parent, depth = queue.popleft()
                    if depth >= max_len:
                        continue
                    for nei in adj[node]:
                        if nei == parent:
                            continue
                        if nei not in visited:
                            visited[nei] = node
                            queue.append((nei, node, depth + 1))
                        else:
                            # Found a cycle
                            # We need to calculate the length
                            # This simple BFS doesn't easily give the cycle length
                            # unless we track depth of the first visit
                            pass
            
            # Better: Check all pairs of nodes at distance < max_len
            # Actually, for girth 9, we just need to ensure no cycle of length 5,6,7,8.
            # We can use a specific check for small cycles.
            
            # Let's use a simpler check: 
            # For each node, do a BFS up to depth 4.
            # If we see a node already in the current path, it's a cycle.
            # But we need the length.
            
            # Efficient check for girth < g:
            # Count closed walks of length L < g.
            # If any non-trivial closed walk exists, girth < g.
            # For a regular graph, we can compute the number of closed walks of length L
            # by raising the adjacency matrix to L.
            # But n=1000, L=8. Matrix multiplication is O(n^3 log L) or O(n^3).
            # n^3 = 10^9. Too slow.
            
            # Alternative: Random sampling of walks?
            # Or just check for triangles, 4-cycles, 5-cycles, etc.
            
            # Let's implement a specific search for cycles of length L.
            # For L=5:
            # Pick 3 nodes, check if they form a path of length 2 that can be closed?
            
            # Simpler: 
            # We will just do a BFS from each node and record the first time we visit a node.
            # If we encounter a node that is an ancestor in the BFS tree, we have a cycle.
            # We need to be careful to not count backtracking.
            
            for start in range(n):
                # BFS
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
                        elif v != parent[u]:
                            # Found a non-tree edge
                            # Cycle length = dist[u] + dist[v] - 2*dist[lca] + 1
                            # We need LCA.
                            # To keep it fast, we only check if dist[u] + dist[v] + 1 < g?
                            # No, the cycle length is at most dist[u] + dist[v] + 1.
                            # If dist[u] + dist[v] + 1 >= g, it might still be a short cycle if LCA is deep.
                            # But if dist[u] + dist[v] + 1 < g, then LCA must be very deep, 
                            # which implies the nodes are close.
                            
                            # Let's just compute the cycle length properly for potential short cycles.
                            # We only care if length < 9.
                            # So we only need to check if there is a cycle of length 5, 6, 7, 8.
                            
                            # Heuristic: If dist[u] + dist[v] + 1 < 9, check LCA.
                            # Actually, if dist[u] + dist[v] + 1 >= 9, the cycle could still be short
                            # if the LCA is close to u and v?
                            # No, dist(u, v) <= dist[u] + dist[v].
                            # Cycle length = dist(u, v) + 1.
                            # So if dist[u] + dist[v] + 1 >= 9, it's possible that dist(u, v) is small?
                            # No, dist(u, v) is the shortest path.
                            # The cycle formed by u-v edge and tree paths has length dist[u] + dist[v] - 2*dist[lca] + 1.
                            # This length is at least dist(u, v) + 1.
                            # And dist(u, v) <= dist[u] + dist[v].
                            # So the cycle length is <= dist[u] + dist[v] + 1.
                            # If dist[u] + dist[v] + 1 < 9, we MUST check.
                            # If dist[u] + dist[v] + 1 >= 9, the cycle formed by this non-tree edge
                            # has length at most dist[u] + dist[v] + 1, which is >= 9.
                            # BUT, there might be a shorter cycle involving this edge?
                            # No, the tree path is the unique path in the tree.
                            # Any cycle using the non-tree edge (u,v) must use the tree path between u and v.
                            # So the length is fixed: dist_tree(u, v) + 1.
                            # And dist_tree(u, v) is the distance in the BFS tree.
                            # So we just need to compute dist_tree(u, v).
                            
                            if dist[u] + dist[v] + 1 < g:
                                # Compute LCA
                                # Move u and v up to same depth
                                x, y = u, v
                                if dist[x] < dist[y]:
                                    x, y = y, x
                                diff = dist[x] - dist[y]
                                for _ in range(diff):
                                    x = parent[x]
                                while x != y:
                                    x = parent[x]
                                    y = parent[y]
                                lca = x
                                cycle_len = dist[u] + dist[v] - 2 * dist[lca] + 1
                                if cycle_len < g:
                                    return cycle_len
                # End BFS for start
            return None

        def repair_girth(edges, n, k, g, time_limit):
            start_t = time.time()
            edges_set = set((min(u, v), max(u, v)) for u, v in edges)
            
            while time.time() - start_t < time_limit:
                # Check girth
                bad_cycle = find_short_cycle(list(edges_set), n, g)
                if bad_cycle is None:
                    return list(edges_set)
                
                # Find a specific short cycle to break
                # We need the nodes of the cycle.
                # find_short_cycle currently just returns length.
                # Let's modify it to return the cycle nodes.
                # For now, let's just pick a random edge and try to swap it.
                # This is less targeted but simpler.
                
                # Pick a random edge
                e1 = random.choice(list(edges_set))
                u1, v1 = e1
                
                # Pick a random other edge
                e2 = random.choice(list(edges_set))
                u2, v2 = e2
                
                # Try swap (u1, v1) and (u2, v2) -> (u1, u2) and (v1, v2)
                # Or (u1, v2) and (v1, u2)
                # We need to ensure no loops and no duplicates.
                
                candidates = [
                    [(u1, u2), (v1, v2)],
                    [(u1, v2), (v1, u2)]
                ]
                
                swapped = False
                for new_e1, new_e2 in candidates:
                    n1, n2 = min(new_e1), max(new_e1)
                    m1, m2 = min(new_e2), max(new_e2)
                    
                    if n1 == n2 or m1 == m2:
                        continue
                    if (n1, n2) == (m1, m2):
                        continue
                    if (n1, n2) in edges_set or (m1, m2) in edges_set:
                        continue
                    if (n1, n2) == (min(u1, v1), max(u1, v1)):
                        continue
                    if (m1, m2) == (min(u2, v2), max(u2, v2)):
                        continue
                        
                    # Perform swap
                    edges_set.remove((min(u1, v1), max(u1, v1)))
                    edges_set.remove((min(u2, v2), max(u2, v2)))
                    edges_set.add((n1, n2))
                    edges_set.add((m1, m2))
                    swapped = True
                    break
                
                if not swapped:
                    # Try a different pair
                    continue
                    
            return None

        # Try different n values
        for n in range(800, 1200, 50):
            if time.time() > deadline:
                break
            
            # Generate
            edges = generate_random_regular(n, k)
            if edges is None:
                continue
                
            # Repair
            result = repair_girth(edges, n, k, g, 5.0) # 5s per n
            if result is not None:
                return n, result

    # Fallback to LUW
    return _lazebnik_construct(k, g)
