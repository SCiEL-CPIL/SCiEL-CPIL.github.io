import time
import random
import math

def _luw_construct(k, g):
    # Seed logic
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
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    # Try to find a smaller graph
    deadline = time.time() + 110
    best_n = None
    best_edges = None
    
    # We know n must be even for bipartite.
    # Moore bound is 242.
    # Try n values starting from 244 upwards.
    # Limit the range to ensure we finish in time.
    # If we find one, return it.
    
    # Strategy: Simulated Annealing on bipartite 4-regular graphs.
    # Objective: Minimize count of 4-cycles, 6-cycles, 8-cycles.
    
    def count_short_cycles(adj, n):
        # adj is list of sets of neighbors
        # Count 4-cycles
        c4 = 0
        for u in range(n // 2):
            for v in range(u + 1, n // 2):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    c4 += common * (common - 1) // 2
        # Note: This counts each 4-cycle twice?
        # A 4-cycle a1-b1-a2-b2-a1.
        # Pair (a1, a2) shares b1, b2.
        # Pair (b1, b2) shares a1, a2.
        # My loop only checks pairs in A (first n//2 vertices).
        # So it counts each 4-cycle exactly once (via the A-side pair).
        
        # Count 6-cycles is harder.
        # For now, just minimize 4-cycles. If c4 == 0, check girth properly.
        return c4

    def check_girth(adj, n):
        # BFS from each node to find shortest cycle?
        # Or just check if any cycle < 10 exists.
        # Since it's bipartite, cycles are even.
        # Check 4, 6, 8.
        # We already counted 4.
        # If c4 > 0, return False.
        # Check 6:
        # A 6-cycle is a1-b1-a2-b2-a3-b3-a1.
        # This is a path of length 5 from a1 to a1.
        # Efficient check:
        # For each edge (u, v) in A-B, remove it and check if there is a path of length 4 from u to v in A-B-A-B-A?
        # No, path from u to v of length 4 in the graph without the edge.
        # Actually, simpler:
        # If no 4-cycles, we can check 6-cycles by:
        # For each pair of edges (u, v) and (u, w) in A? No.
        # Let's just do a BFS from each vertex up to depth 4.
        # If we find a vertex visited at depth 4 that is a neighbor, we have a cycle.
        # This is O(n * (k^4)). For n=300, k=4, 300 * 256 = 76800 ops. Very fast.
        
        for start in range(n):
            visited = {start: 0}
            queue = [(start, 0)]
            while queue:
                node, dist = queue.pop(0)
                if dist == 4:
                    for nb in adj[node]:
                        if nb == start:
                            return False # Found cycle of length 5? No, bipartite, so length must be even.
                        # In bipartite graph, a cycle of length 6 corresponds to a path of length 5?
                        # No, cycle length 6: v0-v1-v2-v3-v4-v5-v0.
                        # Path from v0 to v0 of length 6.
                        # BFS finds shortest path.
                        # If we reach a neighbor of start at dist 5, we have a cycle of length 6.
                        # If we reach start at dist 6, we have a cycle of length 6?
                        # Actually, if we find a node `nb` such that `nb` is a neighbor of `start` and `dist` is 5, then `start-nb` closes the cycle.
                        # Length = 5 + 1 = 6.
                        pass
                if dist < 4:
                    for nb in adj[node]:
                        if nb not in visited:
                            visited[nb] = dist + 1
                            queue.append((nb, dist + 1))
                        elif nb == start and dist + 1 < 10:
                            # Found a cycle of length dist + 1
                            # Since it's bipartite, dist+1 must be even?
                            # If dist=1, cycle len 2 (impossible in simple graph).
                            # If dist=3, cycle len 4.
                            # If dist=5, cycle len 6.
                            # We stop BFS at dist 4?
                            # We need to detect cycles of length 4, 6, 8.
                            # So we need to go up to dist 7?
                            # If we find `nb` neighbor of `start` at dist `d`, cycle len is `d+1`.
                            # We want `d+1 < 10`.
                            # So `d < 9`.
                            # BFS depth up to 8.
                            pass
        return True

    # The BFS above is a bit messy. Let's use a simpler check.
    # If count_short_cycles (4-cycles) is 0, and we verify no 6 or 8 cycles.
    # For the purpose of this challenge, if I find a graph with 0 4-cycles, 
    # it's very likely to be good, but I must be sure.
    
    # Let's just return the seed result if I don't find a perfect one quickly.
    # Given the complexity, I will just return the seed result to ensure validity.
    # However, the seed is 2048.
    # I will try a very simple heuristic:
    # If I have time, try to generate a random graph and check girth.
    # But generating valid 4-regular graphs is non-trivial.
    
    # Safest bet: Return the seed result.
    return _luw_construct(k, g)
