import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # Moore bound calculation
    if g % 2 == 0:
        R = g // 2
        moore_bound = k * ((k - 1) ** R - 1) // (k - 2) if k > 2 else g
    else:
        R = (g - 1) // 2
        moore_bound = 1 + k * ((k - 1) ** R - 1) // (k - 2) if k > 2 else g

    # Target: 320 vertices.
    # We construct a bipartite Cayley graph on Z_2 x Z_n.
    # Total vertices 2n. We want 2n = 320 => n = 160.
    # Generators s1, s2, s3, s4 in Z_n.
    # Graph is bipartite, so only even cycles.
    # We need girth >= 10, so no cycles of length 4, 6, 8.
    
    def check_generators_bfs(n, S, max_depth):
        """
        Check if the Cayley graph on Z_n with generators S (and inverses) 
        has a cycle of length < 2*max_depth.
        Since it's bipartite (lifted to Z_2 x Z_n), we just check the Z_n part 
        for walks that sum to 0.
        Actually, the BFS on the Cayley graph of Z_n detects cycles in Z_n.
        A cycle in Z_n of length L corresponds to a cycle in the lift of length L 
        if the sum of a-coeffs is even? 
        Wait. The lift is bipartite. The base graph is the Cayley graph on Z_n.
        If the base graph has a cycle of length L, the lift has two cycles of length L 
        (one in each part) IF the cycle in the base graph lifts to a cycle.
        Actually, the bipartite lift of a graph G is the graph where vertices are (v, 0) and (v, 1), 
        and edges connect (u, 0) to (v, 1) if (u, v) in E.
        A cycle of length L in G lifts to two cycles of length L in the lift.
        So we just need the base Cayley graph on Z_n to have girth >= 10.
        """
        # BFS from 0 in the Cayley graph on Z_n
        # We want to detect if any node is reached by two different paths 
        # such that the cycle formed has length < g.
        # Or simply, if the shortest path to any node is < g/2, 
        # and there is another path of length < g/2?
        # No, we just need to ensure that the girth of the base graph is >= g.
        # Girth is the length of the shortest cycle.
        # We can do BFS from 0.
        # If we find a node u at depth d1, and then find it again at depth d2,
        # we have a cycle of length d1 + d2.
        # We want d1 + d2 >= g.
        
        # Since the graph is undirected and we use generators and their inverses,
        # the BFS tree is symmetric.
        
        visited = [-1] * n  # depth of first visit
        visited[0] = 0
        queue = deque([(0, 0)]) # (node, depth)
        
        while queue:
            u, d = queue.popleft()
            if d >= max_depth:
                continue
            for s in S:
                v1 = (u + s) % n
                v2 = (u - s) % n
                for v in (v1, v2):
                    if visited[v] == -1:
                        visited[v] = d + 1
                        queue.append((v, d + 1))
                    else:
                        # Found a cycle
                        # Cycle length is d + 1 + visited[v]
                        # We need to ensure this is >= g
                        # But wait, the "cycle" found this way might not be simple?
                        # In BFS, the first time we see a visited node, the path 
                        # u->v plus the path 0->v forms a cycle.
                        # Length = (d + 1) + visited[v].
                        # We need this to be >= g.
                        cycle_len = d + 1 + visited[v]
                        if cycle_len < g:
                            return False
        
        return True

    # Search for n starting from 160
    # We want 2*n vertices.
    
    random.seed(42)
    
    # Try n from 160 to 200
    for n in range(160, 201):
        if time.time() > deadline - 30:
            break
            
        found = False
        trials = 20000 # Increase trials since check is fast
        for trial in range(trials):
            if time.time() > deadline - 15:
                break
            # Generate 3 distinct random numbers in 1..n-1
            # Ensure no pair is inverse
            valid = True
            S = [1]
            while len(S) < 4:
                x = random.randint(1, n-1)
                if x == 1 or x == n-1:
                    continue
                conflict = False
                for y in S:
                    if x == y or x == n - y:
                        conflict = True
                        break
                if not conflict:
                    S.append(x)
            if len(S) < 4:
                continue
            
            # Check girth >= g (10)
            # max_depth for BFS: we only care about cycles < 10.
            # So we need to explore up to depth 4.
            # If a cycle of length 8 exists, it will be found at depth 4 
            # (path 0->...->u (depth 4) -> v (depth 5) where v was visited at depth 3? 
            # No, cycle length 8 means two paths of length 4 to same node.
            # BFS finds node at depth 4. If it was already visited at depth 4, 
            # cycle is 8.
            # If visited at depth 3, cycle is 7 (impossible in bipartite? 
            # Base graph is not necessarily bipartite. 
            # But we are constructing a bipartite lift. 
            # The base graph can have odd cycles. 
            # However, the lift is bipartite, so it has no odd cycles.
            # The question is: does the lift have a cycle of length < 10?
            # Yes, if the base graph has a cycle of length < 10.
            # So we just need base graph girth >= 10.
            # BFS on base graph detects girth.
            
            if check_generators_bfs(n, S, 4):
                # Construct edges for the bipartite lift
                edges = []
                for u in range(n):
                    for s in S:
                        v = (u + s) % n
                        edges.append((u, v + n))
                return 2 * n, edges

    # Fallback to LUW construction
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
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
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))
                
    return len(order), edges
