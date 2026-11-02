"""Combined approach: LUW family with deadline guard and circulant search."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def check_deadline():
        return time.time() > deadline

    # --- Utility: validate graph (simple, k-regular, girth >= g) ---
    def validate(n, edges, k, g):
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return False
            if (min(u,v), max(u,v)) in seen:
                return False
            seen.add((min(u,v), max(u,v)))
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        # Check girth via BFS from each vertex
        for s in range(n):
            depth = [-1] * n
            parent = [-1] * n
            depth[s] = 0
            q2 = deque([s])
            while q2:
                u = q2.popleft()
                if depth[u] >= g - 1:
                    continue
                for v in adj[u]:
                    if depth[v] == -1:
                        depth[v] = depth[u] + 1
                        parent[v] = u
                        q2.append(v)
                    else:
                        # Non-tree edge (u,v). Cycle length = depth[u] + depth[v] + 1
                        # But only if v is not parent of u and u is not parent of v
                        if v != parent[u] and u != parent[v]:
                            clen = depth[u] + depth[v] + 1
                            if clen < g:
                                return False
        return True

    # --- Family 1: Lazebnik-Ustimenko-Woldar (from seed) ---
    def luw_family():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Get LUW result as fallback
    n_luw, e_luw = luw_family()
    best_n = n_luw
    best_e = e_luw

    # --- Search for Circulant Graphs ---
    # Only feasible for small k and specific g
    if k == 5 and g == 9:
        # Moore bound for k=5, g=9 is 426.
        # Try n from 426 to 2000
        # n must be even for k=5 (odd degree)
        for n in range(426, 1500, 2):
            if check_deadline():
                break
            if n % 2 != 0:
                continue
            # Symmetric set S of size 5
            # S = {n/2, a, n-a, b, n-b}
            # a, b in 1..n/2-1, a != b, a != n-b (always true if a,b < n/2)
            # We try small a, b
            found = False
            for a in range(1, min(20, n//2)):
                if check_deadline():
                    break
                for b in range(a+1, min(20, n//2)):
                    if a == b or a == n-b: continue
                    # S = {n//2, a, n-a, b, n-b}
                    S = {n//2, a, n-a, b, n-b}
                    if len(S) != 5:
                        continue
                    # Check girth via BFS from 0
                    # If any neighbor of 0 is reachable in < 8 steps (not using direct edge)
                    # We check if dist(0, v) in G - {(0,v)} < 8 for any v in S
                    # Simpler: BFS from 0, if we see a neighbor of 0 at depth d < 8 via a path not starting with that edge?
                    # Actually, just check if there is a cycle of length < 9.
                    # In a circulant, if there is a cycle, there is one through 0.
                    # Cycle through 0: 0 -> ... -> 0.
                    # This corresponds to a sum of elements in S (with signs) equal to 0 mod n,
                    # with length L < 9, not using the trivial 0+0...
                    # Let's just do BFS from 0 up to depth 7.
                    # If we reach a vertex v that is in S, and the path length is L,
                    # and the last step is not the direct edge 0-v?
                    # Actually, if we reach v in S at depth d < 8, we have a cycle of length d+1.
                    # BUT, we reach v at depth 1 via the direct edge.
                    # So we need to know if there is a *second* path of length d < 8.
                    # This is equivalent to: is the shortest path from 0 to v in G - {(0,v)} less than 8?
                    # We can just run BFS from 0 in the graph, but mark the first visit.
                    # If we visit v (in S) again at depth d < 8, we have a cycle.
                    # Or, simply:
                    # For each v in S, check if v is reachable from 0 in <= 7 steps in G - {(0,v)}.
                    # This is 5 BFS runs per candidate.
                    # 5 * 1500 * (cost of BFS).
                    # Cost of BFS on n=1500, deg=5 is small.
                    # But 5 * 1500 * 1500 = 11M ops. OK.
                    
                    valid = True
                    # To speed up, we can just check one v first? No, all must be valid.
                    # Actually, if ANY v has a short path, it's invalid.
                    # So we check if ANY v in S has dist < 8 in G - {(0,v)}.
                    
                    # Optimization: Just run BFS from 0 in G, and track if we can reach v in S in < 8 steps 
                    # without using the edge (0,v) as the FIRST step?
                    # No, the cycle can enter 0 at the end.
                    # The cycle is 0 -> u1 -> ... -> uk -> 0.
                    # u1 in S, uk in S.
                    # Path 0 -> u1 -> ... -> uk has length k-1.
                    # Then edge uk -> 0.
                    # So we need a path of length L-1 from 0 to uk, where uk in S, L < 9.
                    # And the path must not use the edge (0, uk) at the end? No, the edge (uk, 0) is the last step.
                    # The path 0 -> ... -> uk must not use the edge (0, uk) anywhere?
                    # If it uses (0, uk) at the start, then uk is reached in 1 step.
                    # Then cycle is 0 -> uk -> 0, length 2.
                    # So we must avoid the direct edge.
                    
                    # Let's just do:
                    # For each v in S:
                    #   BFS from 0 in G, but ignore the edge (0, v) initially?
                    #   No, just check if dist(0, v) in G - {(0, v)} < 8.
                    
                    # We will do this check.
                    for v in S:
                        # BFS from 0, excluding edge (0, v)
                        depth = [-1] * n
                        depth[0] = 0
                        q_bfs = deque([0])
                        found_short = False
                        while q_bfs:
                            u = q_bfs.popleft()
                            if depth[u] >= 7: # We want dist < 8, so depth <= 7
                                continue
                            for w in ( (u+x)%n for x in S ):
                                if u == 0 and w == v:
                                    continue # Exclude direct edge
                                if depth[w] == -1:
                                    depth[w] = depth[u] + 1
                                    if w == v:
                                        found_short = True
                                        break
                                    q_bfs.append(w)
                            if found_short:
                                break
                        if found_short:
                            valid = False
                            break
                        if check_deadline():
                            break
                    if check_deadline():
                        break
                    if valid:
                        # Construct edges
                        edges = []
                        for i in range(n):
                            for s in S:
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                        # Validate to be safe
                        if validate(n, edges, k, g):
                            best_n = n
                            best_e = edges
                            found = True
                            break
                if found:
                    break
            if found:
                break

    return best_n, best_e
# EVOLVE-BLOCK-END
