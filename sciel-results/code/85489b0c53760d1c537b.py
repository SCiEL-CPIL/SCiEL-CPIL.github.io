import math
import time
import random
import sys
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = 110  # 10s buffer before 120s kill
    
    # 1. Try Circulant Search (Fast, specific family)
    # Moore bound for k=3, g=16 is 510.
    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1
        
    # Circulants for k=3 are C(n; {a, n-a, n/2})
    # We search up to a small limit because they are unlikely to have high girth
    max_circ_n = 1000
    for n in range(moore_bound, max_circ_n + 1, 2):
        if time.time() - start_time > deadline:
            break
        h = n // 2
        for a in range(1, h):
            if math.gcd(h, a) != 1:
                continue
            if _check_girth_circulant(n, a, g):
                edges = []
                for u in range(n):
                    for s in (a, n - a, h):
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges

    # 2. Try Random 3-Regular Graph Search
    # This is the main improvement. Random 3-regular graphs have high girth.
    # We start from the Moore bound and go up.
    max_random_n = 2000
    # We want to find the SMALLEST n.
    # So we iterate n from moore_bound upwards.
    for n in range(moore_bound, max_random_n + 1, 2):
        if time.time() - start_time > deadline:
            break
        
        # Try a few random instances for this n
        # The probability of a random 3-regular graph having girth >= 16
        # is high for n ~ 1000. For n ~ 512, it might be low.
        # We try up to 50 instances per n.
        for attempt in range(50):
            if time.time() - start_time > deadline:
                break
            
            edges = _generate_random_3_reg(n)
            if edges is None:
                continue
            
            # Check connectivity
            if not _is_connected(n, edges):
                continue
            
            # Check girth
            if _check_girth_general(n, edges, g):
                return n, edges

    # 3. Fallback to LUW
    return _luw_fallback(k, g)

def _check_girth_circulant(n, a, g):
    """Check if C(n; {a, -a, n/2}) has girth >= g."""
    h = n // 2
    S = [a, n - a, h]
    # Non-backtracking walks
    # State: (current_node, prev_node)
    # But for circulant, we can use distance sets.
    # R_d = set of nodes reachable in d steps without backtracking.
    # For k=3, backtracking is determined by the previous step.
    # Simpler: BFS from 0, tracking (node, parent) to prevent immediate backtracking.
    # Since it's a tree-like structure for girth check, we can just do BFS up to depth g//2 - 1.
    # If we revisit a node at depth < g//2, we have a cycle.
    # Actually, for girth >= g, the ball of radius (g-1)/2 must be a tree.
    # For even g, radius g/2 - 1.
    
    max_depth = g // 2 - 1
    visited = {0: -1} # node: depth
    # Queue: (node, parent, depth)
    q = deque([(0, -1, 0)])
    
    while q:
        u, parent, d = q.popleft()
        if d == max_depth:
            continue
        for s in S:
            v = (u + s) % n
            # Prevent backtracking to parent
            if v == parent:
                continue
            if v in visited:
                # Cycle found
                # Cycle length is d + 1 + visited[v]?
                # No, if v is visited at depth dv, and we are at depth d,
                # the cycle length is d + 1 + dv only if the path from v to root
                # and u to root are disjoint except at root?
                # In a BFS tree, if we see a cross edge or back edge,
                # we need to check the distance.
                # For simple girth check in regular graphs,
                # if we encounter a visited node v where v != parent,
                # we have a cycle.
                # The length of the cycle is at least d + 1 + 1?
                # Actually, if v was visited at depth dv, the path from 0 to v has length dv.
                # The path from 0 to u has length d.
                # The edge (u,v) closes a cycle.
                # The length is d + 1 + dv? No.
                # The cycle goes 0 -> ... -> u -> v -> ... -> 0?
                # The path 0->v is length dv.
                # The path 0->u is length d.
                # The edge u-v connects them.
                # The cycle length is d + 1 + dv?
                # Only if the paths 0->u and 0->v are disjoint.
                # In a tree, they are. If they share a prefix, the cycle is smaller.
                # However, for girth >= g, we just need to ensure NO cycle < g.
                # If we find ANY visited node v (v != parent), we have a cycle.
                # We must ensure the cycle length is >= g.
                # The shortest cycle through u and v is d + 1 + dv?
                # No, if LCA(u,v) is w, cycle length is dist(u,w) + dist(v,w) + 1.
                # dist(u,w) = d - dist(w,w)? No.
                # dist(u, root) = d. dist(v, root) = dv.
                # dist(u, v) via root is d + dv.
                # The cycle is d + dv + 1?
                # This logic holds if the graph is a tree up to that point.
                # If we are doing BFS, the first time we see a visited node,
                # the cycle length is d + dv + 1?
                # Actually, if v is visited, it means there is a path of length dv from 0 to v.
                # And a path of length d from 0 to u.
                # And edge (u,v).
                # The cycle length is at least min(d, dv) + 1 + |d - dv|?
                # To be safe, if we see a visited node, we should calculate the exact cycle length.
                # But for a quick check, if d + dv + 1 < g, it's invalid.
                # Is it possible that d + dv + 1 >= g but there is a shorter cycle?
                # Yes, if the paths overlap.
                # However, for high girth graphs, the ball of radius (g-1)/2 is a tree.
                # So if we don't find a cycle in depth max_depth, the girth is >= 2*max_depth + 1?
                # For even g, we check up to g/2 - 1.
                # If the ball of radius g/2 - 1 is a tree, then girth >= 2*(g/2 - 1) + 1 = g - 1?
                # Wait. If radius R is a tree, girth >= 2R + 1.
                # We want girth >= g.
                # So we need 2R + 1 >= g => R >= (g-1)/2.
                # For g=16, R >= 7.5 => R=8?
                # If R=7, girth >= 15. We need 16.
                # So we must check up to R=8?
                # If R=8 is a tree, girth >= 17.
                # So for even g, we need to check if ball of radius g/2 - 1 is a tree?
                # If ball of radius 7 is a tree, girth >= 15.
                # It could have a cycle of length 16.
                # A cycle of length 16 would have a node at distance 8 from the start?
                # Yes.
                # So we must check if there is a cycle of length < 16.
                # This means we must check if any node is reached by two different paths of length < 8?
                # Or if a node at depth d is connected to a node at depth dv such that d+dv+1 < 16?
                # To be rigorous, we should just run BFS and track distances.
                # If we find an edge (u,v) such that dist[u] + dist[v] + 1 < g, it's a bad cycle?
                # No, dist[u] + dist[v] + 1 is the length of the cycle formed by the two tree paths and the edge.
                # This is the SHORTEST cycle through that edge.
                # So if dist[u] + dist[v] + 1 < g, we fail.
                
                dv = visited[v]
                cycle_len = d + dv + 1
                if cycle_len < g:
                    return False
            else:
                visited[v] = d
                q.append((v, u, d + 1))
                
    return True

def _generate_random_3_reg(n):
    """Generate a random simple 3-regular graph on n vertices."""
    if n % 2 != 0 or n < 4:
        return None
    
    # Configuration Model
    # 3n stubs
    stubs = [i for i in range(n) for _ in range(3)]
    random.shuffle(stubs)
    
    # Pair up stubs
    # We need to avoid loops and multi-edges.
    # Simple rejection sampling:
    # 1. Pair randomly.
    # 2. Check for loops/multi-edges.
    # 3. If bad, re-pair the problematic stubs or restart.
    # For n ~ 1000, restarts are cheap.
    
    for _ in range(100): # Max 100 restarts
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        
        edges = []
        valid = True
        edge_set = set()
        
        for i in range(0, 3 * n, 2):
            u = stubs[i]
            v = stubs[i + 1]
            
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                valid = False
                break
            edge_set.add((u, v))
            edges.append((u, v))
            
        if valid:
            return edges
            
    return None

def _is_connected(n, edges):
    """Check if graph is connected."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    visited = [False] * n
    q = deque([0])
    visited[0] = True
    count = 1
    
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                q.append(v)
                
    return count == n

def _check_girth_general(n, edges, g):
    """Check if graph has girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex? Too slow if n is large.
    # But n is small (~1000).
    # We can just BFS from 0? No, we need to check ALL cycles.
    # However, if the graph is random, the girth is determined by the shortest cycle.
    # We can run BFS from each vertex, but optimize by stopping if we find a short cycle.
    # Or, simpler: For each vertex u, do BFS up to depth g//2 - 1.
    # If we find a cycle of length < g, return False.
    
    max_depth = g // 2 - 1
    # For g=16, max_depth = 7.
    # If ball of radius 7 is a tree, girth >= 15.
    # We need to check for cycles of length 16?
    # If we find a cycle of length 16, it's OK.
    # We only care about cycles < 16.
    # So we need to check if there is a cycle of length <= 15.
    # This is equivalent to checking if the ball of radius 7 contains a cycle?
    # No. A cycle of length 16 has diameter 8.
    # So nodes at distance 8 from each other.
    # If we check balls of radius 7, we might miss a 16-cycle?
    # No, a 16-cycle has chords? No.
    # In a 16-cycle, the distance between opposite nodes is 8.
    # If we start BFS from a node u on the cycle, we will reach the opposite node v at depth 8.
    # But we stop at depth 7.
    # So we won't see the closure of the cycle at depth 7?
    # Wait. If we stop at depth 7, we don't check edges between nodes at depth 7?
    # We check edges from nodes at depth < 7 to their neighbors.
    # If a neighbor is already visited, we have a cycle.
    # For a 16-cycle, starting from u:
    # Depth 0: u
    # Depth 1: u-1, u+1
    # ...
    # Depth 7: u-7, u+7
    # Neighbors of u-7: u-8, u-6.
    # u-6 is visited (depth 6). Cycle length: 7 + 6 + 1 = 14?
    # Wait. Path u->u-1->...->u-7 (len 7).
    # Path u->u+1->...->u-6? No.
    # u-7 is connected to u-8 and u-6.
    # u-6 is at depth 6 (via u-1...u-6).
    # So edge (u-7, u-6) creates a cycle.
    # Length: dist(u, u-7) + dist(u, u-6) + 1?
    # dist(u, u-7) = 7.
    # dist(u, u-6) = 6.
    # Cycle length = 7 + 6 + 1 = 14.
    # This is < 16.
    # So a 16-cycle WILL be detected as a shorter cycle if we just look at the tree structure?
    # No, the "cycle" detected is the one formed by the BFS tree paths.
    # In a simple cycle, the BFS tree will have "cross edges" or "back edges".
    # For a 16-cycle, starting at 0:
    # 0 connected to 1, 15.
    # 1 connected to 2, 0.
    # ...
    # 7 connected to 8, 6.
    # 6 is visited at depth 6.
    # Edge (7,6) connects depth 7 and depth 6.
    # Cycle length = 7 + 6 + 1 = 14.
    # This is correct. The cycle 0-1-...-7-6-...-0 has length 14?
    # No. 0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-0 is length 16.
    # The edge (7,6) is NOT an edge in the 16-cycle.
    # In a 16-cycle, 7 is connected to 6 and 8.
    # 6 is connected to 5 and 7.
    # So (7,6) IS an edge.
    # The cycle formed by 0->1->...->7->6->5->...->0?
    # Path 0->1->...->7 is length 7.
    # Path 0->15->14->...->6 is length 6?
    # 0-15 (1), 15-14 (2), ..., 15-6 (9 steps? 15,14,13,12,11,10,9,8,7,6).
    # Distance from 0 to 6 via 15 is 6?
    # 0-15-14-13-12-11-10-9-8-7-6. Length 10.
    # Distance from 0 to 6 via 1 is 6.
    # So dist(0,6) = 6.
    # dist(0,7) = 7.
    # Edge (7,6).
    # Cycle length = 7 + 6 + 1 = 14.
    # This implies there is a 14-cycle?
    # No. The cycle is 0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-0.
    # The edge (7,6) is part of this cycle.
    # The "cycle" detected by the BFS shortcut is 0-1-2-3-4-5-6-7-6-5-4-3-2-1-0?
    # No, the BFS tree path to 6 is 0-1-2-3-4-5-6.
    # The BFS tree path to 7 is 0-1-2-3-4-5-6-7? No, 7 is neighbor of 6.
    # If we visit 6 first (depth 6), then 7 is visited from 6 (depth 7).
    # Then we check neighbors of 7: 6 and 8.
    # 6 is parent. 8 is new.
    # So no cycle detected at 7?
    # Wait. If 7 is visited from 6, then 6 is parent.
    # We check 8.
    # Then 8 is visited from 7.
    # ...
    # 15 is visited from 14.
    # Neighbors of 15: 14 and 0.
    # 0 is visited at depth 0.
    # Edge (15,0).
    # Cycle length = dist(15) + dist(0) + 1 = 15 + 0 + 1 = 16.
    # 16 >= 16. OK.
    
    # So, if we do BFS and check all edges (u,v) where v is visited and v != parent,
    # and calculate cycle_len = dist[u] + dist[v] + 1,
    # and check if cycle_len < g,
    # then we correctly identify if girth < g.
    
    for start in range(n):
        visited = {start: 0}
        q = deque([(start, -1, 0)])
        while q:
            u, parent, d = q.popleft()
            for v in adj[u]:
                if v == parent:
                    continue
                if v in visited:
                    dv = visited[v]
                    cycle_len = d + dv + 1
                    if cycle_len < g:
                        return False
                else:
                    visited[v] = d
                    q.append((v, u, d + 1))
                    
    return True

def _luw_fallback(k, g):
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
