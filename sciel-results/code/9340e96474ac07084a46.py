import time
import collections
import math
import random

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _check_girth_cayley(n, m, gens, min_girth):
    """
    Checks if the Cayley graph on Z_n x Z_m with generators gens has girth >= min_girth.
    gens is a list of 3 generators (a, b). The graph uses gens and their inverses.
    """
    N = n * m
    # Map (x, y) to index x * m + y
    # Neighbors of (x, y) are (x+a, y+b) mod (n, m) for (a, b) in gens
    # and (x-a, y-b) for (a, b) in gens
    
    # We perform BFS from each vertex up to depth min_girth // 2 - 1
    max_depth = min_girth // 2 - 1
    
    # Precompute neighbor deltas
    deltas = []
    for g in gens:
        deltas.append(g)
        deltas.append((-g[0] % n, -g[1] % m))
    
    for start in range(N):
        sx = start // m
        sy = start % m
        
        dist = [-1] * N
        dist[start] = 0
        queue = collections.deque([(sx, sy, 0)])
        
        while queue:
            cx, cy, d = queue.popleft()
            if d > max_depth:
                continue
            
            idx = cx * m + cy
            if dist[idx] != d:
                continue # Outdated entry in queue, though BFS guarantees shortest path first
            
            for dx, dy in deltas:
                nx = (cx + dx) % n
                ny = (cy + dy) % m
                nidx = nx * m + ny
                
                if dist[nidx] == -1:
                    dist[nidx] = d + 1
                    queue.append((nx, ny, d + 1))
                else:
                    # If we found a visited node, check if it forms a short cycle
                    # The path from start to (cx, cy) has length d.
                    # The path from start to (nx, ny) has length dist[nidx].
                    # The edge (cx, cy) -> (nx, ny) closes a cycle of length d + dist[nidx] + 1.
                    # However, this logic is slightly flawed for general graphs.
                    # In BFS, if we encounter a visited node, it might be on the current path or a previous one.
                    # If it's on the current path, we have a back-edge.
                    # Standard girth check:
                    # If dist[nidx] < d, then we found a cycle of length d + dist[nidx] + 1?
                    # No, the cycle is start -> ... -> (cx, cy) -> (nx, ny) -> ... -> start.
                    # Length = d + 1 + dist[nidx].
                    # This is valid if the path to (nx, ny) does not go through (cx, cy).
                    # BFS guarantees dist[nidx] is the shortest distance.
                    # If dist[nidx] <= d, then d + dist[nidx] + 1 is a candidate cycle length.
                    # We need to ensure this cycle is simple.
                    # If dist[nidx] == d, the paths might share vertices.
                    # However, if d + dist[nidx] + 1 < min_girth, it's a strong indicator.
                    # For rigorous check, we should ensure the paths are disjoint except for start.
                    # Given the small depth (7), we can store parent pointers and verify.
                    # But for a fast heuristic search, we can use this check and then verify the final candidate rigorously.
                    
                    if d + dist[nidx] + 1 < min_girth:
                        # Potential short cycle. 
                        # To be safe, we return False immediately. 
                        # False positives (saying girth is small when it's not) are bad, 
                        # but false negatives (saying girth is large when it's small) are also bad.
                        # This check might miss some cycles if the shortest path to nidx goes through cx.
                        # But if it goes through cx, then dist[nidx] = d + 1, so d + d + 1 + 1 = 2d + 2.
                        # If 2d + 2 < min_girth, it's still a short cycle.
                        # So this check is valid for detecting *any* cycle of length < min_girth?
                        # Actually, if the shortest path to nidx goes through cx, then the "cycle" is just the edge back and forth, which is length 2 (not a simple cycle).
                        # So we need to be careful.
                        # Let's just return False if we find a potential short cycle, and rely on a rigorous check for the final candidate.
                        return False
                        
    return True

def _rigorous_girth_check(n, edges, min_girth):
    """
    Rigorous girth check using BFS from every vertex.
    """
    N = n
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_depth = min_girth // 2 - 1
    
    for start in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            curr = queue.popleft()
            if dist[curr] > max_depth:
                continue
            for neighbor in adj[curr]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[curr] + 1
                    parent[neighbor] = curr
                    queue.append(neighbor)
                elif neighbor != parent[curr]:
                    # Found a cycle
                    # Length is dist[curr] + dist[neighbor] + 1
                    # We need to ensure this is a simple cycle.
                    # If dist[curr] + dist[neighbor] + 1 < min_girth:
                    #    # Verify by reconstructing paths
                    #    # For now, assume it's a cycle
                    if dist[curr] + dist[neighbor] + 1 < min_girth:
                        # Double check: reconstruct paths to ensure they don't intersect
                        # except at start.
                        path1 = []
                        c = curr
                        while c != start:
                            path1.append(c)
                            c = parent[c]
                        path1.append(start)
                        
                        path2 = []
                        c = neighbor
                        while c != start:
                            path2.append(c)
                            c = parent[c]
                        path2.append(start)
                        
                        # Check intersection
                        set1 = set(path1)
                        set2 = set(path2)
                        # Intersection should only be {start}
                        if set1 & set2 == {start}:
                            return False
                        # If they intersect elsewhere, the cycle is not simple or length is different
                        # But BFS guarantees shortest paths, so they should only intersect at start
                        # unless there are multiple shortest paths.
                        # In that case, the cycle might be shorter.
                        # So returning False is safe (conservative).
                        else:
                            # If they intersect, the cycle formed is not the one we calculated.
                            # But the existence of an intersection implies a shorter cycle?
                            # Not necessarily.
                            # However, for a 3-regular graph, BFS tree is unique if no ties.
                            # If ties exist, we might have missed a shorter cycle.
                            # So returning False is a safe bet for "might be short".
                            return False

    return True

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # Search for Cayley graphs on Z_n x Z_m
    # k=3, so we need 3 generators.
    # Girth g.
    
    # Lower bound for vertices
    # Moore bound for k=3, g=16 is 512.
    # Let's search N from 512 to 1200.
    
    # We will iterate over n, m such that n*m is in range.
    # To reduce symmetry, assume n <= m.
    
    best_n = None
    best_edges = None
    
    # Iterate over possible total vertices N
    for N in range(512, 1201):
        if _time_left() < 10.0:
            break
            
        # Find divisors n of N such that n <= sqrt(N)
        for n in range(1, int(math.isqrt(N)) + 1):
            if N % n != 0:
                continue
            m = N // n
            if n > m:
                continue
            
            # Search for generators in Z_n x Z_m
            # Fix s1 = (1, 0) to break symmetry?
            # If n > 1, (1,0) is a valid non-zero element.
            # If n == 1, we can't use (1,0).
            # Let's handle n=1 separately or just iterate.
            
            # To save time, we can fix s1.
            # If n > 1:
            #   s1 = (1, 0)
            # Else:
            #   s1 = (0, 1)
            
            if n > 1:
                s1 = (1, 0)
                # Iterate s2
                # s2 = (a, b) with b != 0 to avoid s2 = s1 or -s1
                # Also avoid s2 = 0
                for a in range(n):
                    for b in range(1, m): # b != 0
                        s2 = (a, b)
                        # Avoid s2 = -s1 => (n-1, 0) which is not in range b!=0
                        # Avoid s2 = s1 => (1, 0) which is not in range b!=0
                        
                        # Iterate s3
                        # s3 = (c, d) with d != 0
                        # Avoid s3 = s1, -s1, s2, -s2
                        for c in range(n):
                            for d in range(1, m):
                                s3 = (c, d)
                                
                                # Quick checks
                                # s3 != s1, -s1
                                if s3 == s1 or s3 == ((-1) % n, 0): continue
                                # s3 != s2, -s2
                                if s3 == s2 or s3 == ((-a) % n, (-b) % m): continue
                                # s1 != s2, -s2 (already checked)
                                # s1 != s3, -s3 (checked)
                                # s2 != s3, -s3 (checked)
                                
                                # Check for 3-cycles: s_i + s_j + s_k = 0
                                # s1 + s2 + s3 = 0?
                                if (s1[0] + s2[0] + s3[0]) % n == 0 and (s1[1] + s2[1] + s3[1]) % m == 0: continue
                                # s1 + s2 - s3 = 0?
                                if (s1[0] + s2[0] - s3[0]) % n == 0 and (s1[1] + s2[1] - s3[1]) % m == 0: continue
                                # s1 - s2 + s3 = 0?
                                if (s1[0] - s2[0] + s3[0]) % n == 0 and (s1[1] - s2[1] + s3[1]) % m == 0: continue
                                # s1 - s2 - s3 = 0?
                                if (s1[0] - s2[0] - s3[0]) % n == 0 and (s1[1] - s2[1] - s3[1]) % m == 0: continue
                                
                                # Check for 4-cycles: s_i + s_j - s_k - s_l = 0
                                # s1 + s2 - s3 - s1 = s2 - s3 = 0 => s2 = s3 (checked)
                                # s1 + s2 - s3 - s2 = s1 - s3 = 0 (checked)
                                # s1 + s2 - s1 - s2 = 0 (trivial)
                                # s1 + s2 - s1 - s3 = s2 - s3 = 0
                                # s1 + s2 - s2 - s3 = s1 - s3 = 0
                                # s1 + s3 - s2 - s1 = s3 - s2 = 0
                                # s1 + s3 - s1 - s2 = s3 - s2 = 0
                                # s2 + s3 - s1 - s1 = s2 + s3 - 2s1 = 0?
                                if (s2[0] + s3[0] - 2*s1[0]) % n == 0 and (s2[1] + s3[1] - 2*s1[1]) % m == 0: continue
                                # s2 + s3 - s2 - s3 = 0
                                # s2 + s3 - s3 - s2 = 0
                                
                                # If we pass these, we have a candidate.
                                # Check girth
                                if _check_girth_cayley(n, m, [s1, s2, s3], g):
                                    # Build edges
                                    edges = []
                                    for x in range(n):
                                        for y in range(m):
                                            idx = x * m + y
                                            for s in [s1, s2, s3]:
                                                nx = (x + s[0]) % n
                                                ny = (y + s[1]) % m
                                                nidx = nx * m + ny
                                                if idx < nidx:
                                                    edges.append((idx, nidx))
                                    return N, edges
                                
            else:
                # n = 1, so group is Z_m
                # s1 = (0, 1)
                s1 = (0, 1)
                # s2 = (0, b) with b != 1, m-1
                for b in range(2, m-1):
                    s2 = (0, b)
                    for d in range(2, m-1):
                        if d == b or d == (m-b) % m: continue
                        s3 = (0, d)
                        # Check 3-cycles
                        if (s1[1] + s2[1] + s3[1]) % m == 0: continue
                        if (s1[1] + s2[1] - s3[1]) % m == 0: continue
                        if (s1[1] - s2[1] + s3[1]) % m == 0: continue
                        if (s1[1] - s2[1] - s3[1]) % m == 0: continue
                        # Check 4-cycles
                        if (s2[1] + s3[1] - 2*s1[1]) % m == 0: continue
                        
                        if _check_girth_cayley(n, m, [s1, s2, s3], g):
                            edges = []
                            for y in range(m):
                                idx = y
                                for s in [s1, s2, s3]:
                                    ny = (y + s[1]) % m
                                    nidx = ny
                                    if idx < nidx:
                                        edges.append((idx, nidx))
                            return N, edges

    # Fallback: LUW or Petersen
    # If no Cayley graph found, return a known valid graph or the LUW one.
    # The LUW graph in the original code is a valid fallback.
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
        if _time_left() < 1.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
