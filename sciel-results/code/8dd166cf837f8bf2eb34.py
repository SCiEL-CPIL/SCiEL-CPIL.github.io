import random
from collections import deque
import time
import math
import numpy as np

def construct(k, g):
    # For k=4, g=10, use circulant graph search
    if k != 4 or g != 10:
        return _fallback_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 115  # Leave 35s margin
    
    n = 320
    
    # Search for a circulant graph C_n({s1, s2}) with girth >= g
    # Vertices are 0..n-1
    # Edges are (i, i+s1) and (i, i+s2)
    
    # To avoid duplicates, we can restrict s1, s2 such that 1 <= s1 < s2 <= n//2
    # Also gcd(s1, s2, n) must be 1 for connectivity (though not strictly required for girth check, 
    # but a disconnected graph with components having girth >= 10 is also valid? 
    # The task says "a 4-regular graph". Usually cages are connected. 
    # If disconnected, it's not a cage in the standard sense, but technically satisfies "no cycle shorter than g".
    # However, the Moore bound applies to connected graphs. The lower bound 243 is for connected.
    # The known 320 vertex graph is connected. So we require connectivity.)
    
    for s1 in range(1, n // 2):
        if time.time() > deadline:
            break
        for s2 in range(s1 + 1, n // 2):
            if time.time() > deadline:
                break
            
            # Check connectivity: gcd(s1, s2, n) == 1
            if math.gcd(math.gcd(s1, s2), n) != 1:
                continue
            
            # Check girth
            if _check_circulant_girth(n, s1, s2, g):
                # Construct edges
                edges = []
                for i in range(n):
                    edges.append((i, (i + s1) % n))
                    edges.append((i, (i + s2) % n))
                return n, edges
    
    # Fallback: LUW graph
    return _fallback_construct(k, g)

def _check_circulant_girth(n, s1, s2, min_girth):
    """
    Check if the circulant graph C_n({s1, s2}) has girth >= min_girth.
    Returns True if no cycle shorter than min_girth is found.
    """
    # We can do BFS from each vertex.
    # But for circulants, the structure is symmetric.
    # A cycle is a sequence of steps in {s1, s2, -s1, -s2} summing to 0 mod n.
    # We can use a BFS on the state space of (current_pos, last_step) to find shortest non-backtracking walks.
    # However, a simpler BFS on the graph itself works.
    
    # Optimization: We only need to check from a few vertices? No, girth is global.
    # But if we find a short cycle anywhere, it fails.
    
    # Let's use a BFS from each vertex, but stop early if we find a cycle < min_girth.
    # To speed up, we can use the fact that the graph is vertex-transitive.
    # If a cycle exists, it can be rotated to start at 0.
    # So we only need to find cycles passing through 0?
    # No, a cycle might not pass through 0.
    # However, due to vertex transitivity, if there is a cycle of length L, there is a cycle of length L passing through 0.
    # So we only need to check for cycles passing through 0.
    
    # Let's verify this logic.
    # Graph is vertex transitive. Girth is the minimum cycle length.
    # If min cycle length is L, there is a cycle C of length L.
    # There is an automorphism mapping 0 to any vertex in C.
    # So there is a cycle of length L containing 0.
    # Therefore, we only need to check if there is a cycle of length < min_girth containing 0.
    
    # We can do a BFS from 0, but we must avoid backtracking to ensure we find simple cycles correctly?
    # Standard BFS finds shortest paths. A cycle through 0 corresponds to two distinct shortest paths from 0 to some vertex u meeting?
    # Or simpler: BFS from 0, keeping track of the path?
    # Actually, finding the shortest cycle through a specific vertex v in a graph can be done by:
    # 1. Remove v.
    # 2. For each pair of neighbors (u, w) of v, find shortest path between u and w in G-v.
    # 3. Cycle length = 1 (to u) + dist(u, w) + 1 (to w).
    # This is efficient.
    
    # Neighbors of 0 are s1, -s1, s2, -s2.
    # Let N = [s1 % n, (n-s1) % n, s2 % n, (n-s2) % n].
    # We need to check all pairs (u, w) from N, u != w.
    # There are 4*3 = 12 pairs.
    # For each pair, run BFS in G-v (where v=0) to find dist(u, w).
    # If dist(u, w) + 2 < min_girth, return False.
    
    # G-v is the graph with vertex 0 removed.
    # Since it's a circulant, removing 0 breaks symmetry, but we can still BFS.
    # n=320 is small. 12 BFS runs on 319 vertices is very fast.
    
    neighbors = [s1 % n, (n - s1) % n, s2 % n, (n - s2) % n]
    # Unique neighbors (s1 != s2, s1 != -s2 mod n, etc. assumed by s1 < s2 <= n/2)
    # Just in case, make them unique.
    neighbors = list(set(neighbors))
    
    # Build adjacency for G-0
    # We can just use the circulant definition, skipping 0.
    
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            u = neighbors[i]
            w = neighbors[j]
            
            # BFS from u to w in G-0
            dist = _bfs_dist_circulant(n, s1, s2, u, w, exclude=0)
            
            if dist != -1:
                cycle_len = dist + 2
                if cycle_len < min_girth:
                    return False
    
    return True

def _bfs_dist_circulant(n, s1, s2, start, target, exclude):
    """
    Find shortest path distance from start to target in the circulant graph C_n({s1, s2})
    with vertex 'exclude' removed.
    """
    if start == target:
        return 0
    if start == exclude or target == exclude:
        return -1
        
    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        cur = queue.popleft()
        d = visited[cur]
        
        # Neighbors in circulant
        # (cur + s1) % n, (cur - s1) % n, (cur + s2) % n, (cur - s2) % n
        candidates = [
            (cur + s1) % n,
            (cur - s1) % n,
            (cur + s2) % n,
            (cur - s2) % n
        ]
        
        for nb in candidates:
            if nb == exclude:
                continue
            if nb not in visited:
                if nb == target:
                    return d + 1
                visited[nb] = d + 1
                queue.append(nb)
                
    return -1

def _fallback_construct(k, g):
    """Fallback to LUW-style construction."""
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
