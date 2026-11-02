"""Construct a 3-regular graph of girth 16 (cage problem n(3,16)).

Strategy: Use the Lazebnik-Ustimenko-Woldar (LUW) family with parameters tuned for
girth 16. The LUW graph CD(d, q) is q-regular bipartite with girth at least d+5 (odd d)
or d+4 (even d). For girth g=16 and degree k=3, we need q=3 and appropriate d.

However, the LUW family gives large graphs. A better approach for small 3-regular
graphs of high girth is to use a product construction or a known explicit construction.

Actually, let's think about this more carefully. The task says the smallest known graph
has 936 vertices. The LUW construction with q=3 and suitable d may give a graph with
more than 936 vertices. Let's compute:

For LUW CD(d, q): the number of vertices is 2 * q^d (points and lines).
With q=3, d=11: 2 * 3^11 = 2 * 177147 = 354294 vertices. That's way too many.

So the LUW family is not competitive here. We need a different approach.

A better strategy: Use a random 3-regular graph and verify girth. For n vertices,
a random 3-regular graph has girth approximately (2/3) * log_{2}(n) or similar.
To get girth 16, we need n such that the expected number of short cycles is small.

The number of cycles of length 2i in a random 3-regular graph on n vertices is
approximately (3/2)^{2i} / (2i). For girth 16, we need no cycles of length 2..14.
The expected number of cycles of length 2i is (3/2)^{2i} / (2i).
For i=7 (length 14): (3/2)^14 / 14 ≈ 28.7 / 14 ≈ 2.05. That's still > 1.
For i=8 (length 16): (3/2)^16 / 16 ≈ 65.7 / 16 ≈ 4.1. 

So random graphs won't easily give girth 16 with a small n. We need a smarter approach.

Alternative: Use a known explicit construction. The 936-vertex graph from Exoo et al.
(2026) is the current best. Without access to its exact edge list, let's try a different
algebraic construction.

Another approach: Use a circulant-like construction or a Cayley graph. But 3-regular
Cayley graphs of girth 16 are hard to construct explicitly.

Let's try a hybrid approach: start with a candidate n (e.g., 936 or a bit larger),
generate random 3-regular graphs, and check girth. If we find one with girth >= 16,
return it. We need to be efficient in the girth check.

For girth checking: BFS from each vertex up to depth 7 (since girth 16 means no cycles
of length < 16, so we need to check for cycles of length 2..15, i.e., BFS depth 7 from
each vertex looking for back-edges that form cycles of length <= 15). Actually, for a
bipartite graph, girth is even. But our graph isn't necessarily bipartite.

Wait, 3-regular graphs of girth 16: the girth is 16, which is even. So we're looking for
no cycles of length 2, 4, 6, 8, 10, 12, 14. 

Let me try a different strategy: use the known fact that for (3,16)-cages, n >= 512 (Moore
bound) and the best known is 936. Let's try to construct a graph with n close to 936
using a greedy or randomized approach with girth verification.

Given the 120s time limit, let's try:
1. Start with n = 936.
2. Generate random 3-regular graphs on n vertices.
3. Check girth >= 16.
4. If found, return. Otherwise, try n = 940, 944, etc.

The girth check can be done via BFS from each vertex, tracking the first visit depth.
A cycle of length L exists if during BFS from vertex v, we find an edge to a vertex u
where depth[u] + depth[v] + 1 = L and u is not the parent of v in the BFS tree, or
more precisely, we detect a non-tree edge that closes a cycle.

Actually, a simpler check: for each vertex, do BFS up to depth 7. If we encounter a
vertex that was already visited at depth d' < current_depth, and the edge doesn't go
back to the parent, then we have a cycle of length current_depth + d' + 1 (if going
through different paths) or we need to be more careful.

Let me use a cleaner approach: for each edge (u,v), check if there's a path from u to v
of length < 15 that doesn't use edge (u,v). If so, there's a cycle of length < 16.
This is equivalent to: remove edge (u,v), check if dist(u,v) < 15. But that's O(m * (n+m))
which is too slow.

Better: BFS from each vertex, track parent and depth. When we see an edge to an already
visited node (not the parent), we found a cycle. The cycle length is depth[u] + depth[v] + 1
if they're in different branches, but we need to be careful.

Actually, the standard way: do BFS from each vertex s. Maintain dist[s][v] and parent[s][v].
When exploring edge (u,v) where v is already visited and v != parent[s][u], we have a cycle.
The cycle length is dist[s][u] + dist[s][v] + 1 if the paths from s to u and s to v are
disjoint except at s. But they might share a common ancestor.

This is getting complicated. Let me use a simpler but correct method: for each vertex s,
BFS and track the first time we reach each vertex. If we find an edge (u,v) where both
are already visited and neither is the parent of the other in the BFS tree, we have a
cycle. The length is at most dist[u] + dist[v] + 1, but the actual cycle length depends
on the LCA.

For simplicity and correctness, let me just check: during BFS from s, if we encounter an
edge (u,v) where v is already visited and v is not the parent of u, then there's a cycle
of length <= dist[u] + dist[v] + 1. If this is < 16, we fail.

Wait, that's not quite right either. Let me just use a well-known approach:

For girth >= g, we need no cycles of length < g. A cycle of length L exists iff there's
a vertex s and two distinct paths from s of lengths a and b with a + b = L and the paths
only meet at s (and possibly at the other endpoint). 

The cleanest check: for each vertex s, do BFS. For each vertex v reached at depth d, and
each neighbor w of v, if w is already at depth d' in the BFS tree and w != parent[v] and
parent[w] != v, then we have a cycle of length d + d' + 1 (if the paths are disjoint
except at s) or shorter. To be safe, if d + d' + 1 < g, we have a cycle shorter than g.

But the paths might not be disjoint. However, if d + d' + 1 < g, then even in the worst
case, the cycle is shorter than g. Actually no, if the paths share a prefix, the cycle
could be shorter, which is still bad. So the check d + d' + 1 < g is a sufficient
condition for a short cycle. But it's not necessary.

For a correct check, I'll use: for each vertex s, BFS. When we find a back-edge (u,v)
where v is visited and v != parent[u], compute the cycle length as the distance from u
to v in the BFS tree plus 1. If this is < g, fail.

Let me implement this properly.
"""

import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # Moore bound for (k, g) cage
    if g % 2 == 0:
        half_g = g // 2
        moore = 1 + k * sum(k ** i for i in range(half_g - 1))
    else:
        half_g = (g - 1) // 2
        moore = k * sum(k ** i for i in range(half_g))
    
    # Try to find a 3-regular graph with girth >= g
    # Start from a reasonable n
    n_candidates = [936, 940, 944, 948, 952, 960, 970, 980, 1000, 1020, 1040, 1080, 1120, 1160, 1200]
    
    def check_girth(n, adj, min_girth):
        """Check if graph has girth >= min_girth. Returns True if girth >= min_girth."""
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u] and parent[v] != u:
                        # Found a cycle: length is dist[u] + dist[v] + 1
                        # But this might overestimate if paths share prefix
                        # Conservative check
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    def generate_random_3regular(n):
        """Generate a random 3-regular graph on n vertices using configuration model."""
        # Create stubs
        stubs = []
        for i in range(n):
            stubs.extend([i, i, i])
        random.shuffle(stubs)
        
        # Pair up stubs
        edges = []
        for i in range(0, n * 3, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                return None  # Loop, retry
            if (u, v) in [(a, b) for a, b in edges] or (v, u) in [(a, b) for a, b in edges]:
                return None  # Multiple edge, retry
            edges.append((u, v))
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None
        
        return adj
    
    def generate_3regular_better(n):
        """Better random 3-regular graph generation."""
        for _ in range(100):
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            random.shuffle(stubs)
            
            edges = []
            valid = True
            seen = set()
            for i in range(0, n * 3, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in seen:
                    valid = False
                    break
                seen.add(key)
                edges.append((u, v))
            
            if not valid:
                continue
            
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            if all(len(adj[i]) == 3 for i in range(n)):
                return adj
        
        return None
    
    for n in n_candidates:
        if time.time() > deadline:
            break
        
        # Try multiple random graphs
        attempts = 0
        max_attempts = 5000 if n <= 1000 else 2000
        while attempts < max_attempts and time.time() < deadline:
            adj = generate_3regular_better(n)
            if adj is None:
                attempts += 1
                continue
            
            if check_girth(n, adj, g):
                # Convert to edge list
                edge_list = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edge_list.append((u, v))
                return n, edge_list
            
            attempts += 1
    
    # Fallback: LUW construction (will be large but valid)
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
