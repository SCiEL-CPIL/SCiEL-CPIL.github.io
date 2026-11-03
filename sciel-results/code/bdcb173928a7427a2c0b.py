import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))

    def seed_construct(k, g):
        """Lazebnik-Ustimenko-Woldar type construction."""
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

    seed_n, seed_edges = seed_construct(k, g)

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        """Check that no cycle shorter than g exists using BFS from each vertex."""
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    def greedy_cage_construct(k, g):
        """
        Construct a k-regular graph with girth >= g using BFS tree + greedy matching.
        For g = 2r+1, build the Moore-bound tree, then match leaf stubs
        across different branches to avoid short cycles.
        """
        r = (g - 1) // 2  # radius of BFS tree
        
        # Build BFS tree explicitly
        tree_edges = []
        vertex_data = {}  # vertex_id -> (level, parent, branch)
        
        vid = 0
        root = vid
        vertex_data[vid] = (0, -1, -1)
        vid += 1
        
        # Level 1
        level1_vertices = []
        for i in range(k):
            v = vid
            vertex_data[v] = (1, root, v)
            level1_vertices.append(v)
            tree_edges.append((root, v))
            vid += 1
        
        # Levels 2 to r
        current_level = level1_vertices
        for lvl in range(2, r + 1):
            next_level = []
            for parent in current_level:
                branch = vertex_data[parent][2]
                for i in range(k - 1):
                    v = vid
                    vertex_data[v] = (lvl, parent, branch)
                    next_level.append(v)
                    tree_edges.append((parent, v))
                    vid += 1
            current_level = next_level
        
        n_tree = vid
        
        # Group level-r vertices by branch (level-1 ancestor)
        branch_groups = {}
        for v, (lvl, parent, branch) in vertex_data.items():
            if lvl == r:
                if branch not in branch_groups:
                    branch_groups[branch] = []
                branch_groups[branch].append(v)
        
        # Each level-r vertex has (k-1) free stubs
        # We need to pair stubs from different branches to avoid short cycles
        all_stubs = []
        for branch, vertices in branch_groups.items():
            for v in vertices:
                for s in range(k - 1):
                    all_stubs.append((v, branch))
        
        random.shuffle(all_stubs)
        
        # Greedy matching: pair stubs from different branches
        used = [False] * len(all_stubs)
        new_edges = []
        
        for i in range(len(all_stubs)):
            if used[i]:
                continue
            v1, b1 = all_stubs[i]
            found = False
            for j in range(i + 1, len(all_stubs)):
                if used[j]:
                    continue
                v2, b2 = all_stubs[j]
                if b1 == b2:
                    continue
                # v1 and v2 are both at level r in different branches,
                # so their LCA is the root, and tree distance = 2*r = g-1.
                # Adding edge (v1,v2) creates a cycle of length 2*r + 1 = g. OK.
                new_edges.append((v1, v2))
                used[i] = True
                used[j] = True
                found = True
                break
            if not found:
                return None  # Failed to match
        
        if len(new_edges) != len(all_stubs) // 2:
            return None
        
        all_edges = tree_edges + new_edges
        return n_tree, all_edges

    # Try greedy construction
    result = greedy_cage_construct(k, g)
    if result is not None:
        n, edges = result
        adj = build_adj(edges, n)
        # Verify
        if exact_girth_check(n, adj, g):
            return n, edges

    # Fallback to seed
    return seed_n, seed_edges
