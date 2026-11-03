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
        """Check girth using non-backtracking walks from each vertex up to depth (g-1)//2."""
        half_g = (g - 1) // 2
        for s in range(n):
            # BFS with parent tracking, but only up to depth half_g
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([(s, -1)])
            while q:
                u, par = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if v == par:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append((v, u))
                    else:
                        # Found a cycle
                        # The cycle length is dist[u] + dist[v] + 1
                        # But we need to be careful: this might not be the shortest cycle
                        # For girth check, if any cycle < g, return False
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def greedy_cage_construct(k, g):
        """Construct a k-regular graph with girth >= g using BFS tree completion with structured pairing."""
        r = (g - 1) // 2  # radius of BFS tree
        
        # Build the Moore tree explicitly
        tree_edges = []
        vertex_level = {}
        vertex_branch = {}  # level-1 ancestor vertex id
        
        vid = 0
        root = vid
        vertex_level[vid] = 0
        vertex_branch[vid] = -1
        vid += 1
        
        # Level 1
        level1_vertices = []
        for i in range(k):
            v = vid
            vertex_level[v] = 1
            vertex_branch[v] = v
            level1_vertices.append(v)
            tree_edges.append((root, v))
            vid += 1
        
        # Levels 2 to r
        current_level = level1_vertices
        for lvl in range(2, r + 1):
            next_level = []
            for parent in current_level:
                branch = vertex_branch[parent]
                for i in range(k - 1):
                    v = vid
                    vertex_level[v] = lvl
                    vertex_branch[v] = branch
                    next_level.append(v)
                    tree_edges.append((parent, v))
                    vid += 1
            current_level = next_level
        
        n_tree = vid
        level_r_vertices = current_level
        
        # Free stubs: only level-r vertices have free stubs (k-1 each)
        # Total free stubs = (k-1) * len(level_r_vertices)
        # Group by branch
        branch_groups = {}
        for v in level_r_vertices:
            b = vertex_branch[v]
            if b not in branch_groups:
                branch_groups[b] = []
            branch_groups[b].append(v)
        
        # Build stub list: each level-r vertex has (k-1) stubs
        # We need to pair stubs from different branches
        # Use a structured approach: interleave branches to ensure valid pairings
        
        all_stubs = []
        for branch in sorted(branch_groups.keys()):
            for v in branch_groups[branch]:
                for s in range(k - 1):
                    all_stubs.append((v, branch))
        
        # Shuffle within each branch group, then interleave
        for branch in branch_groups:
            branch_stubs = [(v, branch) for v in branch_groups[branch] for s in range(k-1)]
            random.shuffle(branch_stubs)
            branch_groups[branch] = branch_stubs
        
        # Interleave stubs from different branches
        interleaved = []
        max_len = max(len(v) for v in branch_groups.values())
        for i in range(max_len):
            for branch in sorted(branch_groups.keys()):
                if i < len(branch_groups[branch]):
                    interleaved.append(branch_groups[branch][i])
        
        # Now pair consecutive stubs from different branches
        # This ensures we pair stubs from different branches
        new_edges = []
        i = 0
        while i < len(interleaved):
            if i + 1 < len(interleaved):
                v1, b1 = interleaved[i]
                v2, b2 = interleaved[i + 1]
                if b1 != b2:
                    new_edges.append((v1, v2))
                    i += 2
                else:
                    # Find next stub from different branch
                    j = i + 2
                    while j < len(interleaved) and interleaved[j][1] == b1:
                        j += 1
                    if j < len(interleaved):
                        new_edges.append((v1, interleaved[j][0]))
                        # Remove the stub at position j by swapping with next
                        interleaved[j], interleaved[i+1] = interleaved[i+1], interleaved[j]
                        i += 2
                    else:
                        return None
            else:
                # Odd number of stubs shouldn't happen for valid parameters
                return None
        
        if len(new_edges) != len(all_stubs) // 2:
            return None
        
        all_edges = tree_edges + new_edges
        
        # Verify no duplicate edges
        edge_set = set()
        for u, v in all_edges:
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return None
            edge_set.add(key)
        
        return n_tree, all_edges
    
    # Try greedy construction
    result = greedy_cage_construct(k, g)
    if result is not None:
        n, edges = result
        adj = build_adj(edges, n)
        # Verify degree and girth
        degree_ok = all(len(adj[v]) == k for v in range(n))
        if degree_ok and exact_girth_check(n, adj, g):
            return n, edges
    
    # Fallback to seed
    return seed_n, seed_edges
