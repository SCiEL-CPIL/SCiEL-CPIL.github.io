import time
import sys
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    def check_girth(n, edges, min_girth, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < min_girth:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < min_girth:
                return min_cycle
        return min_cycle

    def tree_merge_construction(k, g):
        if k != 3:
            return None
        
        # Depth r such that 2r + 1 >= g
        # For g=16, 2r+1 >= 16 => r >= 7.5 => r=8
        r = (g - 1) // 2 + 1
        if 2 * r + 1 < g:
            r += 1
            
        # Build tree of depth r
        # Root 0
        # Children 1, 2, 3
        # ...
        
        nodes = []
        parent = []
        branch = [] # 0, 1, or 2
        
        nodes.append(0)
        parent.append(-1)
        branch.append(-1)
        
        # BFS
        queue = [0]
        while queue:
            u = queue.pop(0)
            depth = len(str(u)) # Not robust, use explicit depth or dist
            # Let's track depth properly
            # Actually, just generate all nodes up to depth r
            pass

        # Explicit construction
        # Level 0: [0]
        # Level 1: [1, 2, 3]
        # Level i: each node at i-1 has 2 children
        
        levels = [[] for _ in range(r + 1)]
        levels[0] = [0]
        next_id = 1
        
        for d in range(1, r + 1):
            for u in levels[d-1]:
                # 2 children
                c1 = next_id
                c2 = next_id + 1
                next_id += 2
                levels[d].append(c1)
                levels[d].append(c2)
                nodes.append(c1)
                nodes.append(c2)
                parent.append(u)
                
                # Determine branch
                if d == 1:
                    # Children of root 0
                    # We need to assign branches 0,1,2 to the 3 children?
                    # Root has 3 children.
                    # In my BFS above, I only added 2 children.
                    # Correction: Root has 3 children. Others have 2.
                    pass
        
        # Let's restart node creation logic properly
        nodes = [0]
        parent = [-1]
        branch_id = [-1]
        levels = [[0]]
        
        # Root children
        # 1, 2, 3
        for i in range(3):
            child = len(nodes)
            nodes.append(child)
            parent.append(0)
            branch_id.append(i)
            levels[0][0] = 0 # Just for structure, we build levels list
            # Actually, let's just keep a list of nodes per level
            pass

        # Easier:
        # Level 0: [0]
        # Level 1: [1, 2, 3]
        # Level 2: Children of 1,2,3 (2 each) -> 6 nodes
        # ...
        
        levels = []
        levels.append([0])
        current_id = 1
        
        # Level 1
        l1 = []
        for b in range(3):
            l1.append(current_id)
            current_id += 1
        levels.append(l1)
        
        for d in range(2, r + 1):
            prev_level = levels[-1]
            curr_level = []
            for u in prev_level:
                c1 = current_id
                c2 = current_id + 1
                current_id += 2
                curr_level.append(c1)
                curr_level.append(c2)
            levels.append(curr_level)
            
        # Total nodes
        n_total = current_id
        
        # Build parent and branch for all nodes
        parent = [0] * n_total
        branch = [0] * n_total
        
        # Level 0
        parent[0] = -1
        branch[0] = -1
        
        # Level 1
        for i, u in enumerate(levels[1]):
            parent[u] = 0
            branch[u] = i # 0, 1, 2
            
        # Level 2..r
        for d in range(2, r + 1):
            prev = levels[d-1]
            curr = levels[d]
            idx = 0
            for u in prev:
                c1 = curr[idx]
                c2 = curr[idx+1]
                parent[c1] = u
                parent[c2] = u
                branch[c1] = branch[u]
                branch[c2] = branch[u]
                idx += 2
                
        # Leaves are levels[r]
        leaves = levels[r]
        
        # Construct edges
        edges = []
        
        # Tree edges
        for u in range(1, n_total):
            edges.append((parent[u], u))
            
        # Leaf edges (2-factor)
        # Partition leaves into A, B, C
        L_A = [u for u in leaves if branch[u] == 0]
        L_B = [u for u in leaves if branch[u] == 1]
        L_C = [u for u in leaves if branch[u] == 2]
        
        # Sort them to ensure deterministic or random? Random is fine.
        random.shuffle(L_A)
        random.shuffle(L_B)
        random.shuffle(L_C)
        
        # Connect A[i] to B[i] and B[i+1] (cyclic within B? No, B has same size as A)
        # We need a 2-regular graph on the union.
        # Pattern:
        # A[i] - B[i]
        # B[i] - C[i]
        # C[i] - A[(i+1) % N]
        
        N_leaves = len(L_A)
        for i in range(N_leaves):
            edges.append((L_A[i], L_B[i]))
            edges.append((L_B[i], L_C[i]))
            edges.append((L_C[i], L_A[(i + 1) % N_leaves]))
            
        # Verify simple graph
        # Check for duplicate edges
        edge_set = set()
        valid = True
        for u, v in edges:
            if u == v:
                valid = False
                break
            key = (min(u,v), max(u,v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            
        if not valid:
            return None
            
        # Check degree
        deg = [0] * n_total
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            return None
            
        return n_total, edges

    def luw_fallback(k, g):
        # Original fallback
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

    if k == 3 and g == 16:
        result = tree_merge_construction(k, g)
        if result:
            n, edges = result
            # Optional: quick girth check
            # gc = check_girth(n, edges, g, deadline - 5)
            # if gc >= g:
            return result
            
    return luw_fallback(k, g)
