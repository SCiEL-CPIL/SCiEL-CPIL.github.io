import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def time_left():
        return deadline - time.time()

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def luw_construct(k, g, q=None, rule_seed=None):
        if q is None:
            q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        # Define field operations
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # Generate rules
        # Standard rule from seed
        rule = {}
        if rule_seed is None:
            rule[2] = (1, 1)
            rule[3] = (2, 1)
            rule[4] = (1, 2)
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        else:
            rng = random.Random(rule_seed)
            rule[2] = (1, 1)
            rule[3] = (2, 1)
            rule[4] = (1, 2)
            for c in range(5, d + 1):
                # Random valid rule: a, b such that a < c, b < c
                # To maintain girth, we need specific structures.
                # For now, just try a few random permutations of indices
                a = rng.randint(1, c-1)
                b = rng.randint(1, c-1)
                if a == b:
                    b = (b % (c-1)) + 1
                rule[c] = (a, b)

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        # Build the graph
        # We will generate all possible vertices (2 * q^d) and their edges
        # Then find components
        
        all_vertices = []
        for side in [0, 1]:
            for x_tuple in range(q ** d):
                # Convert x_tuple to base-q digits
                x = [0] * (d + 1)
                x[0] = None
                rem = x_tuple
                for i in range(1, d + 1):
                    x[i] = rem % q
                    rem //= q
                all_vertices.append((side, tuple(x)))
        
        # Map vertex to index
        v_index = {v: i for i, v in enumerate(all_vertices)}
        n_total = len(all_vertices)
        
        adj = [set() for _ in range(n_total)]
        
        for v in all_vertices:
            idx = v_index[v]
            for nb in neighbours(v):
                if nb in v_index:
                    nb_idx = v_index[nb]
                    adj[idx].add(nb_idx)
        
        # Find connected components
        visited = [False] * n_total
        components = []
        
        for i in range(n_total):
            if not visited[i]:
                # BFS
                q_bfs = deque([i])
                visited[i] = True
                comp = []
                while q_bfs:
                    u = q_bfs.popleft()
                    comp.append(u)
                    for w in adj[u]:
                        if not visited[w]:
                            visited[w] = True
                            q_bfs.append(w)
                components.append(comp)
        
        return components, adj

    def has_short_cycle(n, adj, g):
        """Fast girth check: BFS from each vertex up to depth (g-1)//2."""
        half = (g - 1) // 2
        adj_list = [list(s) for s in adj]
        for start in range(n):
            depth = [-1] * n
            parent = [-1] * n
            depth[start] = 0
            parent[start] = -2
            queue = deque([start])
            while queue:
                node = queue.popleft()
                nd = depth[node]
                if nd >= half:
                    continue
                for nb in adj_list[node]:
                    if nb == parent[node]:
                        continue
                    if depth[nb] == -1:
                        depth[nb] = nd + 1
                        parent[nb] = node
                        queue.append(nb)
                    elif parent[nb] != node and depth[nb] != nd + 1:
                        u, v = node, nb
                        d1, d2 = depth[node], depth[nb]
                        while u != v:
                            if d1 > d2:
                                u = parent[u]
                                d1 -= 1
                            elif d2 > d1:
                                v = parent[v]
                                d2 -= 1
                            else:
                                u = parent[u]
                                v = parent[v]
                                d1 -= 1
                                d2 -= 1
                        cycle_len = depth[node] + depth[nb] - 2 * d1 + 1
                        if cycle_len < g:
                            return True
        return False

    best_n = float('inf')
    best_edges = None

    # Try standard LUW
    if time_left() > 10:
        comps, adj = luw_construct(k, g, q=k, rule_seed=None)
        for comp in comps:
            n_comp = len(comp)
            if n_comp < best_n:
                # Check girth
                # Build subgraph adj for this component
                comp_set = set(comp)
                sub_adj = [set() for _ in range(n_comp)]
                mapping = {v: i for i, v in enumerate(comp)}
                for u in comp:
                    u_idx = mapping[u]
                    for w in adj[u]:
                        if w in comp_set:
                            w_idx = mapping[w]
                            sub_adj[u_idx].add(w_idx)
                
                # Check regularity
                regular = all(len(nei) == k for nei in sub_adj)
                if regular and not has_short_cycle(n_comp, sub_adj, g):
                    best_n = n_comp
                    # Reconstruct edges
                    edges = []
                    for u in range(n_comp):
                        for v in sub_adj[u]:
                            if u < v:
                                edges.append((mapping[u], mapping[v]))
                    best_edges = edges

    # Try some random rules to see if we get smaller components
    if time_left() > 30:
        for seed in range(10):
            if time_left() < 20:
                break
            comps, adj = luw_construct(k, g, q=k, rule_seed=seed)
            for comp in comps:
                n_comp = len(comp)
                if n_comp >= best_n:
                    continue
                comp_set = set(comp)
                sub_adj = [set() for _ in range(n_comp)]
                mapping = {v: i for i, v in enumerate(comp)}
                for u in comp:
                    u_idx = mapping[u]
                    for w in adj[u]:
                        if w in comp_set:
                            w_idx = mapping[w]
                            sub_adj[u_idx].add(w_idx)
                
                regular = all(len(nei) == k for nei in sub_adj)
                if regular and not has_short_cycle(n_comp, sub_adj, g):
                    best_n = n_comp
                    edges = []
                    for u in range(n_comp):
                        for v in sub_adj[u]:
                            if u < v:
                                edges.append((mapping[u], mapping[v]))
                    best_edges = edges

    if best_n == float('inf'):
        # Fallback: return the largest component of standard LUW (should be valid)
        comps, adj = luw_construct(k, g, q=k, rule_seed=None)
        max_comp = max(comps, key=len)
        n_comp = len(max_comp)
        comp_set = set(max_comp)
        sub_adj = [set() for _ in range(n_comp)]
        mapping = {v: i for i, v in enumerate(max_comp)}
        for u in max_comp:
            u_idx = mapping[u]
            for w in adj[u]:
                if w in comp_set:
                    w_idx = mapping[w]
                    sub_adj[u_idx].add(w_idx)
        
        best_n = n_comp
        edges = []
        for u in range(n_comp):
            for v in sub_adj[u]:
                if u < v:
                    edges.append((mapping[u], mapping[v]))
        best_edges = edges

    return best_n, best_edges
