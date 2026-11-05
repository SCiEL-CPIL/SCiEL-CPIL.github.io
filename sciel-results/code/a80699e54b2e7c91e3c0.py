import time
from math import lcm, gcd
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 256

    # Precompute conflicts: conflicts[i] is a set of j such that lcm(i, j) <= 256
    conflicts = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        for j in range(i + 1, N + 1):
            if lcm(i, j) <= N:
                conflicts[i].add(j)
                conflicts[j].add(i)

    weights = {i: 1.0 / i for i in range(2, N + 1)}

    # Key insight: all numbers in [129, 256] are pairwise compatible
    # (lcm(a,b) >= max(a,b) > 128, and if both > 128, lcm > 256)
    # So we always include all of [129, 256] as a base, then decide which [2, 128] to add.
    
    base_set = set(range(129, N + 1))
    base_score = sum(weights[x] for x in base_set)

    # For each number in [2, 128], determine which base numbers it conflicts with
    # A number x in [2,128] is compatible with base if it has no conflict with any base element
    # But we need: if we add x, x must not conflict with any other added small number AND with base
    
    # Actually, the constraint is: for ALL pairs in the final set, lcm > 256.
    # Base numbers are pairwise compatible. A small number x is compatible with base
    # iff for all b in base, lcm(x, b) > 256.
    
    # Let's check which small numbers are compatible with the entire base set
    small_compatible_with_base = set()
    for x in range(2, 129):
        compatible = True
        for b in range(129, N + 1):
            if x in conflicts[b]:
                compatible = False
                break
        if compatible:
            small_compatible_with_base.add(x)

    # Now among small_compatible_with_base, we need to find max weight independent set
    # in the conflict graph restricted to these nodes.
    
    # Check if this subgraph is bipartite
    small_nodes = sorted(small_compatible_with_base)
    
    # Build subgraph
    sub_conflicts = {x: set() for x in small_nodes}
    for x in small_nodes:
        for y in small_nodes:
            if x != y and y in conflicts[x]:
                sub_conflicts[x].add(y)

    # Check bipartiteness
    def check_bipartite(nodes, conf):
        color = {}
        for start in nodes:
            if start in color:
                continue
            color[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in conf[u]:
                    if v not in color:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        return False, color
        return True, color

    is_bip, color = check_bipartite(small_nodes, sub_conflicts)

    selected_small = set()

    if is_bip and small_nodes:
        # Use min-cut for max weight independent set on bipartite graph
        # Left: color 0, Right: color 1
        num_nodes = len(small_nodes) + 2
        node_index = {x: i for i, x in enumerate(small_nodes)}
        source = num_nodes - 2
        sink = num_nodes - 1
        INF = float('inf')

        # Use adjacency list with residual capacities
        adj = [[] for _ in range(num_nodes)]
        cap = [[0.0] * num_nodes for _ in range(num_nodes)]

        def add_edge(u, v, c):
            adj[u].append(v)
            adj[v].append(u)
            cap[u][v] += c

        for x in small_nodes:
            idx = node_index[x]
            if color[x] == 0:
                add_edge(source, idx, weights[x])
            else:
                add_edge(idx, sink, weights[x])

        for x in small_nodes:
            for y in sub_conflicts[x]:
                if x < y:
                    if color[x] == 0 and color[y] == 1:
                        add_edge(node_index[x], node_index[y], INF)
                    elif color[x] == 1 and color[y] == 0:
                        add_edge(node_index[y], node_index[x], INF)

        # Edmonds-Karp
        def bfs():
            parent = [-1] * num_nodes
            parent[source] = -2
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if parent[v] == -1 and cap[u][v] > 0:
                        parent[v] = u
                        if v == sink:
                            return parent
                        queue.append(v)
            return None

        total_flow = 0.0
        while time.time() < deadline - 5:
            parent = bfs()
            if parent is None:
                break
            path_flow = INF
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, cap[u][v])
                v = u
            v = sink
            while v != source:
                u = parent[v]
                cap[u][v] -= path_flow
                cap[v][u] += path_flow
                v = u
            total_flow += path_flow

        # Find min-cut: nodes reachable from source
        visited = [False] * num_nodes
        queue = deque([source])
        visited[source] = True
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v] and cap[u][v] > 0:
                    visited[v] = True
                    queue.append(v)

        for x in small_nodes:
            idx = node_index[x]
            if color[x] == 0:
                if not visited[idx]:
                    selected_small.add(x)
            else:
                if visited[idx]:
                    selected_small.add(x)

    else:
        # Fallback: greedy + local search on the small subgraph
        current = set()
        for x in sorted(small_nodes, key=lambda v: -weights[v]):
            if all(y not in current for y in sub_conflicts[x]):
                current.add(x)
        selected_small = set(current)

        # Local search
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try add
            for x in small_nodes:
                if x not in selected_small and all(y not in selected_small for y in sub_conflicts[x]):
                    selected_small.add(x)
                    improved = True
                    break
            if improved:
                continue
            # Try swap
            best_gain = 0
            best_swap = None
            for y in list(selected_small):
                for x in small_nodes:
                    if x in selected_small:
                        continue
                    remaining = selected_small - {y}
                    if all(z not in remaining for z in sub_conflicts[x]):
                        gain = weights[x] - weights[y]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (x, y)
            if best_swap:
                x, y = best_swap
                selected_small.remove(y)
                selected_small.add(x)
                improved = True

    result = base_set | selected_small
    return list(result)
