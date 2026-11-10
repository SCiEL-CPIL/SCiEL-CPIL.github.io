"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _max_clique(n, adj):
    """Tomita's MCQ algorithm for maximum clique."""
    best = []
    best_size = 0
    deadline = time.time() + 80  # safety margin before 90s limit

    def color_sort(P_list):
        """Greedy coloring to get upper bounds. Returns (order, colors)."""
        colors = [0] * len(P_list)
        order = list(range(len(P_list)))
        # Sort by degree in the subgraph induced by P_list (descending)
        deg = [len(adj[P_list[i]] & set(P_list)) for i in range(len(P_list))]
        order.sort(key=lambda i: -deg[i])
        # Greedy coloring
        for idx in order:
            v = P_list[idx]
            used = set()
            for j in order:
                if j == idx:
                    break
                u = P_list[j]
                if u in adj[v] and colors[j] > 0:
                    used.add(colors[j])
            c = 1
            while c in used:
                c += 1
            colors[idx] = c
        # Sort by color ascending for pruning
        order.sort(key=lambda i: colors[i])
        return order, colors

    def expand(clique, P_list):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if len(P_list) == 0:
            if len(clique) > best_size:
                best_size = len(clique)
                best = list(clique)
            return
        # Color sort
        order, colors = color_sort(P_list)
        # Process in reverse order (highest color first)
        i = len(order) - 1
        while i >= 0:
            if time.time() > deadline:
                return
            v_idx = order[i]
            v = P_list[v_idx]
            # Prune: if current clique size + color <= best, stop
            if len(clique) + colors[v_idx] <= best_size:
                return
            # Add v to clique
            clique.append(v)
            # New candidate set: neighbors of v that are in P_list and appear before v in order
            new_P = [P_list[order[j]] for j in range(i) if P_list[order[j]] in adj[v]]
            expand(clique, new_P)
            clique.pop()
            # Remove v from P_list for subsequent iterations
            i -= 1

    expand([], list(range(n)))
    return best


def construct():
    n = P['n']
    adj = _build_adj(n, P['edges'])
    
    # Find maximum clique
    clique = _max_clique(n, adj)
    
    # Ensure at least 2 branches
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    
    # Build paths: for each pair of branch vertices, the path is just the edge
    from itertools import combinations
    paths = []
    for u, v in combinations(clique, 2):
        # Verify they are adjacent (they should be in a clique)
        if v in adj[u]:
            paths.append([u, v])
        else:
            # Should not happen, but fallback: find a path
            # BFS for shortest path
            from collections import deque
            visited = {u}
            parent = {u: None}
            q = deque([u])
            found = False
            while q:
                cur = q.popleft()
                if cur == v:
                    found = True
                    break
                for nb in adj[cur]:
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = cur
                        q.append(nb)
            if found:
                path = []
                cur = v
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                paths.append(path)
            else:
                # Fallback: just use the edge if it exists, otherwise skip
                if (u, v) in adj[u] or (v, u) in adj[u]:
                    paths.append([u, v])
    
    # Ensure all paths have at least 2 vertices and endpoints are correct
    # Filter out invalid paths
    valid_paths = []
    for p in paths:
        if len(p) >= 2:
            valid_paths.append(p)
    
    if len(valid_paths) == 0 and len(clique) >= 2:
        # Last resort: just use direct edges
        valid_paths = [[clique[i], clique[j]] for i, j in combinations(range(len(clique)), 2)
                       if clique[j] in adj[clique[i]]]
    
    # Convert to proper Python types
    branches = [int(x) for x in clique]
    paths_out = [[int(x) for x in p] for p in valid_paths]
    
    return {'branches': branches, 'paths': paths_out}
# EVOLVE-BLOCK-END
