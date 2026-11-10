"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    import time
    deadline = time.time() + 80
    n = P['n']
    es = set()
    for e in P['edges']:
        es.add((e[0], e[1]))
        es.add((e[1], e[0]))
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    # Maximum clique via branch and bound with greedy coloring
    best_clique = []
    
    def greedy_color(order):
        """Color vertices in order, return color count and color classes"""
        colors = {}
        for v in order:
            used = set()
            for u in order:
                if u == v:
                    break
                if u in colors and u in adj[v]:
                    used.add(colors[u])
            c = 0
            while c in used:
                c += 1
            colors[v] = c
        return max(colors.values()) + 1 if colors else 0, colors
    
    def expand(candidates, current):
        nonlocal best_clique
        if time.time() > deadline:
            return
        if not candidates:
            if len(current) > len(best_clique):
                best_clique = list(current)
            return
        
        # Greedy coloring upper bound
        colored, color_map = greedy_color(candidates)
        if len(current) + colored <= len(best_clique):
            return
        
        # Try vertices in reverse color order (higher colors first)
        for v in reversed(candidates):
            if time.time() > deadline:
                return
            new_candidates = [u for u in candidates if u in adj[v]]
            new_current = current + [v]
            expand(new_candidates, new_current)
            candidates = [u for u in candidates if u != v]
            # Update bound
            colored, color_map = greedy_color(candidates)
            if len(current) + colored <= len(best_clique):
                return
    
    expand(list(range(n)), [])
    
    branches = best_clique if len(best_clique) >= 2 else [P['edges'][0][0], P['edges'][0][1]]
    
    # For each pair, find a path avoiding all other branch vertices
    branch_set = set(branches)
    paths = {}
    
    from collections import deque
    
    def bfs_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices"""
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                # Reconstruct path
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for w in adj[u]:
                if w not in visited and w not in forbidden:
                    visited.add(w)
                    parent[w] = u
                    queue.append(w)
        return None
    
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            u, v = branches[i], branches[j]
            forbidden = branch_set - {u, v}
            path = bfs_path(u, v, forbidden)
            if path is None:
                # Fallback: use direct edge if exists
                if v in adj[u]:
                    path = [u, v]
                else:
                    # Should not happen for clique, but fallback
                    path = [u, v]
            paths[(u, v)] = path
    
    path_list = []
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            path_list.append(list(paths[(branches[i], branches[j])]))
    
    return {'branches': list(branches), 'paths': path_list}
# EVOLVE-BLOCK-END
