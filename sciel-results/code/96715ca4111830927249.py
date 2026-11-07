"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    # Map vertex to its index in the order
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
        
    # We will try k = 3, 4, 5
    for k in range(3, 6):
        colors = [-1] * n
        
        # Helper to check if assigning color c to v is valid
        def is_valid(v, c):
            # 1. Adjacent vertices must differ
            for u in adj[v]:
                if colors[u] == c:
                    return False
            
            # 2. Bicolored forest constraint
            # For each other color c', check if adding v with color c creates a cycle
            # in the subgraph induced by {c, c'}
            for c_prime in range(k):
                if c_prime == c:
                    continue
                
                # Find neighbors of v that have color c or c_prime
                neighbors_in_subgraph = []
                for u in adj[v]:
                    if colors[u] == c or colors[u] == c_prime:
                        neighbors_in_subgraph.append(u)
                
                if len(neighbors_in_subgraph) < 2:
                    continue
                
                # Check if any two neighbors in neighbors_in_subgraph are connected
                # in the graph induced by colors {c, c_prime} excluding v
                # We can do BFS from the first neighbor
                start = neighbors_in_subgraph[0]
                target_set = set(neighbors_in_subgraph[1:])
                
                visited = {start}
                queue = [start]
                while queue:
                    curr = queue.pop(0)
                    for nb in adj[curr]:
                        # Only traverse if nb is not v and has color c or c_prime
                        if nb == v:
                            continue
                        if colors[nb] == c or colors[nb] == c_prime:
                            if nb in target_set:
                                return False
                            if nb not in visited:
                                visited.add(nb)
                                queue.append(nb)
                                
            return True

        # Backtracking
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(k):
                if is_valid(v, c):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False

        if backtrack(0):
            # Final validation
            if _validate(colors, edges):
                return [int(c) for c in colors]
                
    # Fallback: use more colors
    # With 5 colors, it should work. If not, try 6.
    for k in range(6, 10):
        colors = [-1] * n
        def is_valid(v, c):
            for u in adj[v]:
                if colors[u] == c:
                    return False
            for c_prime in range(k):
                if c_prime == c:
                    continue
                neighbors_in_subgraph = []
                for u in adj[v]:
                    if colors[u] == c or colors[u] == c_prime:
                        neighbors_in_subgraph.append(u)
                if len(neighbors_in_subgraph) < 2:
                    continue
                start = neighbors_in_subgraph[0]
                target_set = set(neighbors_in_subgraph[1:])
                visited = {start}
                queue = [start]
                while queue:
                    curr = queue.pop(0)
                    for nb in adj[curr]:
                        if nb == v:
                            continue
                        if colors[nb] == c or colors[nb] == c_prime:
                            if nb in target_set:
                                return False
                            if nb not in visited:
                                visited.add(nb)
                                queue.append(nb)
            return True

        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(k):
                if is_valid(v, c):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False

        if backtrack(0):
            if _validate(colors, edges):
                return [int(c) for c in colors]

    # Last resort
    return list(range(n))

def _validate(colors, edges):
    n = len(colors)
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    if not _check_bicolored_forest(colors, edges):
        return False
    return True

def _check_bicolored_forest(colors, edges):
    from collections import defaultdict
    n = len(colors)
    color_map = defaultdict(list)
    for v in range(n):
        color_map[colors[v]].append(v)
    
    colors_used = set(colors)
    for i, c1 in enumerate(colors_used):
        for c2 in colors_used:
            if c1 >= c2:
                continue
            verts = [v for v in range(n) if colors[v] in (c1, c2)]
            if len(verts) < 3:
                continue
            vert_set = set(verts)
            parent = {}
            def find(x):
                if x not in parent:
                    parent[x] = x
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return False
                parent[px] = py
                return True
            for u in verts:
                find(u)
            for u, v in edges:
                if u in vert_set and v in vert_set:
                    if not union(u, v):
                        return False
    return True
# EVOLVE-BLOCK-END
