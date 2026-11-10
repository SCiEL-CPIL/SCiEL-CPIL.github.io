"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def check_bicolored_forest(colors, c1, c2):
        """Check if subgraph induced by colors c1 and c2 is a forest using union-find."""
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        
        for u, v in edges:
            if (colors[u] == c1 or colors[u] == c2) and (colors[v] == c1 or colors[v] == c2):
                if not union(u, v):
                    return False
        return True
    
    def is_valid_full(colors):
        """Full validation: proper coloring + bicolored forest for all pairs."""
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        distinct = sorted(set(colors))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if not check_bicolored_forest(colors, distinct[i], distinct[j]):
                    return False
        return True
    
    # Try to find a valid 3-coloring
    # Strategy: use backtracking with good heuristics
    
    def solve_kcolor(k):
        """Solve for a valid k-coloring, return colors or None."""
        # Vertex ordering: sort by degree descending
        vertex_order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
        colors = [None] * n
        
        def get_forbidden(v):
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] is not None:
                    forbidden.add(colors[nb])
            return forbidden
        
        def check_local(v, c):
            """Quick local check: does assigning c to v create a bicolored cycle?"""
            for c2 in range(k):
                if c2 == c:
                    continue
                # Count vertices and edges in the {c, c2} subgraph
                vert_count = 0
                edge_count = 0
                for i in range(n):
                    if colors[i] in (c, c2):
                        vert_count += 1
                if vert_count == 0:
                    continue
                for u, v2 in edges:
                    if colors[u] in (c, c2) and colors[v2] in (c, c2):
                        edge_count += 1
                if edge_count >= vert_count:
                    return False
            return True
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return is_valid_full(colors)
            
            v = vertex_order[idx]
            forbidden = get_forbidden(v)
            
            for c in range(k):
                if c in forbidden:
                    continue
                
                colors[v] = c
                
                if check_local(v, c):
                    if backtrack(idx + 1):
                        return True
                
                colors[v] = None
            
            return False
        
        if backtrack(0):
            return colors
        return None
    
    best_result = None
    
    # First try some simple patterns for 3 colors
    patterns = []
    patterns.append([i % 3 for i in range(n)])
    patterns.append([(i + i // 7) % 3 for i in range(n)])
    patterns.append([(i * 2) % 3 for i in range(n)])
    patterns.append([(i // 5) % 3 for i in range(n)])
    patterns.append([(i % 5 + i // 5) % 3 for i in range(n)])
    patterns.append([(i + (i // 10)) % 3 for i in range(n)])
    
    for pat in patterns:
        if time.time() > deadline:
            break
        proper = True
        for u, v in edges:
            if pat[u] == pat[v]:
                proper = False
                break
        if proper and is_valid_full(pat):
            best_result = pat
            break
    
    if best_result is None:
        # Try backtracking for 3 colors
        result = solve_kcolor(3)
        if result is not None:
            best_result = result
    
    if best_result is None:
        # Try 4 colors
        result = solve_kcolor(4)
        if result is not None:
            best_result = result
    
    if best_result is None:
        # Fallback: use all distinct colors
        best_result = list(range(n))
    
    best_result = [int(c) for c in best_result]
    return best_result

# EVOLVE-BLOCK-END
