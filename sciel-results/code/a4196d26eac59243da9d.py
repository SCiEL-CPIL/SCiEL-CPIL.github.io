"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 45.0
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def check_coloring(colors):
        """Check if a coloring is valid."""
        # Rule 1: adjacent vertices must differ
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Rule 2: for every pair of colors, induced subgraph is a forest
        color_to_verts = defaultdict(set)
        for i, c in enumerate(colors):
            color_to_verts[c].add(i)
        
        used_colors = sorted(color_to_verts.keys())
        for idx1 in range(len(used_colors)):
            for idx2 in range(idx1, len(used_colors)):
                c1, c2 = used_colors[idx1], used_colors[idx2]
                verts_set = color_to_verts[c1] | color_to_verts[c2]
                # Check for cycles using union-find
                parent = {v: v for v in verts_set}
                rank = {v: 0 for v in verts_set}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True
                
                for u, v in edges:
                    if u in verts_set and v in verts_set:
                        if not union(u, v):
                            return False
        return True
    
    # Try k from 2 upward, using randomized local search + backtracking
    for k in range(2, 8):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts with local search
        found = False
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Initialize with random assignment
            colors = [random.randint(0, k-1) for _ in range(n)]
            
            # Local search: iteratively fix violations
            improved = True
            max_iters = 100
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                if time.time() > deadline:
                    break
                
                # Check validity
                if check_coloring(colors):
                    found = True
                    break
                
                # Find a violated constraint and fix it
                # Pick a random vertex and try changing its color
                v = random.randint(0, n-1)
                old_c = colors[v]
                
                # Try each alternative color
                for new_c in range(k):
                    if new_c == old_c:
                        continue
                    # Check if this assignment is locally valid (no same-color neighbors)
                    valid_local = True
                    for u in adj[v]:
                        if colors[u] == new_c:
                            valid_local = False
                            break
                    if not valid_local:
                        continue
                    
                    colors[v] = new_c
                    # Quick check: does this improve things? Just accept and continue
                    improved = True
                    break
                
                if not improved:
                    # Stuck, try a different vertex
                    for _ in range(5):
                        v = random.randint(0, n-1)
                        old_c = colors[v]
                        for new_c in range(k):
                            if new_c == old_c:
                                continue
                            valid_local = True
                            for u in adj[v]:
                                if colors[u] == new_c:
                                    valid_local = False
                                    break
                            if not valid_local:
                                continue
                            colors[v] = new_c
                            improved = True
                            break
                        if improved:
                            break
            
            if found:
                break
        
        if found:
            return [int(c) for c in colors]
    
    # Fallback: try structured colorings
    for k in range(2, 8):
        if time.time() > deadline:
            break
        # Try various structured patterns
        for pattern_type in range(10):
            if time.time() > deadline:
                break
            if pattern_type == 0:
                colors = [i % k for i in range(n)]
            elif pattern_type == 1:
                colors = [(i // 7) % k for i in range(n)]
            elif pattern_type == 2:
                colors = [(i % 7) % k for i in range(n)]
            elif pattern_type == 3:
                colors = [(i // 5 + i % 5) % k for i in range(n)]
            elif pattern_type == 4:
                colors = [(i * 7) % k for i in range(n)]
            elif pattern_type == 5:
                colors = [(i * 13) % k for i in range(n)]
            elif pattern_type == 6:
                colors = [(i // 10) % k for i in range(n)]
            elif pattern_type == 7:
                colors = [(i % 10) % k for i in range(n)]
            elif pattern_type == 8:
                colors = [((i // 25) * 2 + (i % 25) // 12) % k for i in range(n)]
            else:
                colors = [(i * 3 + i // 10) % k for i in range(n)]
            
            if check_coloring(colors):
                return [int(c) for c in colors]
    
    # Final fallback: use all distinct colors
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
