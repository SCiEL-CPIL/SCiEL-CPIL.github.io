"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Fast bicolored forest check: only consider edges between the two specific colors
    def check_bicolor_forest_fast(colors, c1, c2):
        """Check if subgraph induced by colors c1 and c2 is a forest."""
        # Get vertices with these colors
        verts = set()
        for i in range(n):
            if colors[i] == c1 or colors[i] == c2:
                verts.add(i)
        
        if len(verts) <= 1:
            return True
        
        # Union-Find on just these vertices
        parent = {v: v for v in verts}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        
        # Only check edges where both endpoints have color c1 or c2
        for u, v in edges:
            cu, cv = colors[u], colors[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
        return True
    
    def is_valid_coloring(colors):
        """Check proper coloring and all bicolored subgraphs are forests."""
        # Check proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        
        # Get distinct colors used
        used_colors = list(set(colors))
        
        # Check each pair
        for i in range(len(used_colors)):
            for j in range(i + 1, len(used_colors)):
                if not check_bicolor_forest_fast(colors, used_colors[i], used_colors[j]):
                    return False
        return True
    
    def count_distinct(colors):
        return len(set(colors))
    
    best_result = None
    best_count = n + 1
    
    # Strategy 1: Try structured modular patterns
    # The graph is 5x10, so try patterns based on row and column
    for k in range(3, 10):
        if time.time() > deadline:
            break
        
        # Try pattern: (row * a + col * b) % k for various a, b
        for a in range(k):
            for b in range(k):
                if a == b:
                    continue
                colors = []
                for i in range(n):
                    row = i // 10
                    col = i % 10
                    colors.append((row * a + col * b) % k)
                
                # Check if proper
                proper = True
                for u, v in edges:
                    if colors[u] == colors[v]:
                        proper = False
                        break
                if not proper:
                    continue
                
                # Check bicolored forest
                if is_valid_coloring(colors):
                    cnt = count_distinct(colors)
                    if cnt < best_count:
                        best_count = cnt
                        best_result = colors[:]
        
        if best_count <= k:
            break
    
    # Strategy 2: Try diagonal patterns
    for k in range(3, 10):
        if time.time() > deadline:
            break
        for offset in range(k):
            colors = [(i + offset) % k for i in range(n)]
            proper = True
            for u, v in edges:
                if colors[u] == colors[v]:
                    proper = False
                    break
            if not proper and is_valid_coloring(colors):
                cnt = count_distinct(colors)
                if cnt < best_count:
                    best_count = cnt
                    best_result = colors[:]
    
    # Strategy 3: Random restarts with local search
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 7 + 13)
        
        # Try with small number of colors
        for max_k in [3, 4, 5]:
            if time.time() > deadline:
                break
            
            # Greedy random coloring
            order = list(range(n))
            random.shuffle(order)
            colors = [-1] * n
            
            valid = True
            for v in order:
                used = set(colors[u] for u in adj[v] if colors[u] >= 0)
                avail = [c for c in range(max_k) if c not in used]
                if not avail:
                    valid = False
                    break
                colors[v] = random.choice(avail)
            
            if not valid:
                continue
            
            if not is_valid_coloring(colors):
                continue
            
            cnt = count_distinct(colors)
            if cnt < best_count:
                best_count = cnt
                best_result = colors[:]
            
            # Local search to try to reduce colors
            current = colors[:]
            cur_cnt = cnt
            for _iter in range(100):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(n):
                    old_c = current[i]
                    for new_c in range(max_k):
                        if new_c == old_c:
                            continue
                        # Check proper
                        proper = True
                        for v in adj[i]:
                            if current[v] == new_c:
                                proper = False
                                break
                        if not proper:
                            continue
                        
                        current[i] = new_c
                        if is_valid_coloring(current):
                            new_cnt = count_distinct(current)
                            if new_cnt < cur_cnt:
                                cur_cnt = new_cnt
                                improved = True
                                break
                            else:
                                current[i] = old_c
                        else:
                            current[i] = old_c
                    if improved:
                        break
                if not improved:
                    break
            
            if cur_cnt < best_count:
                best_count = cur_cnt
                best_result = current[:]
    
    # Fallback
    if best_result is None:
        best_result = list(range(n))
        best_count = n
    
    return [int(c) for c in best_result]
# EVOLVE-BLOCK-END
