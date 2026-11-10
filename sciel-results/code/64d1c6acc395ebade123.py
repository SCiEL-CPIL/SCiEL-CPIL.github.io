import time
import random

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 60  # 60s limit, keep margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def is_valid_coloring(colors):
        # Check proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
                
        # Check bicolored cycles using Union-Find for each pair of colors
        # Get distinct colors used
        used_colors = set(colors)
        if len(used_colors) < 2:
            return True # Trivially no cycles if <2 colors? No, proper coloring prevents same color adj.
                        # If 1 color, no edges. If 2 colors, check cycles.
                        
        # We only need to check pairs of colors that are actually present
        color_list = list(used_colors)
        for i in range(len(color_list)):
            for j in range(i+1, len(color_list)):
                c1, c2 = color_list[i], color_list[j]
                # Union Find
                parent = list(range(n))
                rank = [0] * n
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                    
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False # Cycle found
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True
                    
                for u, v in edges:
                    if colors[u] in (c1, c2) and colors[v] in (c1, c2):
                        if colors[u] != colors[v]: # Proper coloring ensures this, but safe
                            if not union(u, v):
                                return False # Bicolored cycle found
        return True

    # Try deterministic patterns first
    patterns = [
        lambda i: i % 3,
        lambda i: (i + i // 7) % 3,
        lambda i: (i * 2) % 3,
        lambda i: (i + i // 7 + i // 25) % 3,
    ]
    
    for p in patterns:
        colors = [p(i) for i in range(n)]
        if is_valid_coloring(colors):
            return [int(c) for c in colors]
            
    # Randomized search for 3 colors
    # Greedy with restarts
    for attempt in range(10000):
        if time.time() > deadline - 5:
            break
            
        # Random order
        order = list(range(n))
        random.shuffle(order)
        
        colors = [-1] * n
        valid = True
        for u in order:
            # Determine available colors
            neighbor_colors = set(colors[v] for v in adj[u] if colors[v] != -1)
            available = [c for c in range(3) if c not in neighbor_colors]
            if not available:
                valid = False
                break
            # Pick random available color
            colors[u] = random.choice(available)
            
        if valid and is_valid_coloring(colors):
            return [int(c) for c in colors]
            
    # If 3 colors fail, try 4 colors (should always work)
    # Greedy for 4 colors
    for attempt in range(10000):
        if time.time() > deadline - 5:
            break
            
        order = list(range(n))
        random.shuffle(order)
        
        colors = [-1] * n
        valid = True
        for u in order:
            neighbor_colors = set(colors[v] for v in adj[u] if colors[v] != -1)
            available = [c for c in range(4) if c not in neighbor_colors]
            if not available:
                valid = False
                break
            colors[u] = random.choice(available)
            
        if valid and is_valid_coloring(colors):
            return [int(c) for c in colors]
            
    # Fallback: 50 colors (always valid)
    return list(range(n))
# EVOLVE-BLOCK-END
