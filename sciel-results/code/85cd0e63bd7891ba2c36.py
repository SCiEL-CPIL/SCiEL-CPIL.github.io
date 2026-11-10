import time
import random

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def has_bicolored_cycle(colors, k):
        """Check if any pair of colors among 0..k-1 forms a cycle."""
        # For each pair of colors, check if the induced subgraph has a cycle
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                # Union-find on vertices with color c1 or c2
                parent = list(range(n))
                
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
                
                for u, v in edges:
                    cu, cv = colors[u], colors[v]
                    if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                        if not union(u, v):
                            return True  # Cycle found
        return False
    
    def is_valid(colors, k):
        """Check proper coloring and bicolored-forest constraint for k colors."""
        # Check proper coloring
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        # Check bicolored forest
        return not has_bicolored_cycle(colors, k)
    
    # Try deterministic patterns for 3 colors
    patterns_3 = [
        lambda i: i % 3,
        lambda i: (i * 2) % 3,
        lambda i: (i + i // 7) % 3,
        lambda i: (i + i // 10) % 3,
        lambda i: (i * 7) % 3,
        lambda i: (i + i // 5) % 3,
        lambda i: (i * 3 + i // 17) % 3,
        lambda i: (i * 5 + i // 11) % 3,
    ]
    
    for p in patterns_3:
        colors = [p(i) for i in range(n)]
        if is_valid(colors, 3):
            return [int(c) for c in colors]
    
    # Fast backtracking for 3 colors with incremental cycle checking
    def backtrack_3(order, colors, idx):
        if time.time() > deadline:
            return None
        
        if idx == len(order):
            if is_valid(colors, 3):
                return colors[:]
            return None
        
        v = order[idx]
        
        # Determine forbidden colors from neighbors
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        
        for c in range(3):
            if c in forbidden:
                continue
            
            colors[v] = c
            
            # Quick check: does adding v create a bicolored cycle?
            # Only need to check pairs (c, c2) where c2 != c
            valid = True
            for c2 in range(3):
                if c2 == c:
                    continue
                # Check if there's a path between two neighbors of v with color c2
                # through the {c, c2} subgraph
                # Build union-find on {c, c2} vertices
                parent = list(range(n))
                
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
                
                # Process edges between c and c2 vertices
                for u, w in edges:
                    cu, cw = colors[u], colors[w]
                    if (cu == c and cw == c2) or (cu == c2 and cw == c):
                        if not union(u, w):
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                result = backtrack_3(order, colors, idx + 1)
                if result is not None:
                    return result
            
            colors[v] = -1
        
        return None
    
    # Try backtracking with several orderings
    orderings = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        # Try ordering by "structure" - vertices with more "diagonal" connections first
        sorted(range(n), key=lambda x: (x % 7, x)),
    ]
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        
        colors = [-1] * n
        result = backtrack_3(order, colors, 0)
        if result is not None:
            return [int(c) for c in result]
    
    # If 3 colors fail, try 4 colors with backtracking
    def backtrack_4(order, colors, idx):
        if time.time() > deadline:
            return None
        
        if idx == len(order):
            if is_valid(colors, 4):
                return colors[:]
            return None
        
        v = order[idx]
        
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        
        for c in range(4):
            if c in forbidden:
                continue
            
            colors[v] = c
            
            # Quick bicolored cycle check
            valid = True
            for c2 in range(4):
                if c2 == c:
                    continue
                parent = list(range(n))
                
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
                
                for u, w in edges:
                    cu, cw = colors[u], colors[w]
                    if (cu == c and cw == c2) or (cu == c2 and cw == c):
                        if not union(u, w):
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                result = backtrack_4(order, colors, idx + 1)
                if result is not None:
                    return result
            
            colors[v] = -1
        
        return None
    
    for order in orderings[:2]:
        if time.time() > deadline - 5:
            break
        
        colors = [-1] * n
        result = backtrack_4(order, colors, 0)
        if result is not None:
            return [int(c) for c in result]
    
    # Fallback: random search for 4 colors
    for attempt in range(10000):
        if time.time() > deadline - 3:
            break
        
        order = list(range(n))
        random.shuffle(order)
        
        colors = [-1] * n
        valid = True
        for v in order:
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] != -1:
                    forbidden.add(colors[nb])
            
            available = [c for c in range(4) if c not in forbidden]
            if not available:
                valid = False
                break
            
            colors[v] = random.choice(available)
        
        if valid and is_valid(colors, 4):
            return [int(c) for c in colors]
    
    # Final fallback: 5 colors
    for attempt in range(5000):
        if time.time() > deadline - 1:
            break
        
        order = list(range(n))
        random.shuffle(order)
        
        colors = [-1] * n
        valid = True
        for v in order:
            forbidden = set()
            for nb in adj[v]:
                if colors[nb] != -1:
                    forbidden.add(colors[nb])
            
            available = [c for c in range(5) if c not in forbidden]
            if not available:
                valid = False
                break
            
            colors[v] = random.choice(available)
        
        if valid and is_valid(colors, 5):
            return [int(c) for c in colors]
    
    # Last resort: unique colors
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
