import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']

def _check_bicolored_cycles(colors):
    """Return set of color-pairs that have a bicolored cycle. Empty set means valid."""
    # Build union-find per color pair
    color_pairs = set()
    for u, v in edges:
        a, b = colors[u], colors[v]
        if a != b:
            color_pairs.add((min(a, b), max(a, b)))
    
    bad_pairs = set()
    for a, b in color_pairs:
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
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
            if (cu == a and cv == b) or (cu == b and cv == a):
                if not union(u, v):
                    bad_pairs.add((a, b))
                    break
    return bad_pairs

def _is_valid(colors):
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    return len(_check_bicolored_cycles(colors)) == 0

def construct():
    deadline = time.time() + 80  # safety margin
    
    # Precompute adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Start with modular 3-coloring
    colors = [v % 3 for v in range(50)]
    
    # Try to fix bicolored cycles via local swaps
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        bad = _check_bicolored_cycles(colors)
        if not bad:
            # Verify properness
            proper = all(colors[u] != colors[v] for u, v in edges)
            if proper:
                return [int(c) for c in colors]
        
        # Pick a bad pair and try to fix it
        if bad:
            a, b = random.choice(list(bad))
        else:
            break
        
        # Find vertices with color a or b
        candidates = [v for v in range(50) if colors[v] in (a, b)]
        
        # Try swapping one vertex's color to break the cycle
        fixed = False
        for _ in range(20):
            if time.time() > deadline:
                break
            v = random.choice(candidates)
            old_color = colors[v]
            # Try changing to the third color
            new_color = 3 - old_color  # if old is 0 or 1, new is 1 or 0; if old is 2, new is 1
            # Actually let's try all other colors
            for nc in range(3):
                if nc == old_color:
                    continue
                # Check properness locally
                ok = True
                for nb in adj[v]:
                    if colors[nb] == nc:
                        ok = False
                        break
                if not ok:
                    continue
                colors[v] = nc
                bad2 = _check_bicolored_cycles(colors)
                if len(bad2) < len(bad):
                    fixed = True
                    break
                colors[v] = old_color
            if fixed:
                break
        
        if not fixed:
            # Try a random perturbation
            v = random.randint(0, 49)
            old_color = colors[v]
            nc = random.randint(0, 2)
            if nc != old_color:
                ok = all(colors[nb] != nc for nb in adj[v])
                if ok:
                    colors[v] = nc
    
    # Fallback to 4-coloring
    colors4 = []
    for v in range(50):
        r = v // 5
        c = v % 5
        col = (r % 2) * 2 + (c % 2)
        colors4.append(int(col))
    
    if _is_valid(colors4):
        return colors4
    
    return [int(i) for i in range(50)]
