import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)

def _is_valid(coloring):
    """Check proper coloring and bicolored-forest constraint."""
    for a, b in _EDGES:
        if coloring[a] == coloring[b]:
            return False
    present = sorted(set(coloring))
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            ci, cj = present[i], present[j]
            parent = list(range(_N))
            rank = [0] * _N

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

            for a, b in _EDGES:
                ca, cb = coloring[a], coloring[b]
                if ca in (ci, cj) and cb in (ci, cj):
                    if not union(a, b):
                        return False
    return True

def _find_4_coloring(deadline):
    """
    Attempt to find a valid 4-coloring using randomized restarts and local search/backtracking.
    """
    # Try a few random greedy initializations
    rng = random.Random(42)
    
    # Vertex order: highest degree first (all are 4, so tie-break by index or random)
    # We can use a static order or randomize it.
    
    for attempt in range(200):
        if time.time() > deadline:
            return None
            
        # Random order
        order = list(range(_N))
        rng.shuffle(order)
        
        coloring = [-1] * _N
        
        # Greedy phase
        failed = False
        for v in order:
            used = set()
            for nb in _ADJ[v]:
                if coloring[nb] != -1:
                    used.add(coloring[nb])
            
            # Try colors 0-3
            available = [c for c in range(4) if c not in used]
            if not available:
                failed = True
                break
            
            # Heuristic: pick color that minimizes future conflicts?
            # For now, pick random from available
            coloring[v] = rng.choice(available)
        
        if failed:
            continue
            
        # Check validity
        if _is_valid(coloring):
            return coloring
            
        # If greedy failed validity, try a simple local search (swap colors)
        # Or just restart. Given the small size, restarts are cheap.
        
    return None

def construct():
    deadline = time.time() + 80.0
    
    # 1. Try to find 4-coloring
    c4 = _find_4_coloring(deadline)
    if c4 is not None:
        return [int(x) for x in c4]
        
    # 2. Fallback to 5, 6, etc. using greedy
    for k in range(5, 12):
        if time.time() > deadline - 5.0:
            break
        # Greedy with random restarts
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            order = list(range(_N))
            random.shuffle(order)
            coloring = [-1] * _N
            ok = True
            for v in order:
                used = set()
                for nb in _ADJ[v]:
                    if coloring[nb] != -1:
                        used.add(coloring[nb])
                available = [c for c in range(k) if c not in used]
                if not available:
                    ok = False
                    break
                coloring[v] = available[0] # deterministic first available
            if ok and _is_valid(coloring):
                return [int(x) for x in coloring]
                
    # 3. Last resort: 50 colors
    return list(range(_N))
