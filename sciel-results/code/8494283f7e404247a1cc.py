import time
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

n = P['n']
edges = P['edges']

adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def has_bicolored_cycle(coloring, k):
    for c1 in range(k):
        for c2 in range(c1 + 1, k):
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
                    return False
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True
            
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if not union(u, v):
                        return True
    return False

def validate(coloring):
    k = max(coloring) + 1
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    return not has_bicolored_cycle(coloring, k)

def solve_csp(k):
    domains = [set(range(k)) for _ in range(n)]
    assigned = [None] * n
    
    def backtrack():
        min_size = k + 1
        min_vertex = -1
        for v in range(n):
            if assigned[v] is None:
                size = len(domains[v])
                if size < min_size:
                    min_size = size
                    min_vertex = v
                    if size == 0:
                        return False
                    if size == 1:
                        break
        
        if min_vertex == -1:
            return True
        
        v = min_vertex
        for c in sorted(domains[v]):
            valid = True
            for u in adj[v]:
                if assigned[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            saved_domain = set(domains[v])
            neighbor_saved = {}
            for u in adj[v]:
                if assigned[u] is None:
                    neighbor_saved[u] = set(domains[u])
            
            assigned[v] = c
            domains[v] = {c}
            
            ok = True
            for u in adj[v]:
                if assigned[u] is None:
                    if c in domains[u]:
                        domains[u].discard(c)
                        if not domains[u]:
                            ok = False
                            break
            
            if ok and backtrack():
                return True
            
            assigned[v] = None
            domains[v] = saved_domain
            for u, saved in neighbor_saved.items():
                domains[u] = saved
        
        return False
    
    if backtrack():
        return assigned
    return None

def construct():
    start = time.time()
    deadline = 5.0
    
    for k in range(3, 6):
        if time.time() - start > deadline:
            break
        result = solve_csp(k)
        if result is not None:
            if validate(result):
                return [int(c) for c in result]
    
    # Fallback: greedy
    import random
    random.seed(42)
    best = None
    for k in range(3, 8):
        for _ in range(100):
            coloring = [-1] * n
            order = list(range(n))
            random.shuffle(order)
            success = True
            for v in order:
                colors = list(range(k))
                random.shuffle(colors)
                placed = False
                for c in colors:
                    if all(coloring[u] != c for u in adj[v]):
                        coloring[v] = c
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            if success and validate(coloring):
                return [int(c) for c in coloring]
                
    return [int(i) for i in range(n)]
