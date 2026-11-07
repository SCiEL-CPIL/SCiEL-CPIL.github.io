"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    start_time = time.time()
    deadline = start_time + 80.0
    
    vertices_by_degree = sorted(range(n), key=lambda x: -len(adj[x]))
    
    for k in range(4, 50):
        if time.time() > deadline:
            break
        solution = _solve(n, edges, adj, vertices_by_degree, k, deadline)
        if solution is not None:
            return list(solution)
    
    return list(range(n))

def _solve(n, edges, adj, order, k, deadline):
    color_of = [-1] * n
    
    # For each color pair, maintain DSU state with rollback
    # pair_state[key] = (parent, rank, changes_log)
    # We'll use a simpler approach: for each pair, keep a list of edges and use 
    # a global DSU that we rebuild when needed, or use a rollback DSU.
    
    # Let's use: for each pair, a DSU with a log for rollback
    class RollbackDSU:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size
            self.log = []
        
        def find(self, x):
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            self.log.append((rx, ry, self.rank[rx], self.rank[ry]))
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def rollback(self, log_size):
            while len(self.log) > log_size:
                rx, ry, rank_rx, rank_ry = self.log.pop()
                self.parent[ry] = ry
                self.rank[rx] = rank_rx
                self.rank[ry] = rank_ry
    
    pair_dsu = {}
    
    def get_dsu(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in pair_dsu:
            pair_dsu[key] = RollbackDSU(n)
        return pair_dsu[key]
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == len(order):
            return True
        
        v = order[idx]
        
        for c in range(k):
            # Check proper coloring
            if any(color_of[u] == c for u in adj[v]):
                continue
            
            # Check bicolored cycles
            ok = True
            dsu_log_sizes = []
            
            for u in adj[v]:
                d = color_of[u]
                if d == -1 or d == c:
                    continue
                dsu = get_dsu(c, d)
                log_size = len(dsu.log)
                dsu_log_sizes.append((dsu, log_size))
                if not dsu.union(v, u):
                    # Rollback previous unions in this step
                    for d2, ls2 in dsu_log_sizes:
                        d2.rollback(ls2)
                    ok = False
                    break
            
            if not ok:
                continue
            
            color_of[v] = c
            if backtrack(idx + 1):
                return True
            
            color_of[v] = -1
            for d2, ls2 in dsu_log_sizes:
                d2.rollback(ls2)
        
        return False
    
    if backtrack(0):
        return color_of[:]
    return None

# EVOLVE-BLOCK-END

