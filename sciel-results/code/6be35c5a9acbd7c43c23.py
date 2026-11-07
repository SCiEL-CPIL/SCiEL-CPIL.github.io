import time
import sys
from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def construct():
    deadline = time.time() + 55.0
    # Try K from 3 upwards
    for K in [3, 4, 5, 6]:
        res = solve(K, deadline)
        if res is not None:
            return res
    # Fallback
    return list(range(N))

def solve(K, deadline):
    start_time = time.time()
    neighbors = [adj[i] for i in range(N)]
    vertex_order = list(range(N))
    
    colors = [-1] * N
    pair_ufs = {}
    
    def get_pair_uf(a, b):
        key = (min(a, b), max(a, b))
        if key not in pair_ufs:
            pair_ufs[key] = UnionFind(N)
        return pair_ufs[key]
    
    def is_valid(v, c):
        for u in neighbors[v]:
            if colors[u] == c:
                return False
        for u in neighbors[v]:
            cu = colors[u]
            if cu == -1:
                continue
            if cu == c:
                return False
            uf = get_pair_uf(c, cu)
            if uf.find(v) == uf.find(u):
                return False
        return True
    
    def backtrack(idx):
        if time.time() > deadline:
            return -1
        if idx == N:
            return 1
        
        v = vertex_order[idx]
        
        for c in range(K):
            if is_valid(v, c):
                colors[v] = c
                
                changes = []
                for u in neighbors[v]:
                    cu = colors[u]
                    if cu == -1 or cu == c:
                        continue
                    key = (min(c, cu), max(c, cu))
                    uf = pair_ufs[key]
                    rv = uf.find(v)
                    ru = uf.find(u)
                    if rv != ru:
                        changes.append((key, rv, ru, uf.parent[ru], uf.rank[rv], uf.rank[ru]))
                        if uf.rank[rv] < uf.rank[ru]:
                            rv, ru = ru, rv
                        uf.parent[ru] = rv
                        if uf.rank[rv] == uf.rank[ru]:
                            uf.rank[rv] += 1
                
                result = backtrack(idx + 1)
                if result == 1:
                    return 1
                elif result == -1:
                    for (key, rv, ru, parent_ru, rank_rv, rank_ru) in changes:
                        uf = pair_ufs[key]
                        uf.parent[ru] = parent_ru
                        uf.rank[rv] = rank_rv
                        uf.rank[ru] = rank_ru
                    colors[v] = -1
                    return -1
                
                for (key, rv, ru, parent_ru, rank_rv, rank_ru) in reversed(changes):
                    uf = pair_ufs[key]
                    uf.parent[ru] = parent_ru
                    uf.rank[rv] = rank_rv
                    uf.rank[ru] = rank_ru
                colors[v] = -1
                
        return 0
    
    result = backtrack(0)
    if result == 1:
        return list(colors)
    return None
