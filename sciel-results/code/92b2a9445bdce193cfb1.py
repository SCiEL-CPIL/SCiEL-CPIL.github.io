import time
import random
import sys
from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

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

def get_cost(colors):
    """
    Calculates the cost of a coloring.
    Cost = 1000 * (proper violations) + 100 * (bicolored cycle violations)
    """
    proper_v = 0
    for u, v in edges:
        if colors[u] == colors[v]:
            proper_v += 1
            
    if proper_v > 0:
        return proper_v * 1000
        
    used_colors = set(colors)
    color_list = list(used_colors)
    acyclic_v = 0
    
    # Check all pairs of used colors
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            a, b = color_list[i], color_list[j]
            uf = UnionFind(N)
            has_cycle = False
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not uf.union(u, v):
                        has_cycle = True
                        break
            if has_cycle:
                acyclic_v += 1
                
    return acyclic_v * 100

def local_search(K, time_limit):
    """
    Runs local search for a valid K-coloring.
    """
    start_time = time.time()
    best_coloring = None
    best_cost = float('inf')
    
    # Try multiple restarts
    while time.time() - start_time < time_limit:
        colors = [random.randint(0, K - 1) for _ in range(N)]
        current_cost = get_cost(colors)
        
        if current_cost == 0:
            return list(colors)
            
        if current_cost < best_cost:
            best_cost = current_cost
            best_coloring = list(colors)
            
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() - start_time < time_limit and no_improve < max_no_improve:
            if current_cost == 0:
                return list(colors)
                
            # Pick a random vertex
            v = random.randint(0, N - 1)
            old_color = colors[v]
            
            best_new_color = old_color
            min_cost = current_cost
            
            # Try all other colors
            for c in range(K):
                if c == old_color:
                    continue
                colors[v] = c
                cost = get_cost(colors)
                if cost < min_cost:
                    min_cost = cost
                    best_new_color = c
                colors[v] = old_color
            
            # Apply best move
            if best_new_color != old_color:
                colors[v] = best_new_color
                new_cost = min_cost
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_coloring = list(colors)
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
    if best_cost == 0 and best_coloring is not None:
        return best_coloring
    return None

def construct():
    deadline = time.time() + 55.0
    
    # Try K=3
    t_start = time.time()
    res = local_search(3, min(20.0, deadline - time.time() - 5.0))
    if res is not None:
        return res
        
    # Try K=4
    remaining = deadline - time.time()
    if remaining > 10:
        res = local_search(4, min(20.0, remaining - 5.0))
        if res is not None:
            return res
            
    # Try K=5
    remaining = deadline - time.time()
    if remaining > 5:
        res = local_search(5, min(10.0, remaining - 3.0))
        if res is not None:
            return res
            
    # Try K=6
    remaining = deadline - time.time()
    if remaining > 2:
        res = local_search(6, min(5.0, remaining - 1.0))
        if res is not None:
            return res
            
    # Fallback: Greedy unique colors
    return list(range(N))
