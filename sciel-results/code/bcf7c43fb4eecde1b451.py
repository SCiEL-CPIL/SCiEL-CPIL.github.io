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

# Precompute adjacency
N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

def count_bicolored_cycles(colors, K):
    """Count bicolored cycles using union-find per color pair."""
    n = len(colors)
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        if cu != cv:
            if cu > cv:
                cu, cv = cv, cu
            pair_edges[(cu, cv)].append((u, v))
    
    cycle_count = 0
    for (a, b), elist in pair_edges.items():
        uf = UnionFind(n)
        for u, v in elist:
            if not uf.union(u, v):
                cycle_count += 1
                break
    return cycle_count

def is_valid(colors, K):
    """Check if coloring is valid (proper + no bicolored cycles)."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    return count_bicolored_cycles(colors, K) == 0

def backtrack_search(K, deadline):
    """Backtracking search with forward checking for bicolored cycle constraint."""
    colors = [-1] * N
    
    def would_create_cycle(v, c, d):
        """Check if adding vertex v (color c) to the (c,d) subgraph creates a cycle."""
        d_neighbors = [u for u in adj[v] if colors[u] == d]
        if len(d_neighbors) <= 1:
            return False
        
        # Check if any two d_neighbors are already connected in the (c,d) subgraph
        # Build connectivity in (c,d) subgraph excluding v
        def bfs(start):
            visited = {start}
            queue = [start]
            while queue:
                cur = queue.pop(0)
                for nb in adj[cur]:
                    if nb == v:
                        continue
                    if colors[nb] == c or colors[nb] == d:
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
            return visited
        
        # Check connectivity between pairs of d_neighbors
        checked = set()
        for i in range(len(d_neighbors)):
            for j in range(i + 1, len(d_neighbors)):
                a, b = d_neighbors[i], d_neighbors[j]
                if a in checked or b in checked:
                    continue
                comp = bfs(a)
                checked.update(comp)
                if b in comp:
                    return True
        
        return False
    
    def try_assign(v, c):
        """Try to assign color c to vertex v. Returns True if valid."""
        for u in adj[v]:
            if colors[u] == c:
                return False
        
        neighbor_colors = set()
        for u in adj[v]:
            if colors[u] >= 0 and colors[u] != c:
                neighbor_colors.add(colors[u])
        
        for d in neighbor_colors:
            if would_create_cycle(v, c, d):
                return False
        
        colors[v] = c
        return True
    
    def unassign(v):
        colors[v] = -1
    
    # Vertex ordering: degree descending
    order = sorted(range(N), key=lambda x: -len(adj[x]))
    
    def solve(idx):
        if time.time() > deadline:
            return False
        if idx == N:
            return True
        
        v = order[idx]
        
        neighbor_colors = set()
        for u in adj[v]:
            if colors[u] >= 0:
                neighbor_colors.add(colors[u])
        
        for c in range(K):
            if c in neighbor_colors:
                continue
            if try_assign(v, c):
                if solve(idx + 1):
                    return True
                unassign(v)
        
        return False
    
    if solve(0):
        return list(colors)
    return None

def local_search(K, time_limit):
    """
    Runs local search for a valid K-coloring.
    Returns the coloring if found, else None.
    """
    start_time = time.time()
    best_coloring = None
    best_cost = float('inf')
    
    for restart in range(2000):
        if time.time() - start_time > time_limit:
            break
            
        colors = [random.randint(0, K - 1) for _ in range(N)]
        current_cost = 0
        for u, v in edges:
            if colors[u] == colors[v]:
                current_cost += 1000
        if current_cost == 0:
            current_cost = count_bicolored_cycles(colors, K)
        
        if current_cost == 0:
            return colors
            
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() - start_time < time_limit:
            if current_cost == 0:
                return colors
                
            v = random.randint(0, N - 1)
            old_color = colors[v]
            
            best_new_color = old_color
            min_cost = current_cost
            
            for c in range(K):
                if c == old_color:
                    continue
                colors[v] = c
                cost = 0
                for u in adj[v]:
                    if colors[u] == c:
                        cost = 1000
                        break
                if cost == 0:
                    cost = count_bicolored_cycles(colors, K)
                if cost < min_cost:
                    min_cost = cost
                    best_new_color = c
                colors[v] = old_color
            
            colors[v] = best_new_color
            new_cost = min_cost
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if current_cost < best_cost:
                best_cost = current_cost
                best_coloring = list(colors)
                
            if no_improve_count > max_no_improve:
                break

    if best_cost == 0 and best_coloring is not None:
        return best_coloring
    return None

def construct():
    # Deadline guard
    deadline = time.time() + 55.0
    
    # Try backtracking search for K=3
    res = backtrack_search(3, deadline - 5.0)
    if res is not None:
        return res
    
    # Try backtracking search for K=4
    remaining = deadline - time.time()
    if remaining > 10:
        res = backtrack_search(4, deadline - 5.0)
        if res is not None:
            return res
    
    # Try local search for K=3
    remaining = deadline - time.time()
    if remaining > 5:
        res = local_search(3, min(10.0, remaining - 3.0))
        if res is not None:
            return res
    
    # Try local search for K=4
    remaining = deadline - time.time()
    if remaining > 5:
        res = local_search(4, min(10.0, remaining - 3.0))
        if res is not None:
            return res
    
    # Try local search for K=5
    remaining = deadline - time.time()
    if remaining > 3:
        res = local_search(5, min(5.0, remaining - 2.0))
        if res is not None:
            return res
            
    # Fallback
    if time.time() < deadline - 1.0:
        res = local_search(10, 2.0)
        if res is not None:
            return res
            
    # Absolute fallback: unique colors
    return list(range(N))

# EVOLVE-BLOCK-END
