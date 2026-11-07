import time
import random
import sys
from collections import defaultdict, deque

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

# Precompute edge indices for faster access
edge_list = list(edges)

def get_cost(colors):
    """
    Returns a tuple (proper_violations, acyclic_violations).
    Both should be 0 for a valid coloring.
    """
    proper_v = 0
    for u, v in edge_list:
        if colors[u] == colors[v]:
            proper_v += 1
    
    if proper_v > 0:
        return proper_v * 1000, 0

    used_colors = set(colors)
    acyclic_v = 0
    color_list = list(used_colors)
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            a, b = color_list[i], color_list[j]
            uf = UnionFind(N)
            has_cycle = False
            for u, v in edge_list:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not uf.union(u, v):
                        has_cycle = True
                        break
            if has_cycle:
                acyclic_v += 1
    
    return 0, acyclic_v

def total_cost(colors):
    p, a = get_cost(colors)
    return p + a

def is_valid(colors):
    cost = total_cost(colors)
    return cost == 0

def local_search_cost(K, time_limit, seed=0):
    """
    Cost-based local search for a valid K-coloring.
    Returns the coloring if found, else None.
    """
    start_time = time.time()
    rng = random.Random(seed)
    best_coloring = None
    best_cost = float('inf')
    
    for restart in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        colors = [rng.randint(0, K - 1) for _ in range(N)]
        current_cost = total_cost(colors)
        
        if current_cost == 0:
            return colors
            
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - start_time < time_limit:
            if current_cost == 0:
                return colors
                
            # Pick a random vertex
            v = rng.randint(0, N - 1)
            old_color = colors[v]
            
            best_new_color = old_color
            min_cost = current_cost
            
            # Try all other colors
            for c in range(K):
                if c == old_color:
                    continue
                colors[v] = c
                cost = total_cost(colors)
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

def backtracking_search(K, time_limit, seed=0):
    """
    Backtracking search with forward checking for a valid K-coloring.
    """
    start_time = time.time()
    rng = random.Random(seed)
    colors = [-1] * N
    
    # Order vertices by degree (highest first)
    order = sorted(range(N), key=lambda x: -len(adj[x]))
    
    def is_proper(v, c):
        for w in adj[v]:
            if colors[w] == c:
                return False
        return True
    
    def check_acyclic(v, c):
        """Check if assigning color c to vertex v creates a bicolored cycle."""
        # For each pair of colors (c, other), check if adding v creates a cycle
        # We need to check all pairs involving c
        used_colors = set(colors)
        used_colors.add(c)
        for other in used_colors:
            if other == c:
                continue
            # Check if there's a path from v to any neighbor of v with color other
            # that doesn't go through v
            uf = UnionFind(N)
            for u, w in edge_list:
                cu = colors[u] if colors[u] != -1 else -1
                cv = colors[w] if colors[w] != -1 else -1
                # If one endpoint is v, use c
                if u == v:
                    cu = c
                if w == v:
                    cv = c
                if (cu == c and cv == other) or (cu == other and cv == c):
                    if not uf.union(u, w):
                        return False
        return True
    
    def backtrack(idx):
        if time.time() - start_time > time_limit:
            return False
        
        if idx == N:
            return True
        
        v = order[idx]
        # Try colors in a heuristic order
        neighbor_colors = set()
        for w in adj[v]:
            if colors[w] != -1:
                neighbor_colors.add(colors[w])
        
        for c in range(K):
            if c in neighbor_colors:
                continue
            
            if not is_proper(v, c):
                continue
            
            colors[v] = c
            if check_acyclic(v, c):
                if backtrack(idx + 1):
                    return True
            colors[v] = -1
        
        return False
    
    if backtrack(0):
        return colors
    return None

def construct():
    deadline = time.time() + 85.0
    
    # Try backtracking for small K first
    for K in [3, 4, 5, 6]:
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        time_alloc = min(10.0, remaining * 0.3)
        result = backtracking_search(K, time_alloc, seed=K)
        if result is not None:
            # Verify
            if is_valid(result):
                return [int(c) for c in result]
    
    # Try cost-based local search
    for K in [3, 4, 5, 6, 7, 8]:
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        
        time_alloc = min(15.0, remaining * 0.4)
        result = local_search_cost(K, time_alloc, seed=K * 100)
        if result is not None:
            if is_valid(result):
                return [int(c) for c in result]
    
    # Fallback: greedy coloring with many colors
    colors = [-1] * N
    for v in range(N):
        neighbor_colors = set()
        for w in adj[v]:
            if colors[w] != -1:
                neighbor_colors.add(colors[w])
        
        for c in range(50):
            if c not in neighbor_colors:
                colors[v] = c
                break
    
    if all(c != -1 for c in colors) and is_valid(colors):
        return [int(c) for c in colors]
    
    # Absolute fallback: unique colors
    return list(range(N))
