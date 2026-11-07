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

def is_valid(colors):
    # Check proper coloring
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # Check no bicolored cycles
    used = list(set(colors))
    for i in range(len(used)):
        for j in range(i+1, len(used)):
            a, b = used[i], used[j]
            uf = UnionFind(N)
            for u, v in edges:
                if (colors[u] == a and colors[v] == b) or (colors[u] == b and colors[v] == a):
                    if not uf.union(u, v):
                        return False
    return True

def count_bicolored_cycles(colors):
    """Count total bicolored cycles (for optimization)."""
    used = list(set(colors))
    total_cycles = 0
    for i in range(len(used)):
        for j in range(i+1, len(used)):
            a, b = used[i], used[j]
            uf = UnionFind(N)
            for u, v in edges:
                if (colors[u] == a and colors[v] == b) or (colors[u] == b and colors[v] == a):
                    if not uf.union(u, v):
                        total_cycles += 1
    return total_cycles

def local_search(colors, max_iters=500, deadline=None):
    """Local search to reduce number of colors used."""
    if deadline is None:
        deadline = time.time() + 60.0
    
    best = list(colors)
    best_colors = len(set(best))
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try to recolor a random vertex to an existing color or a new lower color
        changed = False
        v = random.randint(0, N-1)
        old_color = best[v]
        
        # Get neighbor colors
        neighbor_colors = set()
        for u, w in edges:
            if u == v:
                neighbor_colors.add(best[w])
            elif w == v:
                neighbor_colors.add(best[u])
        
        # Try all possible colors from 0 to best_colors-1 (and maybe one new)
        for new_color in range(best_colors + 1):
            if new_color == old_color:
                continue
            if new_color in neighbor_colors:
                continue
            
            # Test validity
            test_colors = list(best)
            test_colors[v] = new_color
            
            if is_valid(test_colors):
                new_num_colors = len(set(test_colors))
                if new_num_colors < best_colors:
                    best = test_colors
                    best_colors = new_num_colors
                    changed = True
                    break
                elif new_num_colors == best_colors and random.random() < 0.3:
                    # Accept sideways moves to escape local optima
                    best = test_colors
                    changed = True
                    break
        
        if not changed and random.random() < 0.1:
            # Random restart of a vertex's color
            v = random.randint(0, N-1)
            neighbor_colors = set()
            for u, w in edges:
                if u == v:
                    neighbor_colors.add(best[w])
                elif w == v:
                    neighbor_colors.add(best[u])
            
            for new_color in range(best_colors + 2):
                if new_color in neighbor_colors:
                    continue
                test_colors = list(best)
                test_colors[v] = new_color
                if is_valid(test_colors):
                    best = test_colors
                    best_colors = len(set(best))
                    break
    
    return best

def construct():
    deadline = time.time() + 85.0
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best_result = None
    best_num_colors = N + 1
    
    # Strategy 1: Try small k colorings with various patterns
    for k in range(3, 12):
        if time.time() > deadline - 5.0:
            break
        
        # Pattern 1: simple modular
        colors = [i % k for i in range(N)]
        if is_valid(colors):
            num_colors = len(set(colors))
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_result = [int(c) for c in colors]
        
        # Pattern 2: modular with offset
        for offset in range(k):
            if time.time() > deadline - 5.0:
                break
            colors = [(i + offset) % k for i in range(N)]
            if is_valid(colors):
                num_colors = len(set(colors))
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_result = [int(c) for c in colors]
        
        # Pattern 3: use only k-1 colors sometimes
        if k >= 4:
            colors = [i % (k-1) for i in range(N)]
            if is_valid(colors):
                num_colors = len(set(colors))
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_result = [int(c) for c in colors]
    
    # Strategy 2: Greedy coloring with different vertex orderings
    for ordering_seed in range(20):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(ordering_seed)
        order = list(range(N))
        rng.shuffle(order)
        
        colors = [-1] * N
        for v in order:
            neighbor_colors = set()
            for w in adj[v]:
                if colors[w] != -1:
                    neighbor_colors.add(colors[w])
            
            for c in range(50):
                if c not in neighbor_colors:
                    colors[v] = c
                    break
        
        if all(c != -1 for c in colors) and is_valid(colors):
            num_colors = len(set(colors))
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_result = [int(c) for c in colors]
    
    # Strategy 3: Local search from various starting points
    starting_points = []
    
    # Start from bipartite coloring
    bip = [-1] * N
    for start in range(N):
        if bip[start] == -1:
            bip[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if bip[v] == -1:
                        bip[v] = bip[u] ^ 1
                        q.append(v)
    starting_points.append(bip)
    
    # Start from modular colorings
    for k in [3, 4, 5, 6]:
        starting_points.append([i % k for i in range(N)])
    
    # Start from random valid-ish colorings
    for seed in range(10):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        colors = [rng.randint(0, 7) for _ in range(N)]
        starting_points.append(colors)
    
    for start_colors in starting_points:
        if time.time() > deadline - 2.0:
            break
        
        # Ensure it's a proper coloring first
        valid_start = list(start_colors)
        for v in range(N):
            neighbor_colors = set()
            for w in adj[v]:
                if w != v:
                    neighbor_colors.add(valid_start[w])
            if valid_start[v] in neighbor_colors:
                for c in range(50):
                    if c not in neighbor_colors:
                        valid_start[v] = c
                        break
        
        if not is_valid(valid_start):
            continue
        
        result = local_search(valid_start, max_iters=200, deadline=deadline - 2.0)
        num_colors = len(set(result))
        if num_colors < best_num_colors:
            best_num_colors = num_colors
            best_result = [int(c) for c in result]
    
    # Strategy 4: More aggressive local search with simulated annealing-like approach
    if best_result is not None and time.time() < deadline - 3.0:
        current = list(best_result)
        current_colors = len(set(current))
        
        for iteration in range(500):
            if time.time() > deadline - 2.0:
                break
            
            v = random.randint(0, N-1)
            neighbor_colors = set()
            for w in adj[v]:
                neighbor_colors.add(current[w])
            
            # Try all colors
            for new_color in range(current_colors + 2):
                if new_color == current[v] or new_color in neighbor_colors:
                    continue
                
                test_colors = list(current)
                test_colors[v] = new_color
                
                if is_valid(test_colors):
                    new_num_colors = len(set(test_colors))
                    
                    # Accept if better or same with some probability
                    if new_num_colors < current_colors:
                        current = test_colors
                        current_colors = new_num_colors
                        if current_colors < best_num_colors:
                            best_num_colors = current_colors
                            best_result = [int(c) for c in current]
                        break
                    elif new_num_colors == current_colors and random.random() < 0.2:
                        current = test_colors
                        break
    
    # Final fallback
    if best_result is None:
        colors = [0] * N
        for i in range(N):
            used = set()
            for v in adj[i]:
                if v < i:
                    used.add(colors[v])
            for c in range(50):
                if c not in used:
                    colors[i] = c
                    break
        best_result = [int(c) for c in colors]
    
    return best_result
