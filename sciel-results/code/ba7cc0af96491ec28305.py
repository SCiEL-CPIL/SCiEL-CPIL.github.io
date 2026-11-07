"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

# Precompute adjacency
ADJ = [[] for _ in range(P['n'])]
for u, v in P['edges']:
    ADJ[u].append(v)
    ADJ[v].append(u)

# For each vertex, the set of neighbors
NEIGHBOR_SETS = [set(adj) for adj in ADJ]


class DSU:
    """Disjoint Set Union with path compression and union by rank."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def check_bicolored_forest(coloring, color_a, color_b):
    """Check if subgraph induced by vertices with color_a or color_b is a forest.
    Returns True if it's a forest (no cycles)."""
    vertices = [i for i in range(P['n']) if coloring[i] in (color_a, color_b)]
    if len(vertices) <= 1:
        return True
    dsu = DSU(P['n'])
    for u in vertices:
        for v in ADJ[u]:
            if v > u and coloring[v] in (color_a, color_b):
                if not dsu.union(u, v):
                    return False
    return True


def is_valid_coloring(coloring, k):
    """Check if coloring is valid: adjacent differ and no bicolored cycle."""
    # Check adjacent vertices have different colors
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    # Check no bicolored cycle
    used_colors = set(coloring)
    for ca in used_colors:
        for cb in used_colors:
            if ca >= cb:
                continue
            if not check_bicolored_forest(coloring, ca, cb):
                return False
    return True


def count_bicolored_cycles(coloring, k):
    """Count total bicolored cycles across all color pairs."""
    total = 0
    used_colors = set(coloring)
    for ca in used_colors:
        for cb in used_colors:
            if ca >= cb:
                continue
            # Count edges in the subgraph induced by colors ca and cb
            vertices = [i for i in range(P['n']) if coloring[i] in (ca, cb)]
            if len(vertices) <= 1:
                continue
            dsu = DSU(P['n'])
            for u in vertices:
                for v in ADJ[u]:
                    if v > u and coloring[v] in (ca, cb):
                        if not dsu.union(u, v):
                            total += 1
    return total


def backtracking_solve(k, deadline):
    """Try to find a valid k-coloring using backtracking with forward checking."""
    coloring = [-1] * P['n']
    
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(P['n']), key=lambda v: -len(ADJ[v]))
    
    # For each vertex in order, track which colors are forbidden by neighbors
    # We'll assign in the 'order' sequence
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == P['n']:
            # Verify full validity
            if is_valid_coloring(coloring, k):
                return coloring[:]
            return None
        
        v = order[idx]
        
        # Determine which colors are available for vertex v
        # A color is forbidden if any neighbor already has that color
        forbidden = set()
        for nb in ADJ[v]:
            if coloring[nb] != -1:
                forbidden.add(coloring[nb])
        
        # Try each color
        for c in range(k):
            if c in forbidden:
                continue
            
            coloring[v] = c
            
            # Forward checking: verify that for every already-assigned neighbor,
            # the bicolored subgraph doesn't have a cycle
            # Actually, we need to check: for each color pair (c, x) where x is 
            # a color of an assigned neighbor, the subgraph induced by c and x 
            # must not contain a cycle.
            valid = True
            for nb in ADJ[v]:
                if coloring[nb] != -1 and coloring[nb] != c:
                    if not check_bicolored_forest(coloring, c, coloring[nb]):
                        valid = False
                        break
            
            if not valid:
                coloring[v] = -1
                continue
            
            result = backtrack(idx + 1)
            if result is not None:
                return result
            
            coloring[v] = -1
        
        return None
    
    return backtrack(0)


def local_search(k, deadline):
    """Local search: start from a random k-coloring, try to fix bicolored cycles."""
    import random
    
    best_coloring = None
    best_cycles = float('inf')
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Initialize: try to get a proper coloring first
        coloring = [0] * P['n']
        
        # Greedy proper coloring
        for v in range(P['n']):
            used = set(coloring[nb] for nb in ADJ[v] if coloring[nb] != -1)
            # Actually coloring is all 0s initially, let's do it properly
            pass
        
        # Random proper coloring
        coloring = [-1] * P['n']
        for v in range(P['n']):
            forbidden = set(coloring[nb] for nb in ADJ[v] if coloring[nb] != -1)
            available = [c for c in range(k) if c not in forbidden]
            if not available:
                # Backtrack not done here; just pick randomly
                available = list(range(k))
            coloring[v] = random.choice(available)
        
        # Ensure proper coloring
        changed = True
        while changed:
            changed = False
            for u, v in P['edges']:
                if coloring[u] == coloring[v]:
                    # Change v to a different color
                    forbidden = set(coloring[nb] for nb in ADJ[v])
                    available = [c for c in range(k) if c not in forbidden]
                    if available:
                        coloring[v] = random.choice(available)
                        changed = True
        
        cycles = count_bicolored_cycles(coloring, k)
        if cycles < best_cycles:
            best_cycles = cycles
            best_coloring = coloring[:]
        
        if cycles == 0:
            break
        
        # Local search: try to reduce cycles
        for step in range(200):
            if time.time() > deadline:
                break
            
            # Find a vertex to recolor
            v = random.randint(0, P['n'] - 1)
            old_c = coloring[v]
            
            # Try all other colors
            best_new_c = old_c
            best_new_cycles = cycles
            
            for c in range(k):
                if c == old_c:
                    continue
                # Check if c is forbidden by neighbors
                if any(coloring[nb] == c for nb in ADJ[v]):
                    continue
                
                coloring[v] = c
                new_cycles = count_bicolored_cycles(coloring, k)
                if new_cycles < best_new_cycles:
                    best_new_cycles = new_cycles
                    best_new_c = c
            
            coloring[v] = best_new_c
            cycles = best_new_cycles
            
            if cycles == 0:
                break
        
        if cycles < best_cycles:
            best_cycles = cycles
            best_coloring = coloring[:]
    
    if best_coloring is not None and best_cycles == 0:
        return best_coloring
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Try increasing k from 3 upward
    for k in range(3, 50):
        if time.time() > deadline:
            break
        
        # First try backtracking (more reliable for small k)
        result = backtracking_solve(k, deadline)
        if result is not None:
            # Verify
            if is_valid_coloring(result, k):
                return [int(c) for c in result]
        
        # Try local search
        result = local_search(k, deadline)
        if result is not None:
            if is_valid_coloring(result, k):
                return [int(c) for c in result]
    
    # Fallback: use all 50 colors (always valid since no two adjacent share a color
    # and any two-color subgraph has at most one vertex per color, so no cycles)
    return [int(i) for i in range(P['n'])]
# EVOLVE-BLOCK-END
