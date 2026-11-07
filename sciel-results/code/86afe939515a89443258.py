"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency list
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

def _verify_coloring(colors, n, edges):
    """Verify that the coloring is valid: adjacent differ and bicolored subgraphs are forests."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # Check bicolored forest constraint using union-find
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        def find(x):
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
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def _solve_3_coloring(n, edges, deadline):
    """
    Solve 3-coloring with bicolored-forest constraint using backtracking with rollback DSU.
    Returns list of colors or None.
    """
    adj_local = [[] for _ in range(n)]
    for u, v in edges:
        adj_local[u].append(v)
        adj_local[v].append(u)
    
    colors = [-1] * n
    
    # Vertex ordering: degree descending, then index
    degrees = [len(adj_local[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: (-degrees[x], x))
    
    # For 3 colors, pairs are (0,1), (0,2), (1,2)
    pair_to_idx = {(0,1): 0, (0,2): 1, (1,2): 2}
    
    class RollbackDSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.history = []
        
        def find(self, x):
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.history.append((ry, rx, self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def rollback(self, count):
            for _ in range(count):
                child, parent, rank_parent_before = self.history.pop()
                self.parent[child] = child
                self.rank[parent] = rank_parent_before
    
    dsus = [RollbackDSU(n) for _ in range(3)]
    
    def get_pair_idx(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        return pair_to_idx[(c1, c2)]
    
    def try_assign(v, color):
        assignments = []
        for nb in adj_local[v]:
            if colors[nb] == -1:
                continue
            if colors[nb] == color:
                for (pi, u2, v2) in assignments:
                    dsus[pi].rollback(1)
                return None
            pi = get_pair_idx(color, colors[nb])
            if dsus[pi].union(v, nb):
                assignments.append((pi, v, nb))
            else:
                for (pi2, u2, v2) in assignments:
                    dsus[pi2].rollback(1)
                return None
        return assignments
    
    def undo_assign(assignments):
        for (pi, u, v_edge) in assignments:
            dsus[pi].rollback(1)
    
    # Use forward checking: precompute constraints
    node_count = [0]
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        
        # Get forbidden colors from colored neighbors
        forbidden = [False] * 3
        for nb in adj_local[v]:
            if colors[nb] != -1:
                forbidden[colors[nb]] = True
        
        # Try colors in order of least constraining value
        # Count how many future vertices each color would conflict with
        for color in range(3):
            if forbidden[color]:
                continue
            
            assignments = try_assign(v, color)
            if assignments is None:
                continue
            
            colors[v] = color
            if backtrack(idx + 1):
                return True
            colors[v] = -1
            undo_assign(assignments)
        
        return False
    
    if backtrack(0):
        return colors
    return None

def _greedy_with_local_search(n, edges, deadline, num_restarts=200):
    """
    Greedy coloring with local search to eliminate colors.
    """
    adj_local = [[] for _ in range(n)]
    for u, v in edges:
        adj_local[u].append(v)
        adj_local[v].append(u)
    
    best_result = None
    best_num_colors = n + 1
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 12345 + 67890)
        
        # Start with a greedy 4-coloring
        order = list(range(n))
        rng.shuffle(order)
        
        colors = [-1] * n
        max_colors = 4
        
        for v in order:
            forbidden = set()
            for nb in adj_local[v]:
                if colors[nb] != -1:
                    forbidden.add(colors[nb])
            
            assigned = False
            for c in range(max_colors):
                if c in forbidden:
                    continue
                colors[v] = c
                # Quick check: just verify adjacent differ for now
                # Full validation happens later
                assigned = True
                break
            
            if not assigned:
                # Try more colors
                for c in range(max_colors, n):
                    if c not in forbidden:
                        colors[v] = c
                        assigned = True
                        break
            
            if not assigned:
                break
        
        if -1 in colors:
            continue
        
        # Verify and count distinct colors
        if not _verify_coloring(colors, n, edges):
            continue
        
        num_distinct = len(set(colors))
        
        if num_distinct < best_num_colors:
            best_num_colors = num_distinct
            best_result = list(colors)
        
        if best_num_colors == 3:
            break
        
        # Try to reduce colors via local search
        if num_distinct > 3 and time.time() < deadline - 5:
            # Try to eliminate the highest color
            current = list(colors)
            current_max = max(current)
            
            for elim_round in range(3):  # Try to eliminate up to 3 colors
                if time.time() > deadline - 2:
                    break
                
                target = max(set(current))
                verts_to_recolor = [i for i in range(n) if current[i] == target]
                
                if not verts_to_recolor:
                    break
                
                success = True
                rng2 = random.Random(restart * 1000 + elim_round)
                rng2.shuffle(verts_to_recolor)
                
                for v in verts_to_recolor:
                    recolored = False
                    for c in range(target):
                        if c == current[v]:
                            continue
                        # Check if c is forbidden by neighbors
                        if any(current[nb] == c for nb in adj_local[v]):
                            continue
                        
                        new_colors = list(current)
                        new_colors[v] = c
                        
                        if _verify_coloring(new_colors, n, edges):
                            current = new_colors
                            recolored = True
                            break
                    
                    if not recolored:
                        success = False
                        break
                
                if not success:
                    break
            
            # Check if we improved
            if current != colors:
                if _verify_coloring(current, n, edges):
                    num_distinct = len(set(current))
                    if num_distinct < best_num_colors:
                        best_num_colors = num_distinct
                        best_result = list(current)
                    if best_num_colors == 3:
                        break
    
    return best_result

def construct():
    n = N
    edges = EDGES
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s limit
    
    # First, try to find a 3-coloring with backtracking
    result = _solve_3_coloring(n, edges, deadline)
    if result is not None and _verify_coloring(result, n, edges):
        return [int(c) for c in result]
    
    # If 3-coloring failed, try greedy with local search
    result = _greedy_with_local_search(n, edges, deadline, num_restarts=500)
    if result is not None:
        return [int(c) for c in result]
    
    # Fallback: use a simple valid coloring with many colors
    # Greedy with enough colors always works
    colors = [0] * n
    for v in range(n):
        forbidden = set()
        for nb in adj[v]:
            if nb < v:
                forbidden.add(colors[nb])
        c = 0
        while c in forbidden:
            c += 1
        colors[v] = c
    
    if _verify_coloring(colors, n, edges):
        return [int(c) for c in colors]
    
    # Last resort: unique colors
    return list(range(n))
# EVOLVE-BLOCK-END
