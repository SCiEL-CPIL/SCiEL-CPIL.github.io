"""Rewritten program: backtracking search minimizing distinct colors with
incremental bicolored-cycle detection via union-find."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

# Precompute vertex ordering by degree descending (more constrained first)
# Actually for backtracking, we want to assign in an order that prunes early.
# Let's use a simple heuristic: order vertices by degree descending.
degree = [len(adj[i]) for i in range(N)]
order = sorted(range(N), key=lambda x: -degree[x])
# Map: position in order -> vertex
pos_of = [0] * N
for i, v in enumerate(order):
    pos_of[v] = i

# For incremental union-find per color pair, we need to track edges between same-color-pair
# We'll use a different approach: for each color pair (a,b), maintain a union-find
# over the vertices that have colors a or b, and check if adding an edge creates a cycle.

# Actually, a simpler approach for small n: after full assignment, check all color pairs.
# But for backtracking pruning, we need incremental checks.

# Let's use a smarter approach: 
# For each pair of colors (c1, c2), maintain a DSU. When we assign a vertex a color,
# we check all edges from that vertex to already-colored neighbors. For each such edge
# to a neighbor with color c2 (where c2 != c1), we union the vertex and neighbor in DSU[c1,c2].
# If union detects a cycle (they were already in the same set), then we have a bicolored cycle.

# We'll implement this with a dictionary of DSUs keyed by frozenset({c1,c2}).

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # cycle detected
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s margin
    
    best_result = None
    best_colors = N  # start with trivial solution
    
    # Trivial valid solution: all different colors
    trivial = list(range(N))
    best_result = trivial[:]
    best_colors = N
    
    # Try to find solutions with fewer colors
    # We'll do iterative deepening on the number of colors
    for max_colors in range(2, N):
        if time.time() > deadline:
            break
        
        # Backtracking search
        color = [-1] * N  # color[v] = assigned color or -1
        # DSU for each color pair
        dsus = {}  # frozenset({c1,c2}) -> DSU
        
        # We need to be able to undo DSU operations for backtracking.
        # This is complex. Alternative: just check at the end, but prune with a simpler check.
        
        # Simpler approach: use a bipartite check per color pair incrementally.
        # For each color pair, maintain the subgraph and check if it's a forest.
        # A graph is a forest iff edges = vertices - components, and no cycle.
        # We can maintain for each color pair: number of edges, and use DSU with rollback.
        
        # Let's use a different strategy: assign colors, and at each step, 
        # check if the current partial assignment creates a bicolored cycle.
        # We do this by, for the newly assigned vertex v with color c,
        # for each neighbor u already colored with color c2 != c,
        # check if adding edge (u,v) to the subgraph of colors {c,c2} creates a cycle.
        # This requires knowing if u and v are already connected in that subgraph.
        
        # We'll maintain DSUs with a rollback mechanism.
        # For simplicity, let's just do a BFS/DFS check for each affected color pair.
        # Since degree is 4, this is fast.
        
        def has_bicolor_cycle(v, c):
            """Check if assigning color c to vertex v creates a bicolored cycle."""
            for u in adj[v]:
                cu = color[u]
                if cu == -1 or cu == c:
                    continue
                # Check subgraph induced by colors {c, cu}
                # Do a BFS from v in this subgraph to see if we can reach u
                # without using the edge (v,u) directly, or more precisely,
                # check if v and u are in the same connected component of the
                # subgraph of {c,cu} before adding edge (v,u).
                # If they are, adding the edge creates a cycle.
                
                # BFS from v in subgraph of colors {c, cu}, not using edge to u
                visited = set()
                stack = [v]
                visited.add(v)
                while stack:
                    cur = stack.pop()
                    for w in adj[cur]:
                        if w == u and cur == v:
                            continue  # skip the direct edge we're about to add
                        cw = color[w]
                        if cw not in (c, cu):
                            continue
                        if w not in visited:
                            visited.add(w)
                            stack.append(w)
                if u in visited:
                    return True
            return False
        
        # Order of assignment: use the precomputed order
        # But we need to track which vertices are unassigned
        # Let's assign in the order defined by 'order'
        
        def backtrack(idx):
            nonlocal best_result, best_colors
            if time.time() > deadline:
                return False
            if idx == N:
                # Found a valid coloring with max_colors colors
                # Count distinct colors used
                distinct = len(set(color))
                if distinct < best_colors:
                    best_colors = distinct
                    best_result = color[:]
                return True
            
            v = order[idx]
            # Try colors 0..max_colors-1
            for c in range(max_colors):
                # Check adjacency constraint
                conflict = False
                for u in adj[v]:
                    if color[u] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Check bicolored cycle
                if has_bicolor_cycle(v, c):
                    continue
                
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
            
            return False
        
        if backtrack(0):
            if best_colors <= max_colors:
                # We found a solution with best_colors colors
                # Try to see if we can do better
                if best_colors < max_colors:
                    break
                # Otherwise continue to see if fewer colors work
                # Actually, we're iterating max_colors from 2 upward,
                # so if we found a solution with max_colors, we should try max_colors-1?
                # No, we're going up. Let's go down instead.
                # Actually, let's restart: try max_colors from 2 up, and stop when found.
                # The first valid max_colors gives us the answer.
                break
        else:
            # No solution with max_colors colors, try more
            continue
    
    # Ensure we return a valid list of Python ints
    if best_result is None:
        best_result = trivial
    
    # Convert to ensure Python ints
    result = [int(c) for c in best_result]
    return result
# EVOLVE-BLOCK-END
