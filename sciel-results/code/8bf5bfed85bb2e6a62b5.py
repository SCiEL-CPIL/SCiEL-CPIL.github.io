import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
edges = P['edges']
adj = defaultdict(list)
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# Order vertices by descending degree (all degree 4 here, so use index as tiebreak)
# But we can use a smarter ordering: process vertices in an order that maximizes constraints early
# For this graph, let's just use natural order or reverse
vertex_order = list(range(N))

def check_validity(colors):
    """
    Returns True if coloring is proper and all bicolored subgraphs are forests.
    """
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    
    used_colors = set(colors)
    if len(used_colors) < 2:
        return True
        
    color_list = list(used_colors)
    for i in range(len(color_list)):
        for j in range(i + 1, len(color_list)):
            a, b = color_list[i], color_list[j]
            parent = list(range(N))
            rank = [0] * N
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                px = find(x)
                py = find(y)
                if px == py:
                    return False
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1
                return True
            
            for u, v in edges:
                cu, cv = colors[u], colors[v]
                if (cu == a and cv == b) or (cu == b and cv == a):
                    if not union(u, v):
                        return False
    return True

def backtrack(K, time_limit):
    """
    Backtracking search with forward checking for bicolored cycles.
    We assign colors to vertices in a fixed order.
    For each assignment, we check:
      1. Properness: no neighbor has the same color
      2. Bicolored cycle: for each neighbor with a different color, check if adding
         this edge would create a cycle in the bicolored subgraph for that color pair.
    
    We maintain per-pair union-find structures incrementally.
    """
    start_time = time.time()
    colors = [-1] * N
    
    # For each color pair (a,b) with a<b, maintain a union-find
    # pair_uf[(a,b)] = (parent, rank, edge_count)
    # We'll use a dict mapping (a,b) -> [parent_list, rank_list]
    # But copying is expensive. Instead, we'll track the bicolored edges per pair
    # and check connectivity using a lightweight approach.
    
    # Alternative: maintain for each pair (a,b) the set of edges in that bicolored subgraph
    # and check if adding a new edge creates a cycle via BFS/DFS on the current subgraph.
    # Since the subgraphs are small (at most 50 vertices), this is feasible.
    
    # Better: maintain union-find with path compression, and use undo stack for backtracking.
    
    # For simplicity and correctness, let's use a different approach:
    # Maintain for each pair (a,b) a list of edges. When checking if adding edge (u,v)
    # creates a cycle, do a BFS from u to v in the existing (a,b)-bicolored subgraph.
    
    pair_edges = {}  # (a,b) -> list of (u,v) edges
    
    def would_create_cycle(a, b, u, v):
        """Check if adding edge (u,v) to the (a,b)-bicolored subgraph creates a cycle."""
        if a > b:
            a, b = b, a
        key = (a, b)
        if key not in pair_edges:
            return False
        # BFS from u to v in the existing subgraph
        # Build adjacency for this pair
        visited = {u}
        stack = [u]
        while stack:
            node = stack.pop()
            if node == v:
                return True  # Already connected, adding edge creates cycle
            for (x, y) in pair_edges[key]:
                if x == node and y != v:
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)
                elif y == node and x != v:
                    if x not in visited:
                        visited.add(x)
                        stack.append(x)
        return False
    
    def assign(v, c):
        """Assign color c to vertex v. Returns True if valid."""
        # Check properness
        for u in adj[v]:
            if colors[u] == c:
                return False
        # Check bicolored cycles
        for u in adj[v]:
            cu = colors[u]
            if cu != -1 and cu != c:
                if would_create_cycle(c, cu, v, u):
                    return False
        # Commit
        colors[v] = c
        for u in adj[v]:
            cu = colors[u]
            if cu != -1 and cu != c:
                a, b = (c, cu) if c < cu else (cu, c)
                key = (a, b)
                if key not in pair_edges:
                    pair_edges[key] = []
                pair_edges[key].append((v, u))
        return True
    
    def unassign(v):
        """Remove color assignment from vertex v."""
        c = colors[v]
        if c == -1:
            return
        for u in adj[v]:
            cu = colors[u]
            if cu != -1 and cu != c:
                a, b = (c, cu) if c < cu else (cu, c)
                key = (a, b)
                if key in pair_edges:
                    # Remove the edge (v, u)
                    pair_edges[key].remove((v, u))
                    if not pair_edges[key]:
                        del pair_edges[key]
        colors[v] = -1
    
    def solve(idx):
        """Try to assign colors to vertex_order[idx:]."""
        if time.time() - start_time > time_limit:
            return False
        if idx == N:
            return True
        
        v = vertex_order[idx]
        
        # Try colors 0..K-1
        for c in range(K):
            if assign(v, c):
                if solve(idx + 1):
                    return True
                unassign(v)
        
        return False
    
    if solve(0):
        return list(colors)
    return None

def construct():
    deadline = time.time() + 85.0
    
    # Try K=3 first
    remaining = deadline - time.time()
    if remaining > 10:
        res = backtrack(3, min(30.0, remaining - 5.0))
        if res is not None and check_validity(res):
            return res
    
    # Try K=4
    remaining = deadline - time.time()
    if remaining > 10:
        res = backtrack(4, min(30.0, remaining - 5.0))
        if res is not None and check_validity(res):
            return res
    
    # Try K=5
    remaining = deadline - time.time()
    if remaining > 5:
        res = backtrack(5, min(20.0, remaining - 3.0))
        if res is not None and check_validity(res):
            return res
    
    # Fallback: use all distinct colors
    return [int(i) for i in range(N)]

# EVOLVE-BLOCK-END
