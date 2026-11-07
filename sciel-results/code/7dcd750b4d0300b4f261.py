"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(100000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Precompute adjacency
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

# Precompute edge list as list of (u, v)
E = [(u, v) for u, v in EDGES]

# We need a coloring where:
# 1. Adjacent vertices have different colors.
# 2. For any pair of colors (c1, c2), the subgraph induced by vertices colored c1 or c2 is a forest.

# Strategy: Backtracking search with increasing number of colors.
# Start with k=3, then k=4, etc.
# For each k, try to find a valid coloring.

# To check the forest condition efficiently during backtracking:
# When we assign a color to a vertex, we need to check that for every other color c',
# the subgraph induced by {current_color, c'} remains a forest.
# This is expensive to check fully each time.

# Alternative: Use union-find incrementally.
# For each pair of colors, maintain a union-find structure.
# When we assign color c to vertex v:
#   For each neighbor u of v with color d:
#     Check if adding edge (u,v) to the (c,d) subgraph creates a cycle.
#     i.e., find(c) == find(d) in the union-find for pair (c,d).
#   Also, for each other vertex w with color c (same color as v):
#     Wait, same color doesn't form edges in the bicolored subgraph unless... 
#     Actually, the subgraph induced by colors {c1, c2} includes ALL edges where both endpoints have color in {c1, c2}.
#     So edges between two vertices of color c1 are included.
#     So when we assign color c to v, we need to check:
#     - For each neighbor u with color d != c: edge (u,v) is in subgraph (c,d). Check cycle.
#     - For each other vertex w with color c that is adjacent to v: edge (w,v) is in subgraph (c,c)... wait, no.
#       The subgraph induced by colors {c1, c2} where c1 != c2 includes edges where both endpoints are in {c1, c2}.
#       If both endpoints have color c1, the edge is in the subgraph for pair (c1, c2) only if c2 is also one of the colors.
#       Actually, re-reading: "the subgraph induced by any two colors must be a forest."
#       This means for any distinct pair of colors {c1, c2}, consider the subgraph containing only vertices with color c1 or c2, and all edges between them. This must be a forest.
#       Edges between two vertices of color c1 are included in this subgraph.
#       So when assigning color c to v, for each pair (c, d) where d is a color used by some neighbor or... actually d can be any other color.
#       But we only need to check pairs where there's a potential new edge.
#       New edges involving v:
#       - (v, u) where u is a neighbor. If color(u) = d, then edge (v,u) is in subgraph for pair (c, d) if c != d, and in pair (c, c) if c == d.
#       Wait, if c == d, then the pair is (c, c)? No, "any two colors" means two distinct colors.
#       Let me re-read: "the subgraph induced by any two colors must be a forest."
#       I think this means: pick any two distinct colors c1, c2. The subgraph induced by vertices with color c1 or c2 must be a forest.
#       So edges between two c1 vertices ARE included.
#       So when we assign color c to v:
#       For each neighbor u:
#         d = color(u)
#         If d == c: edge (v,u) is in subgraph for any pair (c, x) where x is any other color. But that's complicated.
#         Actually, edge (v,u) with both having color c is in the subgraph for pair (c, d') for ANY d' != c.
#         So we need to check: for the pair (c, d') where d' is any color, does adding this edge create a cycle?
#         But the edge (v,u) is only in the subgraph if both v and u have colors in {c, d'}. Since both have color c, they are in {c, d'} for any d'.
#         So we need to check all pairs (c, d') for d' != c.
#         This is expensive.

# Let me reconsider. Maybe the problem means: for any two colors c1, c2, the subgraph consisting of edges where one endpoint has color c1 and the other has color c2 must be a forest. I.e., only edges BETWEEN the two color classes.
# "the subgraph induced by any two colors" - in graph theory, the subgraph induced by a set of vertices S is the subgraph containing all vertices in S and all edges with both endpoints in S.
# So if S = {vertices with color c1} ∪ {vertices with color c2}, then all edges within this set are included, including edges within c1-class and within c2-class.

# This makes the problem much harder. Let me assume the stricter interpretation.

# Given the complexity, let's try a different approach:
# Use a small number of colors and verify at the end.
# Try k=3, k=4, k=5 with backtracking.

# For efficiency, let's use a smarter backtracking:
# Order vertices by degree (highest first).
# When assigning a color, check constraints with already-colored neighbors.

# Let me implement a backtracking solver with pruning.

def is_forest_incremental(color_assign, v, c, adj, N):
    """
    Check if assigning color c to vertex v (given current color_assign for other vertices)
    maintains the forest property for all relevant pairs.
    color_assign[v] is None or the new color c.
    We need to check all pairs (c, d) for d in used colors.
    """
    # This is too slow to do naively. Let's just do full verification at the end for small k.
    pass

def full_verify(colors, adj, N, E):
    """Verify the coloring satisfies all constraints."""
    # 1. Adjacent vertices must differ
    for u, v in E:
        if colors[u] == colors[v]:
            return False
    
    # 2. For each pair of colors, the induced subgraph must be a forest
    used_colors = set(colors)
    for c1 in used_colors:
        for c2 in used_colors:
            if c1 >= c2:
                continue
            # Check subgraph induced by vertices with color c1 or c2
            # Build a DSU for this subgraph
            parent = list(range(N))
            rank = [0] * N
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False  # cycle
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for u, v in E:
                if colors[u] in (c1, c2) and colors[v] in (c1, c2):
                    if not union(u, v):
                        return False  # cycle found
    return True

# Backtracking search
def solve():
    deadline = time.time() + 80.0  # Safety margin
    
    # Order vertices by degree descending
    deg = [len(adj[i]) for i in range(N)]
    order = sorted(range(N), key=lambda i: -deg[i])
    
    # Map: position in order -> vertex
    # We'll assign colors in this order.
    
    def backtrack(pos, colors, used_colors):
        if time.time() > deadline:
            return None
        
        if pos == N:
            return colors[:]
        
        v = order[pos]
        
        # Try each color
        for c in range(50):  # Try colors 0..49
            # Check adjacency constraint
            valid = True
            for u in adj[v]:
                if colors[u] is not None and colors[u] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Assign
            colors[v] = c
            new_used = used_colors | {c}
            
            result = backtrack(pos + 1, colors, new_used)
            if result is not None:
                return result
            
            colors[v] = None
        
        return None
    
    # Try with increasing number of colors
    for k in range(3, 50):
        # Modify backtracking to only use colors 0..k-1
        def backtrack_k(pos, colors):
            if time.time() > deadline:
                return None
            
            if pos == N:
                return colors[:]
            
            v = order[pos]
            
            # Try each color from 0 to k-1
            for c in range(k):
                # Check adjacency constraint
                valid = True
                for u in adj[v]:
                    if colors[u] is not None and colors[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign
                colors[v] = c
                
                result = backtrack_k(pos + 1, colors)
                if result is not None:
                    return result
                
                colors[v] = None
            
            return None
        
        colors = [None] * N
        result = backtrack_k(0, colors)
        if result is not None:
            # Verify
            if full_verify(result, adj, N, E):
                return result
    
    return None

def construct():
    result = solve()
    if result is None:
        # Fallback: return a valid coloring with many colors
        # The identity coloring (each vertex unique color) is always valid
        # since no two adjacent vertices share a color, and any pair of colors
        # induces at most one edge (or a star), which is a forest.
        return [int(i) for i in range(N)]
    return [int(c) for c in result]
# EVOLVE-BLOCK-END
