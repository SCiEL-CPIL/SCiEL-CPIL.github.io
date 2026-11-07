"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_bicolored_cycles(colors, edges):
    """
    Check if the subgraph induced by any pair of colors is a forest.
    A forest has no cycles.
    We use Union-Find to detect cycles in the subgraph for each color pair.
    """
    n = len(colors)
    # Map color to list of vertices
    color_to_verts = {}
    for i, c in enumerate(colors):
        if c not in color_to_verts:
            color_to_verts[c] = []
        color_to_verts[c].append(i)
    
    # Get all unique colors
    unique_colors = sorted(color_to_verts.keys())
    k = len(unique_colors)
    
    # For each pair of colors, check for cycles
    # Actually, the condition is: "the subgraph induced by any two colors must be a forest".
    # This means if we take all vertices with color A or color B, the edges between them must not form a cycle.
    # Note: Edges between A and A or B and B are not possible because adjacent vertices must differ.
    # So the induced subgraph on {A, B} consists only of edges between A and B.
    # This is a bipartite graph. A bipartite graph is a forest if and only if it has no cycles.
    
    # Optimization: We don't need to check all pairs if we are just minimizing colors.
    # But we must ensure validity.
    
    # Union-Find implementation
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False # Cycle found
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    # We need to check for each pair of colors (c1, c2)
    # However, doing this naively for every pair in every check is O(K^2 * E).
    # For small K (2,3,4) and E=100, this is fine.
    
    for i in range(k):
        for j in range(i+1, k):
            c1 = unique_colors[i]
            c2 = unique_colors[j]
            
            # Reset Union-Find for this specific subgraph check
            # Actually, we can just check if any edge connects a vertex in c1 to a vertex in c2
            # and if that forms a cycle within the subgraph of vertices {v | color[v] in {c1, c2}}
            
            # To be efficient, we can just run Union-Find on the subset of edges
            # where both endpoints have colors in {c1, c2}.
            
            # Since we are re-using the global parent array, we must reset it or use a local one.
            # Given n=50, creating a new local parent list is cheap.
            local_parent = list(range(n))
            local_rank = [0] * n
            
            def local_find(x):
                while local_parent[x] != x:
                    local_parent[x] = local_parent[local_parent[x]]
                    x = local_parent[x]
                return x
            
            def local_union(x, y):
                px, py = local_find(x), local_find(y)
                if px == py:
                    return False
                if local_rank[px] < local_rank[py]:
                    px, py = py, px
                local_parent[py] = px
                if local_rank[px] == local_rank[py]:
                    local_rank[px] += 1
                return True

            for u, v in edges:
                cu, cv = colors[u], colors[v]
                # Only consider edges where both endpoints are in the pair {c1, c2}
                # Since adjacent vertices have different colors, an edge (u,v) contributes to the subgraph of (c1, c2)
                # if {cu, cv} == {c1, c2}.
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not local_union(u, v):
                        return False # Cycle found
    
    return True

def _try_coloring(max_colors):
    """
    Try to find a valid coloring using backtracking with max_colors.
    Returns the coloring list if found, else None.
    """
    n = P['n']
    edges = P['edges']
    colors = [-1] * n
    
    # Pre-compute neighbors for fast constraint checking
    neighbors = [[] for _ in range(n)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
        
    # Order vertices by degree (descending) to prune earlier
    # But we must maintain the vertex indices for the final output.
    # We can just iterate 0 to n-1, n is small.
    
    def backtrack(idx):
        if idx == n:
            # Check for bicolored cycles
            if _check_bicolored_cycles(colors, edges):
                return True
            return False
            
        # Heuristic: Try colors that have been used less? Or just 0..max_colors-1
        for c in range(max_colors):
            # Check if color c is allowed for vertex idx
            valid = True
            for nb in neighbors[idx]:
                if colors[nb] == c:
                    valid = False
                    break
            if valid:
                colors[idx] = c
                if backtrack(idx + 1):
                    return True
                colors[idx] = -1
        return False

    if backtrack(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    # Try 2 colors
    if time.time() < deadline:
        res = _try_coloring(2)
        if res is not None:
            return res
    
    # Try 3 colors
    if time.time() < deadline:
        res = _try_coloring(3)
        if res is not None:
            return res
            
    # Try 4 colors
    if time.time() < deadline:
        res = _try_coloring(4)
        if res is not None:
            return res
            
    # Fallback: 5 colors (should almost certainly work for planar-ish graphs)
    if time.time() < deadline:
        res = _try_coloring(5)
        if res is not None:
            return res
            
    # Emergency fallback: Identity (50 colors)
    return list(range(P['n']))
# EVOLVE-BLOCK-END
