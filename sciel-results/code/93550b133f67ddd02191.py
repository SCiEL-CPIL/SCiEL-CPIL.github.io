"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    def check_valid(coloring):
        """Check if coloring is valid: proper and no bicolored cycles."""
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Check no bicolored cycles using union-find
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2:
                    continue
                # Union-find on vertices with color c1 or c2
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
                        return True  # cycle detected
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return False
                for u, v in edges:
                    if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                        if coloring[u] != coloring[v]:  # edge between different colors in this pair
                            if union(u, v):
                                return False  # bicolored cycle
        return True
    
    def count_bicolored_cycles(coloring):
        """Count number of bicolored cycles (for optimization)."""
        count = 0
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2:
                    continue
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
                        return True
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return False
                for u, v in edges:
                    if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                        if coloring[u] != coloring[v]:
                            if union(u, v):
                                count += 1
                                break  # one cycle is enough to know it's invalid
        return count
    
    # Try to find a coloring with few colors
    # Start with greedy coloring
    def greedy_coloring(max_colors):
        coloring = [-1] * n
        for i in range(n):
            neighbor_colors = set()
            for nb in adj[i]:
                if coloring[nb] != -1:
                    neighbor_colors.add(coloring[nb])
            for c in range(max_colors):
                if c not in neighbor_colors:
                    coloring[i] = c
                    break
            if coloring[i] == -1:
                coloring[i] = 0  # fallback
        return coloring
    
    # Local search to fix bicolored cycles
    def fix_bicolored_cycles(coloring, max_colors, iterations=10000):
        coloring = coloring[:]
        for it in range(iterations):
            # Find a bicolored cycle
            colors_used = set(coloring)
            cycle_found = False
            for c1 in colors_used:
                for c2 in colors_used:
                    if c1 >= c2:
                        continue
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
                            return True
                        if rank[px] < rank[py]:
                            px, py = py, px
                        parent[py] = px
                        if rank[px] == rank[py]:
                            rank[px] += 1
                        return False
                    cycle_vertices = []
                    for u, v in edges:
                        if coloring[u] in (c1, c2) and coloring[v] in (c1, c2):
                            if coloring[u] != coloring[v]:
                                if union(u, v):
                                    # Found a cycle, pick a random vertex in this bicolored subgraph
                                    bicolored_verts = [x for x in range(n) if coloring[x] in (c1, c2)]
                                    if bicolored_verts:
                                        v_to_recolor = random.choice(bicolored_verts)
                                        # Try to recolor to a different color
                                        neighbor_colors = set()
                                        for nb in adj[v_to_recolor]:
                                            neighbor_colors.add(coloring[nb])
                                        for new_c in range(max_colors):
                                            if new_c not in neighbor_colors and new_c != coloring[v_to_recolor]:
                                                coloring[v_to_recolor] = new_c
                                                cycle_found = True
                                                break
                                        if not cycle_found:
                                            # Try a new color
                                            coloring[v_to_recolor] = max_colors
                                            max_colors += 1
                                            cycle_found = True
                                    break
                if cycle_found:
                    break
            if not cycle_found:
                break
        return coloring, max_colors
    
    # Try different numbers of colors
    best_coloring = list(range(n))  # fallback: 50 colors
    best_num_colors = n
    
    for k in range(3, n + 1):
        random.seed(42)
        coloring = greedy_coloring(k)
        coloring, k_used = fix_bicolored_cycles(coloring, k, iterations=5000)
        if check_valid(coloring):
            num_colors = len(set(coloring))
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = coloring[:]
    
    # Try to reduce colors by merging
    def try_reduce_colors(coloring):
        coloring = coloring[:]
        colors_used = sorted(set(coloring))
        changed = True
        while changed:
            changed = False
            for c1 in colors_used:
                for c2 in colors_used:
                    if c1 >= c2:
                        continue
                    # Try to merge c2 into c1
                    new_coloring = coloring[:]
                    for i in range(n):
                        if new_coloring[i] == c2:
                            new_coloring[i] = c1
                    if check_valid(new_coloring):
                        coloring = new_coloring
                        colors_used = sorted(set(coloring))
                        changed = True
                        break
                if changed:
                    break
        return coloring
    
    best_coloring = try_reduce_colors(best_coloring)
    
    # Ensure proper Python ints in a list
    result = [int(c) for c in best_coloring]
    return result

# EVOLVE-BLOCK-END
