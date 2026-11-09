"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try a 4-coloring based on grid position
    # The graph appears to be a 7x7 grid with vertex 49
    # Color by (row + col) % 4
    coloring = []
    for i in range(n):
        if i < 49:
            row = i // 7
            col = i % 7
            coloring.append((row + col) % 4)
        else:
            # Vertex 49 is connected to 0, 6, 42, 48
            # Neighbors: 0(color 0), 6(color 6%4=2), 42(color (6+0)%4=2), 48(color (6+6)%4=12%4=0)
            # So neighbors have colors 0 and 2. Use color 1 or 3.
            coloring.append(1)
    
    # Verify the coloring
    def is_valid(coloring):
        # Check proper coloring
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Check bicolored subgraphs are forests
        colors_used = set(coloring)
        for c1 in colors_used:
            for c2 in colors_used:
                if c1 >= c2:
                    continue
                # Get vertices with color c1 or c2
                vset = set()
                for i in range(n):
                    if coloring[i] in (c1, c2):
                        vset.add(i)
                if len(vset) < 3:
                    continue
                # Build subgraph and check for cycles using union-find
                parent = {v: v for v in vset}
                rank = {v: 0 for v in vset}
                
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                
                def union(x, y):
                    px, py = find(x), find(y)
                    if px == py:
                        return False
                    if rank[px] < rank[py]:
                        px, py = py, px
                    parent[py] = px
                    if rank[px] == rank[py]:
                        rank[px] += 1
                    return True
                
                has_cycle = False
                for v in vset:
                    for w in adj[v]:
                        if w in vset and v < w:
                            if not union(v, w):
                                has_cycle = True
                                break
                    if has_cycle:
                        break
                if has_cycle:
                    return False
        return True
    
    if is_valid(coloring):
        return [int(c) for c in coloring]
    
    # Fallback: try other 4-colorings
    for offset in range(4):
        coloring = []
        for i in range(n):
            if i < 49:
                row = i // 7
                col = i % 7
                coloring.append((row + col + offset) % 4)
            else:
                # Find a color not used by neighbors
                neighbor_colors = set()
                for w in adj[49]:
                    neighbor_colors.add(coloring[w])
                for c in range(4):
                    if c not in neighbor_colors:
                        coloring.append(c)
                        break
                else:
                    coloring.append(0)
        
        if is_valid(coloring):
            return [int(c) for c in coloring]
    
    # Fallback: 5 colors
    for offset in range(5):
        coloring = []
        for i in range(n):
            if i < 49:
                row = i // 7
                col = i % 7
                coloring.append((row + col + offset) % 5)
            else:
                neighbor_colors = set()
                for w in adj[49]:
                    neighbor_colors.add(coloring[w])
                for c in range(5):
                    if c not in neighbor_colors:
                        coloring.append(c)
                        break
                else:
                    coloring.append(0)
        
        if is_valid(coloring):
            return [int(c) for c in coloring]
    
    # Final fallback: distinct colors
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
