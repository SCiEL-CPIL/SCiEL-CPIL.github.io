"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80.0
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best_colors = None
    best_num = n
    
    def has_bicolor_cycle(colors, c1, c2):
        """Check if the subgraph induced by vertices with color c1 or c2 has a cycle."""
        # Collect vertices with these two colors
        verts = [v for v in range(n) if colors[v] in (c1, c2)]
        if len(verts) < 3:
            return False
        # Build adjacency within this subgraph
        vert_set = set(verts)
        # Use BFS to detect cycle
        visited = set()
        parent = {}
        for start in verts:
            if start in visited:
                continue
            visited.add(start)
            parent[start] = -1
            queue = [start]
            while queue:
                cur = queue.pop(0)
                for nb in adj[cur]:
                    if nb not in vert_set:
                        continue
                    if nb not in visited:
                        visited.add(nb)
                        parent[nb] = cur
                        queue.append(nb)
                    elif parent[cur] != nb and parent[nb] != cur:
                        return True
        return False
    
    def is_valid_coloring(colors):
        """Check all constraints."""
        # Adjacent vertices must differ
        for u, v in edges:
            if colors[u] == colors[v]:
                return False
        # Check bicolored cycles for all pairs of colors used
        used_colors = set(colors)
        color_list = list(used_colors)
        for i in range(len(color_list)):
            for j in range(i + 1, len(color_list)):
                if has_bicolor_cycle(colors, color_list[i], color_list[j]):
                    return False
        return True
    
    # Try greedy with randomization
    for max_colors in range(2, n + 1):
        if time.time() > deadline - 5:
            break
        
        for trial in range(500):
            if time.time() > deadline - 5:
                break
            
            colors = [-1] * n
            order = list(range(n))
            random.shuffle(order)
            
            success = True
            for v in order:
                used_by_neighbors = set()
                for nb in adj[v]:
                    if colors[nb] != -1:
                        used_by_neighbors.add(colors[nb])
                
                available = [c for c in range(max_colors) if c not in used_by_neighbors]
                if not available:
                    success = False
                    break
                
                # Try each available color, check if it would create a bicolored cycle
                valid = []
                for c in available:
                    # Tentatively assign
                    colors[v] = c
                    # Check if this creates any bicolored cycle involving color c
                    ok = True
                    for cn in used_by_neighbors:
                        if has_bicolor_cycle(colors, c, cn):
                            ok = False
                            break
                    if ok:
                        valid.append(c)
                    colors[v] = -1
                
                if not valid:
                    success = False
                    break
                
                colors[v] = random.choice(valid)
            
            if success:
                if is_valid_coloring(colors):
                    num_distinct = len(set(colors))
                    if num_distinct < best_num:
                        best_num = num_distinct
                        best_colors = list(colors)
                    if best_num <= 3:
                        break
        
        if best_colors is not None and best_num <= max_colors:
            break
    
    if best_colors is None:
        return list(range(n))
    
    return [int(c) for c in best_colors]


# EVOLVE-BLOCK-END
