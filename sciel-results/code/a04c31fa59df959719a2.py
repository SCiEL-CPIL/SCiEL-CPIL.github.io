"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # Helper: Check if assigning color 'c' to vertex 'v' is valid given current coloring
    # We only need to check constraints involving 'v' and its neighbors.
    # Specifically, for each neighbor u with color cu != c, the edge (v,u) must not create a cycle in G[{c, cu}].
    # This is equivalent to checking if v's neighbors of color cu are in distinct components of G[{c, cu}] (excluding v).
    # Since v is not yet in the graph for this check, we check if any two neighbors of v with color cu are connected in G[{c, cu}].
    
    def is_assignment_valid(v, c, coloring):
        # 1. Proper coloring check
        for u in adj[v]:
            if coloring[u] == c:
                return False
        
        # 2. Bicolored forest check
        # Group neighbors by their color
        neighbors_by_color = {}
        for u in adj[v]:
            cu = coloring[u]
            if cu == -1:
                continue
            if cu == c:
                return False # Already checked, but safe
            if cu not in neighbors_by_color:
                neighbors_by_color[cu] = []
            neighbors_by_color[cu].append(u)
        
        for cu, neighs in neighbors_by_color.items():
            # Check if adding v (color c) and edges to neighs (color cu) creates a cycle in G[{c, cu}]
            # This happens if any two nodes in 'neighs' are connected in the current G[{c, cu}] subgraph.
            # Also, if a node in 'neighs' is connected to another node in 'neighs' via a path that goes through other c/cu vertices.
            
            # We perform a BFS/DFS on the subgraph induced by colors {c, cu}
            # to see if any two nodes in 'neighs' are in the same component.
            # Note: v is NOT in this subgraph yet.
            
            if len(neighs) <= 1:
                continue # No cycle possible with just one edge
            
            # BFS from neighs[0]
            start = neighs[0]
            target_set = set(neighs[1:])
            
            visited = {start}
            queue = [start]
            found_connection = False
            
            while queue:
                curr = queue.pop(0)
                for nb in adj[curr]:
                    # Only traverse if nb is in subgraph {c, cu}
                    if coloring[nb] == c or coloring[nb] == cu:
                        if nb in target_set:
                            found_connection = True
                            break
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
                if found_connection:
                    break
            
            if found_connection:
                return False
        
        return True

    # Backtracking search
    def backtrack(idx, coloring, max_color):
        if idx == n:
            return True
        
        # Heuristic: Try colors that are already used first? Or just 0..max_color
        # Simple order: 0 to max_color
        for c in range(max_color + 1):
            if is_assignment_valid(idx, c, coloring):
                coloring[idx] = c
                if backtrack(idx + 1, coloring, max_color):
                    return True
                coloring[idx] = -1
        
        return False

    # Try increasing number of colors
    # 2 colors is impossible for this graph (has cycles)
    for num_colors in range(3, 10):
        coloring = [-1] * n
        if backtrack(0, coloring, num_colors - 1):
            return [int(c) for c in coloring]
    
    # Fallback
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
