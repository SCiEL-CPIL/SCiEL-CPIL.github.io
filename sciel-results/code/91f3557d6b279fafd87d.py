"""Rewritten program: structural analysis + backtracking with bicolored-forest pruning."""
import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adjacency(edges):
    adj = [set() for _ in range(P['n'])]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _check_bicolored_forest(coloring, adj, n):
    """Check that for every pair of colors, the induced subgraph is a forest."""
    colors_used = set(coloring)
    color_list = list(colors_used)
    m = len(color_list)
    
    def find(x, parent):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    for i in range(m):
        for j in range(i + 1, m):
            c1, c2 = color_list[i], color_list[j]
            parent = list(range(n))
            has_cycle = False
            for u in range(n):
                if coloring[u] not in (c1, c2):
                    continue
                for v in adj[u]:
                    if v <= u:
                        continue
                    if coloring[v] not in (c1, c2):
                        continue
                    ru = find(u, parent)
                    rv = find(v, parent)
                    if ru == rv:
                        has_cycle = True
                        break
                    parent[ru] = rv
                if has_cycle:
                    break
            if has_cycle:
                return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(edges)
    
    # Try k=3 first with backtracking and bicolored-forest pruning
    # Use a vertex ordering based on degree (descending) for better pruning
    
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Map: position in order -> vertex
    pos_to_vertex = order
    # For each vertex, its position in the ordering
    vertex_to_pos = [0] * n
    for pos, v in enumerate(order):
        vertex_to_pos[v] = pos
    
    # Precompute: for each vertex in order, which neighbors come later in the order
    later_neighbors = [[] for _ in range(n)]
    for pos in range(n):
        v = pos_to_vertex[pos]
        for nb in adj[v]:
            if vertex_to_pos[nb] > pos:
                later_neighbors[pos].append(nb)
    
    # Backtracking with bicolored-forest check at each step
    def backtrack(k, pos, coloring, color_count):
        """Try to assign color to vertex at position pos using k colors."""
        if time.time() > deadline - 2.0:
            return None
        
        if pos == n:
            return list(coloring)
        
        v = pos_to_vertex[pos]
        
        # Determine which colors are available (not used by already-colored neighbors)
        neighbor_colors = set()
        for nb in adj[v]:
            if vertex_to_pos[nb] < pos:
                neighbor_colors.add(coloring[nb])
        
        # Try each color
        for c in range(k):
            if c in neighbor_colors:
                continue
            
            coloring[v] = c
            
            # Check bicolored-forest constraint incrementally
            # For each pair (c, c') where c' is a color used by a colored neighbor,
            # check if adding v with color c creates a cycle in the induced subgraph
            valid = True
            for c2 in neighbor_colors:
                if c2 == c:
                    continue
                # Check if the subgraph induced by colors {c, c2} has a cycle
                # involving v
                # v has color c, and its neighbors with color c2 are...
                # We need to check if there's already a path between two of v's
                # c2-colored neighbors in the subgraph induced by {c, c2}
                
                # Find neighbors of v with color c2 that are already colored
                c2_neighbors = []
                for nb in adj[v]:
                    if vertex_to_pos[nb] < pos and coloring[nb] == c2:
                        c2_neighbors.append(nb)
                
                if len(c2_neighbors) >= 2:
                    # Check if any two c2-neighbors are connected in the {c,c2} subgraph
                    # excluding v
                    # Use BFS/DFS from one c2-neighbor to see if we can reach another
                    # using only vertices colored c or c2 (excluding v)
                    visited = set()
                    stack = [c2_neighbors[0]]
                    visited.add(c2_neighbors[0])
                    found_connection = False
                    while stack:
                        cur = stack.pop()
                        for nb2 in adj[cur]:
                            if nb2 == v:
                                continue
                            if coloring[nb2] not in (c, c2):
                                continue
                            if nb2 in visited:
                                continue
                            if nb2 in c2_neighbors:
                                found_connection = True
                                break
                            visited.add(nb2)
                            stack.append(nb2)
                        if found_connection:
                            break
                    if found_connection:
                        valid = False
                        break
            
            if not valid:
                continue
            
            result = backtrack(k, pos + 1, coloring, color_count + (1 if c not in neighbor_colors else 0))
            if result is not None:
                return result
        
        coloring[v] = -1
        return None
    
    # Try k from 3 upward
    for k in range(3, 8):
        if time.time() > deadline - 5.0:
            break
        
        coloring = [-1] * n
        result = backtrack(k, 0, coloring, 0)
        if result is not None:
            # Verify the result
            valid = True
            for u, v in edges:
                if result[u] == result[v]:
                    valid = False
                    break
            if valid and _check_bicolored_forest(result, adj, n):
                return [int(c) for c in result]
    
    # Fallback: use a simple valid coloring (n colors)
    return [int(i) for i in range(n)]


# EVOLVE-BLOCK-END
