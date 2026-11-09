"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_connected(adj, color_a, color_b, target, start, colored):
    """Check if target is reachable from start using only vertices with color_a or color_b."""
    if start == target:
        return True
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if nb in visited:
                continue
            if colored[nb] == color_a or colored[nb] == color_b:
                if nb == target:
                    return True
                visited.add(nb)
                stack.append(nb)
    return False

def _is_valid_assignment(adj, colored, vertex, color, max_color):
    """Check if assigning 'color' to 'vertex' is valid:
    1. No neighbor has the same color.
    2. The two-color subgraph (color + some other color) doesn't form a cycle.
    Actually, the constraint is: for ANY pair of colors, the induced subgraph is a forest.
    When we assign 'color' to 'vertex', we only potentially create a cycle involving 'color'
    and some other color c'. A cycle would form if two neighbors of 'vertex' that have color c'
    are already connected in the subgraph induced by {color, c'}.
    """
    # Check 1: no adjacent vertex has the same color
    for nb in adj[vertex]:
        if colored[nb] == color:
            return False
    
    # Check 2: for each other color c' present in neighbors, check if any two neighbors
    # with color c' are connected in the {color, c'} subgraph (without vertex)
    # Actually, vertex is being assigned 'color', so in the {color, c'} subgraph,
    # vertex connects to all its neighbors that have color c'.
    # A cycle forms if there exist two neighbors nb1, nb2 of vertex both with color c'
    # such that nb1 and nb2 are connected in the subgraph induced by {color, c'} minus vertex.
    # But since vertex is new with color 'color', the only new edges are vertex-nb for nb with color c'.
    # A cycle would be: vertex -> nb1 -> ... -> nb2 -> vertex, where the path from nb1 to nb2
    # uses only vertices with color in {color, c'} and doesn't go through vertex.
    
    # Collect neighbors by their color (excluding vertex's new color assignment)
    neighbor_colors = {}
    for nb in adj[vertex]:
        c = colored[nb]
        if c != color:  # they can't have 'color' due to check 1
            if c not in neighbor_colors:
                neighbor_colors[c] = []
            neighbor_colors[c].append(nb)
    
    for c, nbs in neighbor_colors.items():
        if len(nbs) < 2:
            continue
        # Check if any pair in nbs is connected in {color, c} subgraph (excluding vertex)
        # Since vertex isn't colored yet in the subgraph, we check connectivity among nbs
        # using only vertices with color in {color, c}, not including vertex.
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                if _bfs_connected(adj, color, c, nbs[j], nbs[i], colored):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    adj = _build_adjacency()
    n = P['n']
    
    # Try to find a valid coloring with minimum number of colors
    # Start with 2 colors, then 3, etc.
    for max_colors in range(2, 10):
        colored = [-1] * n
        # Use backtracking with a good vertex ordering
        # Order vertices by degree (all are 4 here), then by index
        order = list(range(n))
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            v = order[idx]
            for c in range(max_colors):
                if _is_valid_assignment(adj, colored, v, c, max_colors):
                    colored[v] = c
                    if backtrack(idx + 1):
                        return True
                    colored[v] = -1
            return False
        
        if backtrack(0):
            result = [int(c) for c in colored]
            return result
    
    # Fallback: use all distinct colors
    return list(range(n))

# EVOLVE-BLOCK-END
