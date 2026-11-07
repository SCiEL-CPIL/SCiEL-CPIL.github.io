"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    adj = [[] for _ in range(P['n'])]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_cycles(coloring, adj, max_colors):
    """Check if any pair of colors induces a cycle. Returns True if valid (no bicolored cycles)."""
    n = P['n']
    colors_used = set(coloring[i] for i in range(n) if coloring[i] >= 0)
    for a in colors_used:
        for b in colors_used:
            if a >= b:
                continue
            # Build subgraph with only vertices of color a or b
            # Check if it has a cycle using BFS/DFS
            visited = set()
            has_cycle = False
            for start in range(n):
                if coloring[start] not in (a, b) or start in visited:
                    continue
                # BFS
                queue = deque([start])
                visited.add(start)
                while queue:
                    node = queue.popleft()
                    for nb in adj[node]:
                        if coloring[nb] not in (a, b):
                            continue
                        if nb in visited:
                            # Check if this creates a cycle
                            # A cycle exists if nb is not the parent and we've seen it
                            # Actually in BFS, if we see a visited node that's not the parent, it's a cycle
                            # But we need to be careful - let's use a different approach
                            pass
                        else:
                            visited.add(nb)
                            queue.append(nb)
            # Better approach: count edges vs vertices in each connected component
            # A forest has edges = vertices - components
            # Let's redo: for each connected component in the bicolored subgraph,
            # count vertices and edges. If edges >= vertices for any component, there's a cycle.
            visited2 = set()
            for start in range(n):
                if coloring[start] not in (a, b) or start in visited2:
                    continue
                # BFS to find component
                queue = deque([start])
                visited2.add(start)
                comp_vertices = [start]
                while queue:
                    node = queue.popleft()
                    for nb in adj[node]:
                        if coloring[nb] not in (a, b) or nb in visited2:
                            continue
                        visited2.add(nb)
                        comp_vertices.append(nb)
                        queue.append(nb)
                # Count edges within this component
                edge_count = 0
                for v in comp_vertices:
                    for nb in adj[v]:
                        if nb in visited2 and coloring[nb] in (a, b) and nb > v:
                            edge_count += 1
                if edge_count >= len(comp_vertices):
                    return False  # Cycle found
    return True

def _check_incremental(coloring, adj, vertex, color):
    """Check if assigning 'color' to 'vertex' creates a bicolored cycle.
    Only checks pairs involving the new color."""
    n = P['n']
    # For each other color c, check if the subgraph induced by {color, c} has a cycle
    # involving vertex
    for c in range(50):
        if c == color:
            continue
        # Check if any vertex with color c exists
        # We'll check if adding vertex (with 'color') to the {color,c} subgraph creates a cycle
        # The subgraph has vertices with color in {color, c}
        # A cycle is created if there's a path from vertex to itself through other vertices
        # of colors {color, c} that doesn't use vertex directly
        # BFS from vertex's neighbors (that have color c or color) to see if we can reach back
        
        # Actually, let's just do a full check for efficiency with small graphs
        # But for incremental, we can be smarter
        pass
    # For now, do full check
    return _check_bicolored_cycles(coloring, adj, 50)

def _backtrack_coloring(adj, max_colors, deadline):
    n = P['n']
    coloring = [-1] * n
    
    # Order vertices: use degree (all 4), then some heuristic
    # Let's try ordering by position or use a simple order
    # For structured graphs, try a snake-like order
    order = list(range(n))
    
    # Try to find a coloring with at most max_colors
    def bt(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        
        v = order[idx]
        for c in range(max_colors):
            # Check if color c is allowed (not same as any neighbor's color)
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            coloring[v] = c
            # Check bicolored cycles
            if _check_bicolored_cycles(coloring, adj, max_colors):
                result = bt(idx + 1)
                if result is not None:
                    return result
            coloring[v] = -1
        
        return None
    
    return bt(0)

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    adj = _build_adj()
    
    # Try with increasing number of colors
    for mc in [3, 4, 5, 6]:
        result = _backtrack_coloring(adj, mc, deadline)
        if result is not None:
            # Verify
            if _check_bicolored_cycles(result, adj, mc):
                return [int(x) for x in result]
    
    # Fallback: use all distinct colors (always valid)
    return [int(i) for i in range(P['n'])]
# EVOLVE-BLOCK-END
