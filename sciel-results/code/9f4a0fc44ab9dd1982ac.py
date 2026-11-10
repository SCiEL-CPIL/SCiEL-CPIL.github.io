"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict, deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build adjacency and edge structures
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # For each vertex, list of incident edge indices
    vertex_edges = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        vertex_edges[u].append(i)
        vertex_edges[v].append(i)
    
    # Edge endpoints
    edge_u = [e[0] for e in edges]
    edge_v = [e[1] for e in edges]
    
    # CSP: assign each edge a color in {0, 1, 2}
    # Constraint: each vertex has exactly 2 edges of each color
    
    # We'll use backtracking with forward checking
    # Variables: edges 0..m-1
    # Domain: {0, 1, 2} for each edge
    
    # Count how many edges of each color each vertex has
    # vertex_color_count[v][c] = number of edges incident to v with color c
    
    # Order edges by some heuristic - maybe by vertex connectivity
    # Let's try a simple ordering first
    
    sys.setrecursionlimit(10000)
    
    # Try to find a valid 3-edge-coloring where each vertex has 2 edges per color
    # Then check if each color class is a single Hamiltonian cycle
    
    def is_hamiltonian_cycle(color_class_edges):
        """Check if the edges form a single cycle on all n vertices."""
        if len(color_class_edges) != n:
            return None
        # Build adjacency for this color class
        adj_cc = defaultdict(list)
        for i in color_class_edges:
            u, v = edge_u[i], edge_v[i]
            adj_cc[u].append(v)
            adj_cc[v].append(u)
        # Each vertex should have degree 2
        for v in range(n):
            if len(adj_cc[v]) != 2:
                return None
        # Traverse to find the cycle
        visited = set()
        path = []
        start = 0
        current = start
        visited.add(current)
        path.append(current)
        for _ in range(n - 1):
            nexts = adj_cc[current]
            # Pick the unvisited neighbor
            nxt = None
            for nb in nexts:
                if nb not in visited:
                    nxt = nb
                    break
            if nxt is None:
                return None
            current = nxt
            visited.add(current)
            path.append(current)
        # Check that we can close the cycle
        if path[0] not in adj_cc[path[-1]]:
            return None
        if len(visited) != n:
            return None
        return path
    
    # Backtracking search with constraint propagation
    # State: assignment array (edge -> color or -1 for unassigned)
    # For each vertex, track count of each color assigned to its incident edges
    
    assignment = [-1] * m
    vertex_color_count = [[0, 0, 0] for _ in range(n)]  # [vertex][color]
    
    # Precompute: for each edge, its two endpoints
    # For forward checking, when we assign edge i color c, we increment
    # vertex_color_count[u][c] and vertex_color_count[v][c]
    # and check that no count exceeds 2
    
    # Variable ordering: use minimum remaining values or just sequential
    # Let's try sequential with forward checking
    
    def solve(edge_idx):
        if edge_idx == m:
            # All edges assigned, verify each color class is Hamiltonian
            color_classes = [[], [], []]
            for i in range(m):
                color_classes[assignment[i]].append(i)
            cycles = []
            for c in range(3):
                cyc = is_hamiltonian_cycle(color_classes[c])
                if cyc is None:
                    return False
                cycles.append(cyc)
            # Store result
            result['cycles'] = cycles
            return True
        
        u = edge_u[edge_idx]
        v = edge_v[edge_idx]
        
        # Try each color
        for c in range(3):
            # Check if assigning color c to this edge is feasible
            if vertex_color_count[u][c] >= 2:
                continue
            if vertex_color_count[v][c] >= 2:
                continue
            
            # Assign
            assignment[edge_idx] = c
            vertex_color_count[u][c] += 1
            vertex_color_count[v][c] += 1
            
            # Forward check: for each unassigned edge incident to u or v,
            # check if it still has at least one valid color
            feasible = True
            for other_edge in vertex_edges[u]:
                if other_edge == edge_idx or other_edge < edge_idx:
                    continue
                if assignment[other_edge] != -1:
                    continue
                ou = edge_u[other_edge]
                ov = edge_v[other_edge]
                # Check if any color is still possible
                valid = False
                for oc in range(3):
                    if vertex_color_count[ou][oc] < 2 and vertex_color_count[ov][oc] < 2:
                        valid = True
                        break
                if not valid:
                    feasible = False
                    break
            if feasible:
                for other_edge in vertex_edges[v]:
                    if other_edge == edge_idx or other_edge < edge_idx:
                        continue
                    if assignment[other_edge] != -1:
                        continue
                    ou = edge_u[other_edge]
                    ov = edge_v[other_edge]
                    valid = False
                    for oc in range(3):
                        if vertex_color_count[ou][oc] < 2 and vertex_color_count[ov][oc] < 2:
                            valid = True
                            break
                    if not valid:
                        feasible = False
                        break
            
            if feasible:
                if solve(edge_idx + 1):
                    return True
            
            # Unassign
            assignment[edge_idx] = -1
            vertex_color_count[u][c] -= 1
            vertex_color_count[v][c] -= 1
        
        return False
    
    result = {'cycles': None}
    
    if solve(0):
        cycles = result['cycles']
        # Convert to proper format
        return [list(c) for c in cycles]
    else:
        # Fallback: try to find at least 2 edge-disjoint Hamiltonian cycles
        # using the old approach
        return _fallback()
    
    def _fallback():
        # Simple backtracking for 2 Hamiltonian cycles
        adj1 = {v: set(neighs) for v, neighs in adj.items()}
        
        def find_ham(start, graph_adj, n):
            visited = [False] * n
            path = [start]
            visited[start] = True
            
            def bt():
                if len(path) == n:
                    if start in graph_adj[path[-1]]:
                        return True
                    return False
                last = path[-1]
                for nb in sorted(graph_adj[last]):
                    if not visited[nb]:
                        if len(path) == n - 1:
                            if start not in graph_adj[nb]:
                                continue
                        visited[nb] = True
                        path.append(nb)
                        if bt():
                            return True
                        path.pop()
                        visited[nb] = False
                return False
            
            if bt():
                return path
            return None
        
        c1 = find_ham(0, adj1, n)
        if c1 is None:
            return [list(range(n))]
        
        adj2 = {v: set(neighs) for v, neighs in adj.items()}
        for i in range(len(c1)):
            u = c1[i]
            v = c1[(i + 1) % len(c1)]
            adj2[u].discard(v)
            adj2[v].discard(u)
        
        c2 = find_ham(0, adj2, n)
        if c2 is None:
            return [c1]
        
        return [c1, c2]
    
# EVOLVE-BLOCK-END
