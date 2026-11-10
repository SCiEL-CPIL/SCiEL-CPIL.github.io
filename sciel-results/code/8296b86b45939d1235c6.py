"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # We want to find edge-disjoint Hamiltonian cycles on all 21 vertices.
    # The graph is 5-regular, so max 2 edge-disjoint Hamiltonian cycles.
    # Use a CSP approach: assign each edge to one of 2 cycle groups (or unused),
    # such that each vertex has exactly 2 incident edges per group.
    
    # For each edge, assign a color: 0, 1, or 2 (unused)
    # Constraint: for each vertex, exactly 2 edges of color 0 and exactly 2 edges of color 1
    
    # Let's use backtracking with constraint propagation
    
    # First, fix the first edge to break symmetry
    # Edge (0,1) -> color 0
    edge_colors = [-1] * m  # -1 means unassigned
    edge_colors[0] = 0  # (0,1) is color 0
    
    # Count of color assignments per vertex
    # vertex_color_count[v][c] = number of edges of color c incident to v
    vertex_color_count = [[0, 0, 0] for _ in range(n)]
    vertex_color_count[0][0] += 1
    vertex_color_count[1][0] += 1
    
    # Remaining capacity per vertex per color
    # Each vertex needs exactly 2 of color 0 and 2 of color 1
    vertex_capacity = [[2, 2, 0] for _ in range(n)]  # [color0, color1, unused]
    vertex_capacity[0][0] -= 1
    vertex_capacity[1][0] -= 1
    
    # Find unassigned edges
    def get_unassigned_edges():
        return [i for i in range(m) if edge_colors[i] == -1]
    
    def try_assign(edge_idx, color):
        """Try to assign a color to an edge. Returns True if successful."""
        u, v = edges[edge_idx]
        
        # Check capacity
        if vertex_capacity[u][color] < 1:
            return False
        if vertex_capacity[v][color] < 1:
            return False
        
        # Assign
        edge_colors[edge_idx] = color
        vertex_color_count[u][color] += 1
        vertex_color_count[v][color] += 1
        vertex_capacity[u][color] -= 1
        vertex_capacity[v][color] -= 1
        return True
    
    def unassign(edge_idx):
        """Remove assignment from an edge."""
        u, v = edges[edge_idx]
        color = edge_colors[edge_idx]
        edge_colors[edge_idx] = -1
        vertex_color_count[u][color] -= 1
        vertex_color_count[v][color] -= 1
        vertex_capacity[u][color] += 1
        vertex_capacity[v][color] += 1
    
    def is_complete():
        """Check if all edges are assigned."""
        return all(c != -1 for c in edge_colors)
    
    def get_conflicting_edges():
        """Get edges that can still be assigned some color."""
        result = []
        for i in range(m):
            if edge_colors[i] != -1:
                continue
            u, v = edges[i]
            valid_colors = []
            for c in range(3):
                if vertex_capacity[u][c] >= 1 and vertex_capacity[v][c] >= 1:
                    valid_colors.append(c)
            if valid_colors:
                result.append((i, valid_colors))
        return result
    
    # Backtracking search
    def backtrack(depth):
        if is_complete():
            return True
        
        # Get edges that can still be assigned
        assignable = get_conflicting_edges()
        if not assignable:
            return False
        
        # Choose the edge with fewest valid colors (MRV heuristic)
        edge_idx, valid_colors = min(assignable, key=lambda x: len(x[1]))
        
        for color in valid_colors:
            if try_assign(edge_idx, color):
                if backtrack(depth + 1):
                    return True
                unassign(edge_idx)
        
        return False
    
    # Run the backtracking search
    if not backtrack(0):
        # Fallback: try a simpler approach
        return [list(range(n))]
    
    # Extract cycles from edge colors
    # For each color, build the cycle
    cycles = []
    for color in [0, 1]:
        # Get edges of this color
        color_edges = []
        for i, (u, v) in enumerate(edges):
            if edge_colors[i] == color:
                color_edges.append((u, v))
        
        # Build adjacency for this color
        color_adj = defaultdict(list)
        for u, v in color_edges:
            color_adj[u].append(v)
            color_adj[v].append(u)
        
        # Each vertex should have degree 2 in this subgraph
        # Find the cycle
        visited = set()
        cycle = []
        start = 0
        current = start
        visited.add(start)
        cycle.append(start)
        
        for _ in range(n - 1):
            nexts = color_adj[current]
            # Choose the next vertex
            next_vertex = None
            for nv in nexts:
                if nv not in visited:
                    next_vertex = nv
                    break
            if next_vertex is None:
                # Should not happen if we have a valid 2-regular subgraph
                break
            visited.add(next_vertex)
            cycle.append(next_vertex)
            current = next_vertex
        
        if len(cycle) == n:
            cycles.append(cycle)
    
    if len(cycles) == 2:
        return [cycles[0], cycles[1]]
    elif len(cycles) == 1:
        return [cycles[0]]
    else:
        return [list(range(n))]

# EVOLVE-BLOCK-END
