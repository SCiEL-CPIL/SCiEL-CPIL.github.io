"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build adjacency and incidence lists
    adj = defaultdict(list)
    edge_to_vertices = {}
    for i, (u, v) in enumerate(edges):
        adj[u].append(i)
        adj[v].append(i)
        edge_to_vertices[i] = (u, v)
        
    # CSP: Assign color 0, 1, or 2 to each edge.
    # Constraint: For each vertex v and color c, exactly 2 incident edges have color c.
    
    # Order edges to improve backtracking efficiency.
    # We can order edges by the "constrainedness" of their endpoints or just use a fixed order.
    # Let's try ordering by the sum of degrees of endpoints (all are 5, so no help).
    # We will just iterate edges 0 to m-1.
    
    # State:
    # color_of_edge: list of length m, initialized to -1 (unassigned)
    # count_color_at_vertex: 2D list [n][3], counts of assigned edges of each color at each vertex
    
    color_of_edge = [-1] * m
    count_color_at_vertex = [[0, 0, 0] for _ in range(n)]
    
    # Precompute neighbors for each edge to update counts
    edge_endpoints = [(u, v) for u, v in edges]
    
    def is_valid_assignment(edge_idx, color):
        u, v = edge_endpoints[edge_idx]
        # Check if adding this edge violates the max 2 constraint
        if count_color_at_vertex[u][color] >= 2:
            return False
        if count_color_at_vertex[v][color] >= 2:
            return False
        return True

    def assign(edge_idx, color):
        u, v = edge_endpoints[edge_idx]
        color_of_edge[edge_idx] = color
        count_color_at_vertex[u][color] += 1
        count_color_at_vertex[v][color] += 1

    def unassign(edge_idx, color):
        u, v = edge_endpoints[edge_idx]
        color_of_edge[edge_idx] = -1
        count_color_at_vertex[u][color] -= 1
        count_color_at_vertex[v][color] -= 1

    # Backtracking search
    # We need to ensure that at the end, for every vertex v, count_color_at_vertex[v][c] == 2 for all c.
    # During search, we enforce <= 2.
    
    def backtrack(edge_idx):
        if edge_idx == m:
            # Check if all constraints are met (exactly 2 for each color at each vertex)
            for v in range(n):
                for c in range(3):
                    if count_color_at_vertex[v][c] != 2:
                        return False
            # Check if the subgraphs for colors 0 and 1 are single cycles (Hamiltonian)
            # Color 2 will automatically be a perfect matching if 0 and 1 are valid Hamiltonian cycles
            # in a 5-regular graph? Not necessarily, but if 0 and 1 are 2-regular and connected, they are cycles.
            # We only care about colors 0 and 1 being Hamiltonian cycles.
            
            # Build adjacency for color 0
            adj0 = defaultdict(list)
            adj1 = defaultdict(list)
            
            for i in range(m):
                u, v = edge_endpoints[i]
                if color_of_edge[i] == 0:
                    adj0[u].append(v)
                    adj0[v].append(u)
                elif color_of_edge[i] == 1:
                    adj1[u].append(v)
                    adj1[v].append(u)
            
            # Check if adj0 is a single cycle of length n
            # Since every vertex has degree 2 in adj0, it's a union of cycles.
            # We just need to check if it's connected.
            visited0 = [False] * n
            # Start from 0
            stack = [0]
            visited0[0] = True
            count_nodes = 0
            while stack:
                node = stack.pop()
                count_nodes += 1
                for nei in adj0[node]:
                    if not visited0[nei]:
                        visited0[nei] = True
                        stack.append(nei)
            
            if count_nodes != n:
                return False
            
            # Check if adj1 is a single cycle of length n
            visited1 = [False] * n
            stack = [0]
            visited1[0] = True
            count_nodes = 0
            while stack:
                node = stack.pop()
                count_nodes += 1
                for nei in adj1[node]:
                    if not visited1[nei]:
                        visited1[nei] = True
                        stack.append(nei)
            
            if count_nodes != n:
                return False
            
            # Extract the cycles
            # Get cycle 0
            cycle0 = []
            curr = 0
            prev = -1
            while True:
                cycle0.append(curr)
                nexts = adj0[curr]
                if prev == -1:
                    # Pick a neighbor
                    nxt = nexts[0]
                else:
                    # Pick the neighbor that is not prev
                    if nexts[0] == prev:
                        nxt = nexts[1]
                    else:
                        nxt = nexts[0]
                if nxt == 0:
                    break
                prev, curr = curr, nxt
            
            # Get cycle 1
            cycle1 = []
            curr = 0
            prev = -1
            while True:
                cycle1.append(curr)
                nexts = adj1[curr]
                if prev == -1:
                    nxt = nexts[0]
                else:
                    if nexts[0] == prev:
                        nxt = nexts[1]
                    else:
                        nxt = nexts[0]
                if nxt == 0:
                    break
                prev, curr = curr, nxt
            
            # Store solution
            construct.solution = [cycle0, cycle1]
            return True
        
        # Try assigning colors 0, 1, 2 to edge edge_idx
        for color in range(3):
            if is_valid_assignment(edge_idx, color):
                assign(edge_idx, color)
                if backtrack(edge_idx + 1):
                    return True
                unassign(edge_idx, color)
        
        return False

    # Heuristic: Order edges. 
    # A good heuristic is to assign edges that are in "tight" vertices first.
    # However, since all vertices have degree 5 and need 2 of each color, the symmetry is high.
    # We can just iterate 0 to m-1.
    
    if backtrack(0):
        # Ensure types are correct
        result = construct.solution
        # Convert to lists of ints
        cycles = []
        for c in result:
            cycles.append([int(x) for x in c])
        return cycles
    else:
        # Fallback if CSP fails (should not happen for this graph)
        return [list(range(n))]

# EVOLVE-BLOCK-END
