"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices is triangle-free."""
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable using backtracking."""
    color = [-1] * n
    # Order vertices by degree (greedy heuristic)
    deg = [sum(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Try colors 0, 1, 2
        for c in range(3):
            # Check if color c is available for v
            ok = True
            for u in range(n):
                if adj[v][u] and color[u] == c:
                    ok = False
                    break
            if ok:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return not backtrack(0)

def _count_edges(adj, n):
    return sum(sum(adj[i]) for i in range(n)) // 2

def _get_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def _groetzsch_adj(n=16):
    """Create adjacency matrix for Grötzsch graph on vertices 0-10, isolated 11-15."""
    adj = [[0]*n for _ in range(n)]
    # Cycle on 0-4
    for i in range(5):
        u, v = i, (i+1) % 5
        adj[u][v] = adj[v][u] = 1
    # Vertices 5-9: vertex 5+w connected to u and w for each edge (u,w) in cycle
    for i in range(5):
        u, w = i, (i+1) % 5
        adj[u][5+w] = adj[5+w][u] = 1
        adj[w][5+u] = adj[5+u][w] = 1
    # Vertex 10 connected to all of 5-9
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe
    
    n = 16
    # Start with Grötzsch graph
    adj = _groetzsch_adj(n)
    best_adj = [row[:] for row in adj]
    best_edges = _count_edges(adj, n)
    
    # All possible edges not already present
    all_possible = []
    for i in range(n):
        for j in range(i+1, n):
            if not adj[i][j]:
                all_possible.append((i, j))
    
    # Greedy phase: try to add edges in order of some heuristic
    # Sort by: prefer edges involving isolated vertices first, then by degree
    def edge_priority(e):
        i, j = e
        # Prefer edges that connect to high-degree vertices
        deg_i = sum(adj[i])
        deg_j = sum(adj[j])
        return -(deg_i + deg_j)
    
    all_possible.sort(key=edge_priority)
    
    # Try adding each edge greedily
    for (i, j) in all_possible:
        if time.time() > deadline:
            break
        # Check triangle-freeness
        if adj[i][j]:
            continue
        # Check if adding edge (i,j) creates a triangle
        creates_triangle = False
        for k in range(n):
            if k != i and k != j and adj[i][k] and adj[j][k]:
                creates_triangle = True
                break
        if creates_triangle:
            continue
        # Tentatively add edge
        adj[i][j] = 1
        adj[j][i] = 1
        # Check 3-colorability
        if _is_not_3_colorable(adj, n):
            # Keep the edge
            new_edges = _count_edges(adj, n)
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = [row[:] for row in adj]
        else:
            # Remove the edge
            adj[i][j] = 0
            adj[j][i] = 0
    
    # Local search / hill climbing phase
    if time.time() < deadline:
        # Try to improve by swapping edges
        current_adj = [row[:] for row in adj]
        current_edges = _count_edges(current_adj, n)
        
        # Build set of current edges for quick lookup
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if current_adj[i][j]:
                    edge_set.add((i, j))
        
        # Try removing one edge and adding another
        for _ in range(50):  # Limited iterations
            if time.time() > deadline:
                break
            improved = False
            # Try removing each edge
            for (ri, rj) in list(edge_set):
                if time.time() > deadline:
                    break
                # Remove edge
                current_adj[ri][rj] = 0
                current_adj[rj][ri] = 0
                edge_set.discard((ri, rj))
                
                # Try adding each non-edge
                for (ai, aj) in all_possible:
                    if current_adj[ai][aj]:
                        continue
                    # Check triangle
                    creates_triangle = False
                    for k in range(n):
                        if k != ai and k != aj and current_adj[ai][k] and current_adj[aj][k]:
                            creates_triangle = True
                            break
                    if creates_triangle:
                        continue
                    # Tentatively add
                    current_adj[ai][aj] = 1
                    current_adj[aj][ai] = 1
                    if _is_not_3_colorable(current_adj, n):
                        new_count = _count_edges(current_adj, n)
                        if new_count > current_edges:
                            current_edges = new_count
                            edge_set.add((ai, aj))
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = [row[:] for row in current_adj]
                            improved = True
                            break
                        else:
                            current_adj[ai][aj] = 0
                            current_adj[aj][ai] = 0
                    else:
                        current_adj[ai][aj] = 0
                        current_adj[aj][ai] = 0
                
                if improved:
                    break
                # If no improvement, restore removed edge
                current_adj[ri][rj] = 1
                current_adj[rj][ri] = 1
                edge_set.add((ri, rj))
    
    return _get_edges(best_adj, n)

# EVOLVE-BLOCK-END
