"""Rewritten program: degree-ordered edge-replacement heuristic to minimize max triangle degree sum."""
import random
import time

def _triangles_and_scores(adj, n):
    """Compute all triangles and their degree sums."""
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                # Find common neighbors k > j to avoid duplicates
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        triangles.append((i, j, k))
    return triangles

def _max_triangle_degree_sum(adj, n, triangles):
    """Return the maximum degree sum over all triangles."""
    if not triangles:
        return 0
    max_sum = 0
    for (u, v, w) in triangles:
        s = adj[u].sum() + adj[v].sum() + adj[w].sum()
        if s > max_sum:
            max_sum = s
    return max_sum

def _find_worst_triangle(adj, n, triangles):
    """Return the triangle with the maximum degree sum, or None."""
    if not triangles:
        return None
    best = None
    best_sum = -1
    for tri in triangles:
        u, v, w = tri
        s = adj[u].sum() + adj[v].sum() + adj[w].sum()
        if s > best_sum:
            best_sum = s
            best = tri
    return best

def _try_improve(adj, n, max_iter=50000, time_limit=110):
    """
    Iteratively try to reduce the maximum triangle degree sum by swapping edges.
    Strategy: For the worst triangle, try to remove one of its edges and add a non-edge
    that reduces the degree of the highest-degree vertex in that triangle, while not
    creating worse triangles.
    """
    start_time = time.time()
    
    # Initial setup
    triangles = _triangles_and_scores(adj, n)
    current_max = _max_triangle_degree_sum(adj, n, triangles)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Recompute triangles and find worst
        triangles = _triangles_and_scores(adj, n)
        if not triangles:
            break
        worst = _find_worst_triangle(adj, n, triangles)
        if worst is None:
            break
        
        u, v, w = worst
        cur_sum = adj[u].sum() + adj[v].sum() + adj[w].sum()
        
        # Try each edge of the worst triangle for removal
        improved = False
        for (a, b) in [(u, v), (u, w), (v, w)]:
            if not adj[a][b]:
                continue
            
            # Try adding a non-edge (x, y) such that we can remove (a, b)
            # We want to reduce degree of the highest-degree vertex among u, v, w
            # Let's identify the highest-degree vertex in the triangle
            degs = [(adj[u].sum(), u), (adj[v].sum(), v), (adj[w].sum(), w)]
            degs.sort(reverse=True)
            high_deg_vertex = degs[0][1]
            
            # For each non-edge (x, y), check if swapping (a,b) -> (x,y) improves things
            # To be efficient, we only consider non-edges that involve the high-degree vertex
            # or could reduce its degree impact
            candidates = []
            for x in range(n):
                for y in range(x + 1, n):
                    if adj[x][y]:
                        continue
                    # Check if this non-edge is the same as (a,b)
                    if (x == a and y == b) or (x == b and y == a):
                        continue
                    candidates.append((x, y))
            
            # Shuffle for randomness
            random.shuffle(candidates)
            
            found_swap = False
            for (x, y) in candidates:
                if found_swap:
                    break
                # Perform the swap tentatively
                adj[a][b] = 0
                adj[b][a] = 0
                adj[x][y] = 1
                adj[y][x] = 1
                
                # Recompute triangles
                new_triangles = _triangles_and_scores(adj, n)
                new_max = _max_triangle_degree_sum(adj, n, new_triangles)
                
                if new_max < cur_sum:
                    # Accept the swap
                    cur_sum = new_max
                    improved = True
                    found_swap = True
                else:
                    # Revert
                    adj[a][b] = 1
                    adj[b][a] = 1
                    adj[x][y] = 0
                    adj[y][x] = 0
            
            if improved:
                break
        
        if not improved:
            # Try a random restart with perturbation if stuck
            # Add a random edge and remove a random edge to escape local optimum
            if it % 1000 == 999:
                # Find all edges
                all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
                all_non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i][j]]
                if all_edges and all_non_edges:
                    e_remove = random.choice(all_edges)
                    e_add = random.choice(all_non_edges)
                    adj[e_remove[0]][e_remove[1]] = 0
                    adj[e_remove[1]][e_remove[0]] = 0
                    adj[e_add[0]][e_add[1]] = 1
                    adj[e_add[1]][e_add[0]] = 1
    
    return adj

def construct():
    n = 16
    # Start with K_{8,8} plus one edge inside a part
    # This gives 64 + 1 = 65 edges
    # Vertices 0-7 and 8-15
    adj = [[0] * n for _ in range(n)]
    
    # Add K_{8,8} edges
    for u in range(8):
        for v in range(8, 16):
            adj[u][v] = 1
            adj[v][u] = 1
    
    # Add one edge inside part A: (0, 1)
    adj[0][1] = 1
    adj[1][0] = 1
    
    # Now apply the edge-replacement heuristic
    adj = _try_improve(adj, n, max_iter=100000, time_limit=105)
    
    # Extract edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append([u, v])
    
    # Verify we have exactly 65 edges
    assert len(edges) == 65, f"Expected 65 edges, got {len(edges)}"
    
    return {'n': n, 'edges': edges}
