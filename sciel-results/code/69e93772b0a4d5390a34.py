"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 115.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _has_triangle(adj, n):
    """Check if adjacency matrix has a triangle."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return True
    return False


def _max_independent_set_size(adj, n):
    """Find size of maximum independent set using branch and bound."""
    best = [0]

    def bnb(vertices, current_size):
        if not vertices:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Upper bound: current_size + remaining vertices
        if current_size + len(vertices) <= best[0]:
            return

        v = vertices[0]
        rest = vertices[1:]

        # Branch: include v
        # Neighbors of v (in the graph) are excluded
        neighbors_of_v = [u for u in rest if adj[v][u]]
        # Non-neighbors of v in rest
        non_neighbors = [u for u in rest if not adj[v][u]]
        bnb(non_neighbors, current_size + 1)

        # Branch: exclude v
        bnb(rest, current_size)

    bnb(list(range(n)), 0)
    return best[0]


def _validate(adj, n):
    """Check triangle-free and no independent 10-set."""
    if _has_triangle(adj, n):
        return False
    if _max_independent_set_size(adj, n) >= 10:
        return False
    return True


def _build_erdos_rao_style(n, seed=0):
    """
    Construct a triangle-free graph on n vertices using a probabilistic/construction approach.
    We use a known construction: take vertices as pairs (i, j) where i in range(a), j in range(b),
    n = a*b, and connect (i,j) to (i',j') if i != i' and some condition on j,j'.
    """
    # We'll use the classical construction:
    # Let p be prime, take n = p^2. Vertices are (x,y) in F_p x F_p.
    # Edge between (x,y) and (x',y') iff x + x' = 0 and y + y' = 1 (or similar).
    # Actually, let's use the Andrasfai-type or Clebsch-inspired approach.
    
    # For n up to 40, let's use a known family:
    # The Andrásfai graph And(k) has 3k vertices, is triangle-free, and alpha = k.
    # For k=10, n=30, alpha=10. But we need alpha < 10, so this won't work directly.
    
    # Better: use the Clebsch graph (16 vertices, alpha=5) as a base and extend.
    # Or use known constructions for R(3,k) > n.
    
    # Known: R(3,10) > 39 is claimed in the task. Let's try to build such a graph.
    # One approach: use the probabilistic method to find a triangle-free graph with alpha < 10.
    
    # Let's try a random greedy approach with triangle constraint and alpha constraint.
    rng = random.Random(seed)
    
    # Start with empty graph, add edges greedily avoiding triangles
    adj = [[0] * n for _ in range(n)]
    
    # For each potential edge (i,j), decide to add it if it doesn't create a triangle
    # and helps reduce the independence number.
    edges_order = list(combinations(range(n), 2))
    rng.shuffle(edges_order)
    
    for i, j in edges_order:
        # Check if adding (i,j) creates a triangle
        creates_triangle = False
        for k in range(n):
            if k != i and k != j and adj[i][k] and adj[j][k]:
                creates_triangle = True
                break
        if not creates_triangle:
            adj[i][j] = 1
            adj[j][i] = 1
    
    return adj


def _build_clebsch_extension(n, seed=0):
    """
    Start from known small triangle-free graphs and extend.
    Clebsch graph: 16 vertices, triangle-free, alpha=5.
    We can try to build larger graphs by taking disjoint unions or other operations,
    but disjoint union increases alpha additively, which is bad.
    
    Instead, let's use a known explicit construction for R(3,10) > 39.
    One known construction uses the (3,10) Ramsey graph from the literature.
    
    Let's try the following: use a circulant graph or a Paley-type construction.
    """
    # For now, return None to signal this approach isn't implemented yet
    return None


def _local_search(adj, n, seed=0):
    """
    Try to improve the graph by flipping edges (while maintaining triangle-free)
    to reduce the independence number.
    """
    rng = random.Random(seed)
    adj = [row[:] for row in adj]
    
    # Compute initial alpha
    alpha = _max_independent_set_size(adj, n)
    
    if alpha < 10:
        return adj, alpha
    
    # Try to add edges (where possible without creating triangles) to reduce alpha
    for _ in range(200):
        if _remaining_time() < 5:
            break
        
        i, j = rng.choice([(a, b) for a in range(n) for b in range(a + 1, n)])
        if adj[i][j]:
            continue
        
        # Check if adding (i,j) creates a triangle
        creates_triangle = False
        for k in range(n):
            if k != i and k != j and adj[i][k] and adj[j][k]:
                creates_triangle = True
                break
        
        if creates_triangle:
            continue
        
        # Add the edge
        adj[i][j] = 1
        adj[j][i] = 1
        
        new_alpha = _max_independent_set_size(adj, n)
        if new_alpha < alpha:
            alpha = new_alpha
            if alpha < 10:
                break
        else:
            # Revert
            adj[i][j] = 0
            adj[j][i] = 0
    
    return adj, alpha


def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj = None
    best_n = 0
    best_alpha = float('inf')
    
    # Try various n values from high to low
    # The task says R(3,10) <= 41, so max valid n is 40.
    # The survey says 39 is the largest known. Let's try 40, 39, 38, ...
    
    for n_target in range(40, 25, -1):
        if _remaining_time() < 10:
            break
        
        found = False
        for seed in range(50):
            if _remaining_time() < 5:
                break
            
            adj = _build_erdos_rao_style(n_target, seed)
            
            if _has_triangle(adj, n_target):
                continue
            
            alpha = _max_independent_set_size(adj, n_target)
            
            if alpha < 10:
                # Found a valid graph
                best_adj = adj
                best_n = n_target
                best_alpha = alpha
                found = True
                break
            
            # Try local search
            if _remaining_time() > 10:
                adj_ls, alpha_ls = _local_search(adj, n_target, seed)
                if alpha_ls < 10:
                    best_adj = adj_ls
                    best_n = n_target
                    best_alpha = alpha_ls
                    found = True
                    break
        
        if found:
            break
    
    # If we didn't find anything, fall back to the known Andrasfai(9) construction
    if best_adj is None:
        n = 26
        adj = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    adj[i][j] = 1
                    adj[j][i] = 1
        best_adj = adj
        best_n = n
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
