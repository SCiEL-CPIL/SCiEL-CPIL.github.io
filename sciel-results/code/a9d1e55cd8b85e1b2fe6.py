# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_k222_bitmask(adj, n):
    """
    Count K2,2,2 subgraphs in graph G (defined by adj).
    K2,2,2 in G <=> Induced matching of size 3 in complement of G.
    
    Uses bitmask representation for fast neighbor intersection.
    """
    if n < 6:
        return 0
    
    # Build complement adjacency as bitmasks
    # comp_mask[i] = bitmask of vertices connected to i in complement
    comp_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and not adj[i, j]:
                mask |= (1 << j)
        comp_mask[i] = mask
    
    # Get list of edges in complement
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i, j]:
                edges.append((i, j))
    
    m = len(edges)
    if m < 3:
        return 0
    
    # For each edge (u,v) in complement, compute the set of vertices w
    # such that w is NOT connected to u or v in complement (i.e., w is "compatible")
    # Two edges (u,v) and (x,y) form an induced matching if:
    # - They are disjoint (no shared vertices)
    # - No cross edges in complement: comp[u,x]=comp[u,y]=comp[v,x]=comp[v,y]=False
    
    # Precompute for each edge the forbidden set
    # For edge (u,v), a vertex w is forbidden if comp[u,w] or comp[v,w]
    # So compatible vertices for edge (u,v) are those NOT in comp_mask[u] | comp_mask[v]
    
    # For two edges (u,v) and (x,y) to be compatible:
    # x,y must not be in comp_mask[u] | comp_mask[v]
    # AND u,v must not be in comp_mask[x] | comp_mask[y]
    # AND edges must be disjoint
    
    # Build compatibility: for each pair of edges, check if they form induced matching
    # Then count triangles in compatibility graph = count of 3 mutually compatible edges
    
    # For efficiency, let's use a different approach:
    # Iterate over pairs of compatible edges, then count how many edges are compatible with both.
    
    # Actually, let's just do the O(m^2) check but with bitmasks for speed.
    # For each edge i, precompute which edges j > i are compatible with i.
    
    # Better: Use the fact that for edge (u,v), the "forbidden" vertices are 
    # comp_mask[u] | comp_mask[v] (excluding u,v themselves)
    # An edge (x,y) is compatible with (u,v) iff:
    #   x,y not in {u,v} and (x|y) not in (comp_mask[u] | comp_mask[v])
    #   and (u|v) not in (comp_mask[x] | comp_mask[y])
    
    # Let's build a list of compatible partners for each edge
    # compat[i] = set of j > i such that edges[i] and edges[j] are compatible
    
    # For n=30, m can be up to ~435. O(m^2) = ~190k checks, each O(1) with bitmasks.
    
    # Actually, let me think about this more carefully.
    # The number of K2,2,2 = number of triangles in the compatibility graph.
    # 
    # Alternative: iterate over all triples of edges and check if they form an induced matching.
    # That's O(m^3) which is too slow for m=435.
    #
    # Better: For each pair of compatible edges (i,j), count how many edges k are compatible with both.
    # But we need to avoid double counting.
    #
    # Standard triangle counting: for each vertex i, look at its neighbors, count triangles.
    # 
    # Let me use a different strategy: 
    # For each edge e=(u,v), compute N(e) = set of edges compatible with e.
    # Then count triangles by: for each pair (e1,e2) in N(e), if e1 and e2 are also compatible, count it.
    # This is O(sum over e of |N(e)|^2) which could be large.
    #
    # For n=30, let me just do the direct O(m^2) compatibility check and then use matrix multiplication
    # for triangle counting, but with the compatibility check being fast.
    
    # Precompute: for each edge (u,v), the bitmask of vertices that are "blocked"
    # blocked[i] for edge i=(u,v) is: (1<<u)|(1<<v) | comp_mask[u] | comp_mask[v]
    # An edge j=(x,y) is compatible with edge i iff:
    #   (1<<x)|(1<<y) has no bits in blocked[i] AND (1<<u)|(1<<v) has no bits in blocked[j]
    
    blocked = [0] * m
    edge_vertices = [0] * m  # bitmask of the two vertices
    for idx, (u, v) in enumerate(edges):
        edge_vertices[idx] = (1 << u) | (1 << v)
        blocked[idx] = edge_vertices[idx] | comp_mask[u] | comp_mask[v]
    
    # Build compatibility matrix
    # compat[i,j] = True iff edges[i] and edges[j] form an induced matching
    # i.e., no shared vertices and no cross edges in complement
    
    # Two edges are compatible iff:
    # edge_vertices[i] & edge_vertices[j] == 0  (disjoint)
    # AND (edge_vertices[j] & blocked[i]) == 0  (no cross edges from j to i's vertices in comp)
    # AND (edge_vertices[i] & blocked[j]) == 0  (no cross edges from i to j's vertices in comp)
    
    # But wait: blocked[i] already includes comp_mask[u] | comp_mask[v], so if x is in comp_mask[u],
    # then x is in blocked[i]. So (edge_vertices[j] & blocked[i]) == 0 ensures that neither x nor y
    # is connected to u or v in the complement. Good.
    
    # Let's build the compat matrix
    compat = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            if edge_vertices[i] & edge_vertices[j]:
                continue  # shared vertex
            if edge_vertices[j] & blocked[i]:
                continue  # cross edge in complement
            if edge_vertices[i] & blocked[j]:
                continue  # cross edge in complement
            compat[i, j] = True
            compat[j, i] = True
    
    # Count triangles in compat graph
    if m < 3:
        return 0
    
    # Use matrix multiplication
    c_int = compat.astype(np.int8)
    c2 = c_int @ c_int
    triangles = np.trace(c2 @ c_int) // 6
    return int(triangles)

def _violation_score(adj, n):
    """Total number of monochromatic K2,2,2."""
    v0 = _count_k222_bitmask(adj, n)
    # Complement
    comp = ~adj
    np.fill_diagonal(comp, False)
    v1 = _count_k222_bitmask(comp, n)
    return v0 + v1

def _is_valid_fast(adj, n):
    return _violation_score(adj, n) == 0

def _paley_coloring(p):
    if p % 4 != 1:
        return None
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            colors.append(0 if diff in squares else 1)
    return colors

def _colors_to_adj(colors, n):
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            adj[i, j] = (colors[idx] == 0)
            adj[j, i] = (colors[idx] == 0)
            idx += 1
    return adj

def _adj_to_colors(adj, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if adj[i, j] else 1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_n = 1
    best_colors = []
    
    # 1. Get a valid base coloring (n=29)
    start_adj = None
    for p in [29, 17, 13, 5]:
        if p % 4 != 1:
            continue
        cols = _paley_coloring(p)
        if cols is None:
            continue
        adj_test = _colors_to_adj(cols, p)
        if _is_valid_fast(adj_test, p):
            start_adj = adj_test
            best_n = p
            best_colors = cols
            break
            
    if start_adj is None:
        return {'n': 1, 'colors': []}
        
    # 2. Try to find n=30 using local search on violation score
    target_n = 30
    
    # Multiple restarts with different initializations
    best_score_overall = float('inf')
    best_adj_overall = None
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # First restart: extend the known valid n=29 coloring
            adj_curr = np.zeros((target_n, target_n), dtype=bool)
            adj_curr[:best_n, :best_n] = start_adj
            # Randomize edges involving new vertex 29
            for i in range(best_n):
                val = random.random() < 0.5
                adj_curr[i, target_n - 1] = val
                adj_curr[target_n - 1, i] = val
        else:
            # Subsequent restarts: random initialization
            adj_curr = np.random.rand(target_n, target_n) > 0.5
            np.fill_diagonal(adj_curr, False)
            adj_curr = (adj_curr | adj_curr.T)
            np.fill_diagonal(adj_curr, False)
        
        current_score = _violation_score(adj_curr, target_n)
        best_score_sa = current_score
        best_adj_sa = adj_curr.copy()
        
        if current_score == 0:
            return {'n': target_n, 'colors': _adj_to_colors(adj_curr, target_n)}
        
        # Local search with simulated annealing
        T = 2.0
        T_min = 0.01
        cooling = 0.995
        max_steps = 3000
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if T < T_min:
                break
                
            # Pick random edge to flip
            i = random.randint(0, target_n - 1)
            j = random.randint(0, target_n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_val = adj_curr[i, j]
            adj_curr[i, j] = not old_val
            adj_curr[j, i] = not old_val
            
            new_score = _violation_score(adj_curr, target_n)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score_sa:
                    best_score_sa = current_score
                    best_adj_sa = adj_curr.copy()
                    if current_score == 0:
                        return {'n': target_n, 'colors': _adj_to_colors(adj_curr, target_n)}
            elif new_score == current_score:
                # Accept with some probability to escape plateaus
                if random.random() < 0.3:
                    current_score = new_score
            else:
                delta = new_score - current_score
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Revert
                    adj_curr[i, j] = old_val
                    adj_curr[j, i] = old_val
            
            T *= cooling
        
        # Track overall best
        if best_score_sa < best_score_overall:
            best_score_overall = best_score_sa
            best_adj_overall = best_adj_sa.copy()
        
        if best_score_overall == 0:
            return {'n': target_n, 'colors': _adj_to_colors(best_adj_overall, target_n)}
    
    # 3. If we found a valid one, return it
    if best_adj_overall is not None and best_score_overall == 0:
        return {'n': target_n, 'colors': _adj_to_colors(best_adj_overall, target_n)}
        
    # 4. Return best valid found (likely 29)
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
