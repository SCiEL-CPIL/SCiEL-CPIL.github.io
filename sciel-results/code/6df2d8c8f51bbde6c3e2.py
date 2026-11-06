import time
from random import Random

def _random_graph(n, p, seed):
    rng = Random(seed)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    return adj

def _greedy_maximal_clique(adj, start, n):
    """
    Fallback greedy maximal clique expansion.
    """
    clique = [start]
    cand = adj[start]
    while cand:
        best = -1
        best_deg = -1
        temp = cand
        while temp:
            low = temp & (-temp)
            u = low.bit_length() - 1
            deg = (adj[u] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best = u
            temp ^= low
        if best == -1:
            break
        clique.append(best)
        cand &= adj[best]
    return clique

def _coloring_bound(cand, adj, n):
    """
    Greedy coloring of the subgraph induced by cand.
    Returns a list of vertices sorted by color class (descending) and the max color.
    This is used for the upper bound in B&B.
    """
    # Simple greedy coloring
    # We want to assign colors such that adjacent vertices have different colors.
    # The number of colors is an upper bound on the clique size.
    # We process vertices in some order. A good order is by degree in cand (descending).
    
    verts = []
    t = cand
    while t:
        low = t & (-t)
        v = low.bit_length() - 1
        verts.append(v)
        t ^= low
        
    # Sort by degree in cand descending
    verts.sort(key=lambda v: (adj[v] & cand).bit_count(), reverse=True)
    
    color_of = {}
    next_color = 0
    for v in verts:
        used = set()
        for w in verts:
            if w == v:
                continue
            c = color_of.get(w)
            if c is None:
                continue
            if (adj[v] >> w) & 1:
                used.add(c)
        c = 0
        while c in used:
            c += 1
        color_of[v] = c
        if c + 1 > next_color:
            next_color = c + 1
            
    # Sort vertices by color descending for B&B branching order
    # Vertices with higher color are more likely to be in the max clique
    verts_sorted = sorted(verts, key=lambda v: color_of[v], reverse=True)
    return verts_sorted, next_color

def _max_clique_bb(adj, n, deadline, max_nodes=5000):
    """
    Branch and Bound to find a maximum clique in the graph defined by adj.
    adj: list of bitmasks.
    Returns the clique as a list of vertices.
    """
    best_clique = []
    best_size = 0
    nodes_visited = 0
    node_limit = max_nodes
    time_limit = deadline - 0.5 # Safety margin
    
    # Initial candidates: all vertices with degree > 0
    cand = 0
    for i in range(n):
        if adj[i]:
            cand |= (1 << i)
            
    if not cand:
        return []

    def expand(cand, current_clique):
        nonlocal best_clique, best_size, nodes_visited
        if time.time() > time_limit or nodes_visited > node_limit:
            return
            
        if not cand:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_clique = list(current_clique)
            return
            
        nodes_visited += 1
        
        # Get coloring bound
        verts, max_color = _coloring_bound(cand, adj, n)
        
        if len(current_clique) + max_color <= best_size:
            return
            
        # Branch on vertices in reverse color order
        # We iterate through verts, but we must be careful to only consider
        # vertices that are still in cand. The coloring was done on cand,
        # so all verts are in cand.
        # However, as we branch, we remove vertices from cand.
        # Standard Tomita:
        # while cand:
        #   v = select vertex
        #   ...
        
        # We'll iterate through the sorted list, but check if v is still in cand
        # This is inefficient if we remove many. 
        # Better: iterate through cand bits, but use the color order to pick the "best" one.
        # Or simply:
        for v in verts:
            if not (cand & (1 << v)):
                continue
                
            # Include v
            current_clique.append(v)
            new_cand = cand & adj[v]
            expand(new_cand, current_clique)
            current_clique.pop()
            
            # Exclude v
            cand &= ~(1 << v)
            
            # Pruning check after removing v
            # Re-compute bound? Or just check if remaining can beat best?
            # For efficiency, we can just let the loop continue.
            # The bound at the top of the next iteration (or recursive call) will handle it.
            # However, if we are in the main loop, we should check if the *remaining* cand
            # can still form a clique larger than best_size.
            # The coloring bound `max_color` was for the *original* cand in this call.
            # As we remove vertices, the bound decreases.
            # A simple check: if popcount(cand) + len(current_clique) <= best_size: break
            if cand.bit_count() + len(current_clique) <= best_size:
                break

    # Start B&B
    # We can start with an empty clique
    # To get a good initial lower bound, we can run a greedy clique first
    start_v = -1
    max_deg = -1
    t = cand
    while t:
        low = t & (-t)
        v = low.bit_length() - 1
        deg = (adj[v] & cand).bit_count()
        if deg > max_deg:
            max_deg = deg
            start_v = v
        t ^= low
        
    if start_v != -1:
        init_clique = _greedy_maximal_clique(adj, start_v, n)
        best_size = len(init_clique)
        best_clique = init_clique
        # Remove edges of init_clique from a temporary adj? 
        # No, B&B finds the max clique in the *current* graph.
        # We just use it as a lower bound.
        
    expand(cand, [])
    
    return best_clique

def _solve_instance(adj, n, deadline):
    rem = [adj[i] for i in range(n)]
    remaining_edges = sum(rem[i].bit_count() for i in range(n)) // 2
    cliques = []
    
    if remaining_edges == 0:
        return cliques

    while remaining_edges > 0:
        if time.time() > deadline:
            break
            
        # Check if we should use B&B or Greedy
        # B&B is expensive. Use it for the first few cliques or if graph is small enough.
        # Given n=32, 64, B&B should be fast.
        
        # Find max clique
        clique = _max_clique_bb(rem, n, deadline, max_nodes=2000)
        
        # Fallback to greedy if B&B fails or returns small clique
        if len(clique) < 2:
            # Find a vertex with max degree
            max_deg = -1
            v = -1
            for i in range(n):
                d = rem[i].bit_count()
                if d > max_deg:
                    max_deg = d
                    v = i
            if v == -1:
                break
            clique = _greedy_maximal_clique(rem, v, n)
            
        if len(clique) < 2:
            # Should not happen if remaining_edges > 0
            # Pick any edge
            for i in range(n):
                if rem[i]:
                    j = (rem[i] & (-rem[i])).bit_length() - 1
                    clique = [i, j]
                    break
        
        cliques.append(clique)
        
        # Remove edges of clique
        m = len(clique)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = clique[i], clique[j]
                rem[a] &= ~(1 << b)
                rem[b] &= ~(1 << a)
        remaining_edges -= m * (m - 1) // 2
        
    # Handle remaining edges if deadline hit or B&B failed to clear all
    if remaining_edges > 0:
        for i in range(n):
            if time.time() > deadline:
                break
            temp = rem[i]
            while temp:
                if time.time() > deadline:
                    break
                low = temp & (-temp)
                j = low.bit_length() - 1
                cliques.append([i, j])
                rem[i] &= ~(1 << j)
                rem[j] &= ~(1 << i)
                temp &= temp - 1
                
    return cliques

def construct():
    deadline = time.time() + 140
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            results.append([])
            continue
        adj = _random_graph(n, p, seed)
        cliques = _solve_instance(adj, n, deadline)
        results.append(cliques)
    return results
