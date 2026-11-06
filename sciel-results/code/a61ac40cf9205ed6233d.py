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

def _expand_clique(rem, u, v, n, rng):
    """
    Expand the edge (u, v) to a maximal clique in the graph defined by rem.
    Returns the list of vertices in the clique.
    """
    clique = [u, v]
    # Candidates must be neighbors of both u and v in rem
    cand = rem[u] & rem[v]
    
    while cand:
        # Pick vertex in cand with max degree within cand
        best_deg = -1
        best_v = -1
        temp = cand
        while temp:
            low = temp & (-temp)
            w = low.bit_length() - 1
            deg = (rem[w] & cand).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = w
            temp ^= low
        
        if best_v == -1:
            break
        
        clique.append(best_v)
        # Update candidates: must be neighbor of best_v too
        cand &= rem[best_v]
        
    return clique

def _solve_instance(adj, n, deadline, rng_seed):
    rem = [adj[i] for i in range(n)]
    remaining_edges = sum(rem[i].bit_count() for i in range(n)) // 2
    cliques = []
    
    if remaining_edges == 0:
        return cliques
    
    rng = Random(rng_seed)
    
    while remaining_edges > 0:
        if time.time() > deadline:
            break
        
        # Find an edge to start with.
        # Heuristic: Edge (u, v) with max (deg(u) + deg(v)) in rem.
        best_u, best_v = -1, -1
        max_score = -1
        
        for u in range(n):
            deg_u = rem[u].bit_count()
            if deg_u == 0:
                continue
            temp = rem[u]
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                if v > u:
                    deg_v = rem[v].bit_count()
                    score = deg_u + deg_v
                    if score > max_score:
                        max_score = score
                        best_u = u
                        best_v = v
                temp ^= low
        
        if best_u == -1:
            break
        
        # Expand to clique
        clique = _expand_clique(rem, best_u, best_v, n, rng)
        
        # Remove edges of this clique from rem
        m = len(clique)
        for i in range(m):
            for j in range(i + 1, m):
                a, b = clique[i], clique[j]
                rem[a] &= ~(1 << b)
                rem[b] &= ~(1 << a)
        
        cliques.append(clique)
        remaining_edges -= m * (m - 1) // 2
        
    # Handle any remaining edges
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
                if i < j:
                    cliques.append([i, j])
                    rem[i] &= ~(1 << j)
                    rem[j] &= ~(1 << i)
                    remaining_edges -= 1
                temp &= temp - 1
                
    return cliques

def construct():
    deadline = time.time() + 110
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        adj = _random_graph(n, p, seed)
        
        best_cliques = None
        best_count = float('inf')
        
        start_time = time.time()
        run_id = 0
        while time.time() < deadline - 5:
            sub_deadline = time.time() + 2.0
            cliques = _solve_instance(adj, n, sub_deadline, run_id)
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques
            run_id += 1
            
        if best_cliques is None:
            best_cliques = _solve_instance(adj, n, deadline, 0)
            
        results.append(best_cliques)
        
    return results
