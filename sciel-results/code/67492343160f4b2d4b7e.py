import time
import random

def construct():
    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    total_start_time = time.time()
    total_time_limit = 140.0
    
    for idx, (n, p, seed) in enumerate(instances):
        elapsed = time.time() - total_start_time
        remaining_time = total_time_limit - elapsed
        
        if remaining_time < 1.0:
            # Fallback to trivial if no time
            cliques = trivial_partition(n, p, seed)
        else:
            # Allocate time
            if idx == 0:
                instance_time_limit = min(remaining_time * 0.4, 40.0)
            else:
                instance_time_limit = min(remaining_time * 0.6, 100.0)
            
            cliques = solve_instance(n, p, seed, instance_time_limit)
        
        results.append(cliques)
        
    return results

def trivial_partition(n, p, seed):
    rng = random.Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append([u, v])
    return edges

def solve_instance(n, p, seed, time_limit):
    # Generate graph
    rng = random.Random(seed)
    adj = [0] * n
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    cliques = []
    start_time = time.time()
    
    # Mask for all vertices
    full_mask = (1 << n) - 1
    
    # We will maintain the residual graph
    # adj is mutated
    
    while True:
        # Check if any edges remain
        has_edges = False
        for i in range(n):
            if adj[i]:
                has_edges = True
                break
        if not has_edges:
            break
            
        current_time = time.time()
        if current_time - start_time > time_limit:
            # Timeout: add remaining edges as trivial cliques
            for u in range(n):
                mask = adj[u]
                # Iterate bits in mask
                while mask:
                    lsb = mask & -mask
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                    mask &= mask - 1
            break
        
        # Find max clique
        clique = find_max_clique(adj, n)
        
        if not clique or len(clique) < 2:
             # Safety: if max clique finder fails or returns <2, just pick an edge
             # This shouldn't happen if has_edges is True
             found = False
             for u in range(n):
                 mask = adj[u]
                 while mask:
                     lsb = mask & -mask
                     v = lsb.bit_length() - 1
                     if u < v:
                         clique = [u, v]
                         found = True
                         break
                     mask &= mask - 1
                 if found: break
             if not found: break
             
        # Remove edges in clique
        for i in range(len(clique)):
            u = clique[i]
            for j in range(i + 1, len(clique)):
                v = clique[j]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
        cliques.append(clique)
        
    return cliques

def find_max_clique(adj, n):
    # Tomita's algorithm
    best_clique = []
    
    # We use a recursive helper
    # To avoid deep recursion overhead, we can try to keep it shallow
    # But for n=64, depth <= 64 is fine.
    
    def expand(R, P):
        nonlocal best_clique
        
        if not P:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        # Get list of vertices in P
        P_list = []
        mask = P
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            P_list.append(v)
            mask &= mask - 1
            
        # Color P
        # colors[v] will store the color index
        # We need to map vertex -> color
        # Since n is small, we can use a dict or list
        color = {}
        # Order P_list by degree in P (optional but helpful)
        # Actually, Tomita uses the coloring to order.
        
        # Simple greedy coloring
        # We want to assign colors such that adjacent vertices have different colors
        # The color of a vertex is an upper bound on the clique size containing it
        # (specifically, the max clique size in the subgraph induced by P that includes v
        #  is at most the color assigned to v if we process in a specific order?
        #  No, the standard bound is: if we color P, the max clique size in P is
        #  at most the number of colors.
        #  For a specific vertex v, the max clique size containing v is at most
        #  the color assigned to v + 1? Or just color?
        #  Let's use the standard:
        #  Sort P by color.
        #  For each v in reverse order:
        #     if len(R) + color[v] <= len(best_clique): continue
        #     expand(R + [v], P & adj[v])
        
        # Coloring algorithm:
        # Iterate through vertices. Assign smallest available color.
        # To make the bound tight, we should order vertices by degree or just process them.
        # A common heuristic: process vertices in order of decreasing degree in P.
        
        # Let's calculate degrees in P
        degs = {}
        for v in P_list:
            # Count bits in P & adj[v]
            degs[v] = bin(P & adj[v]).count('1')
            
        # Sort P_list by degree descending
        P_list.sort(key=lambda x: degs[x], reverse=True)
        
        # Now color
        # We need to know which colors are used by neighbors in the *colored* set?
        # Standard greedy coloring for this bound:
        # We assign colors 1, 2, ... 
        # For each v in P_list:
        #    used_colors = set()
        #    for u in P_list:
        #        if u == v: continue
        #        if u in color: # already colored
        #             if P & (1 << u) and adj[v] & (1 << u): # u is neighbor of v in P
        #                 used_colors.add(color[u])
        #    # Find smallest color not in used_colors
        #    c = 1
        #    while c in used_colors:
        #        c += 1
        #    color[v] = c
        
        # This is O(N^2) per expand call. With N=64, it's 4096 ops. Very fast.
        
        for v in P_list:
            used = set()
            # Check neighbors in P that are already colored
            # Neighbors of v in P are in (P & adj[v])
            # We need to iterate through P_list again? Or just check the colored ones?
            # Since we are coloring in P_list order, we can just check previous vertices in P_list
            # if they are neighbors.
            for u in P_list:
                if u == v:
                    continue
                if u in color:
                    # Check if u is neighbor of v in original graph AND in P
                    if (adj[v] >> u) & 1 and (P >> u) & 1:
                        used.add(color[u])
            c = 1
            while c in used:
                c += 1
            color[v] = c
            
        # Now iterate P_list in reverse order of color (or just reverse of P_list if we want to try high degree first? 
        # Tomita says: iterate in reverse order of the coloring process.
        # Since we colored in P_list order, reversing P_list means we process the last colored first.
        # The last colored vertices tend to have higher colors? Not necessarily.
        # But the bound is: if len(R) + color[v] <= best_len, skip.
        
        for v in reversed(P_list):
            if len(R) + color[v] <= len(best_clique):
                continue
                
            # New P for recursion
            new_P = P & adj[v]
            # Remove v from P for the next iteration in the loop?
            # The loop iterates over a static list. We don't modify P_list.
            # But we must ensure we don't pick v again.
            # The recursion handles the new P.
            # We don't need to modify P in the current scope for the loop, 
            # because the loop is just generating candidates.
            
            expand(R + [v], new_P)
            
            # Pruning: if we found a clique of size len(R) + 1 + ... 
            # The standard Tomita also removes v from P? 
            # No, the loop iterates over the initial P_list.
            # However, if we include v in R, we effectively "used" it.
            # The recursion explores all cliques containing v.
            # The next iteration of the loop will consider a different vertex u.
            # This is correct.
            
    # Start with all vertices
    all_vertices_mask = (1 << n) - 1
    # But we only care about vertices with edges?
    # If a vertex has no edges, it won't be in any clique of size > 1.
    # But expand handles P=0.
    # We can start with all vertices.
    
    # Optimization: Start with a greedy clique to set a good initial bound?
    # Let's do a quick greedy to set best_clique
    # Pick vertex with max degree
    if n > 0:
        # Find max degree vertex
        max_deg = -1
        start_v = -1
        for i in range(n):
            d = bin(adj[i]).count('1')
            if d > max_deg:
                max_deg = d
                start_v = i
        
        if start_v != -1 and max_deg > 0:
            # Greedy expansion
            R = [start_v]
            P = adj[start_v]
            while P:
                # Pick u in P with max degree in P
                best_u = -1
                best_d = -1
                mask = P
                while mask:
                    lsb = mask & -mask
                    u = lsb.bit_length() - 1
                    d = bin(P & adj[u]).count('1')
                    if d > best_d:
                        best_d = d
                        best_u = u
                    mask &= mask - 1
                if best_u == -1:
                    break
                R.append(best_u)
                P = P & adj[best_u]
            
            if len(R) > len(best_clique):
                best_clique = R[:]
                
    expand([], all_vertices_mask)
    
    return best_clique
