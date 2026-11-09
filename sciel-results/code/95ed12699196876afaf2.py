import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed, perturb_rng=None):
        edges = instance(n, k, p, seed)
        
        # Build adjacency for the whole graph
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        # Use bitmask representation for clique part edges
        edge_mask = [0] * k
        for u in range(k):
            for v in range(u + 1, k):
                edge_mask[u] |= (1 << v)
                edge_mask[v] |= (1 << u)
        
        partition = []
        
        # Precompute initial degrees (within C)
        full_mask = (1 << k) - 1
        deg = [bin(edge_mask[u] & full_mask).count('1') for u in range(k)]
        
        # Process independent set vertices: descending neighborhood size
        # Larger neighborhoods first allows greedy to form bigger cliques
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)), reverse=True)
        
        for v in I_sorted:
            N_v = adj[v] & set(C)
            
            if not N_v:
                continue
            
            while N_v:
                # Build bitmask for N_v
                nv_mask = 0
                for u in N_v:
                    nv_mask |= (1 << u)
                
                # Find vertex with max degree within N_v
                max_deg_vertex = None
                max_score = -1
                temp = nv_mask
                while temp:
                    lsb = temp & (-temp)
                    u = lsb.bit_length() - 1
                    temp ^= lsb
                    d = bin(edge_mask[u] & nv_mask).count('1')
                    if perturb_rng:
                        d += perturb_rng.random() * 0.5
                    if d > max_score:
                        max_score = d
                        max_deg_vertex = u
                
                current_clique = {max_deg_vertex}
                common_mask = edge_mask[max_deg_vertex] & nv_mask
                
                while True:
                    best_candidate = None
                    best_score = -1
                    candidates = []
                    
                    temp = common_mask
                    while temp:
                        lsb = temp & (-temp)
                        w = lsb.bit_length() - 1
                        temp ^= lsb
                        
                        score = bin(edge_mask[w] & nv_mask).count('1')
                        if perturb_rng:
                            score += perturb_rng.random() * 0.5
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = w
                            candidates = [w]
                        elif score == best_score:
                            candidates.append(w)
                            
                    if best_candidate is None:
                        break
                    
                    if len(candidates) > 1 and perturb_rng:
                        best_candidate = perturb_rng.choice(candidates)
                    
                    current_clique.add(best_candidate)
                    common_mask &= edge_mask[best_candidate]
                
                clique_list = sorted(list(current_clique | {v}))
                partition.append(clique_list)
                
                # Remove edges of current_clique from edge_mask
                clique_list_sorted = sorted(current_clique)
                for i in range(len(clique_list_sorted)):
                    u = clique_list_sorted[i]
                    for j in range(i + 1, len(clique_list_sorted)):
                        w = clique_list_sorted[j]
                        edge_mask[u] &= ~(1 << w)
                        edge_mask[w] &= ~(1 << u)
                
                N_v -= current_clique
        
        # Remaining edges in C
        remaining_edges = []
        for u in range(k):
            for v in range(u + 1, k):
                if edge_mask[u] & (1 << v):
                    remaining_edges.append((u, v))
        
        while remaining_edges:
            best_edge = None
            best_common = -1
            
            for e in remaining_edges:
                u, v = e
                common_mask = edge_mask[u] & edge_mask[v]
                common_count = bin(common_mask).count('1')
                if perturb_rng:
                    common_count += perturb_rng.random() * 0.5
                if common_count > best_common:
                    best_common = common_count
                    best_edge = e
            
            if best_edge is None:
                best_edge = remaining_edges[0]
            
            u, v = best_edge
            current_clique = {u, v}
            
            common_mask = edge_mask[u] & edge_mask[v]
            
            while True:
                best_candidate = None
                best_score = -1
                candidates = []
                
                temp = common_mask
                while temp:
                    lsb = temp & (-temp)
                    w = lsb.bit_length() - 1
                    temp ^= lsb
                    
                    score = bin(edge_mask[w]).count('1')
                    if perturb_rng:
                        score += perturb_rng.random() * 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = w
                        candidates = [w]
                    elif score == best_score:
                        candidates.append(w)
                
                if best_candidate is None:
                    break
                
                if len(candidates) > 1 and perturb_rng:
                    best_candidate = perturb_rng.choice(candidates)

                current_clique.add(best_candidate)
                common_mask &= edge_mask[best_candidate]
            
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
            
            clique_list_sorted = sorted(current_clique)
            for i in range(len(clique_list_sorted)):
                u2 = clique_list_sorted[i]
                for j in range(i + 1, len(clique_list_sorted)):
                    v2 = clique_list_sorted[j]
                    edge_mask[u2] &= ~(1 << v2)
                    edge_mask[v2] &= ~(1 << u2)
            
            clique_set = set(current_clique)
            new_remaining = []
            for e in remaining_edges:
                if e[0] in clique_set and e[1] in clique_set:
                    continue
                new_remaining.append(e)
            remaining_edges = new_remaining
        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    # Try multiple times for each instance
    # We will run the solver multiple times and keep the best (smallest number of cliques)
    # Limit total time to 140s to be safe.
    
    for idx, spec in enumerate(specs):
        if time.time() > deadline - 5.0:
             # Not enough time for this instance, use deterministic (no perturb)
             res = solve_instance(*spec, None)
             cleaned = [[int(x) for x in clique] for clique in res]
             results.append(cleaned)
             continue
             
        best_partition = None
        best_count = float('inf')
        
        # Determine how many restarts we can afford
        # Each run is fast (<1ms for these sizes). Use 100 restarts per instance.
        max_restarts = 100
        
        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Use a specific seed for the perturbation RNG to be reproducible if needed, 
            # or just random.
            perturb_seed = spec[3] * 100 + r
            perturb_rng = Random(perturb_seed)
            
            res = solve_instance(*spec, perturb_rng)
            count = len(res)
            
            if count < best_count:
                best_count = count
                best_partition = res
                
            # If we found a very good solution, maybe stop early?
            # Theoretical min for split graph is ~3n^2/16. 
            # For n=90, 3*8100/16 = 1518. 
            # If we get close to that, we can stop.
            # But we don't know the exact min.
            # Let's just run all restarts.
        
        if best_partition is None:
            best_partition = solve_instance(*spec, None)
            
        cleaned = [[int(x) for x in clique] for clique in best_partition]
        results.append(cleaned)
        
    return results
