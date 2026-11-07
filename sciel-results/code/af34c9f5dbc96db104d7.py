import math
import time

def construct():
    N = 256
    t0 = time.time()
    deadline = t0 + 80  # safety margin before 90s kill
    
    # Weights: vertex i (0-indexed) has weight i+1
    weights = [i + 1 for i in range(N)]
    
    # Build conflict graph: neighbors[i] is bitset of j where gcd(i+1, j+1) > 1
    neighbors = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if i != j and math.gcd(vi, j + 1) > 1:
                mask |= (1 << j)
        neighbors[i] = mask
    
    # can_coexist[i] = all vertices j where j==i or gcd(i+1,j+1)==1
    can_coexist = [0] * N
    all_mask = (1 << N) - 1
    for i in range(N):
        can_coexist[i] = all_mask ^ neighbors[i]
    
    # Precompute weight sums for quick upper-bound computation
    # Use a lookup: for each vertex, store its weight
    # We'll compute sum-of-available incrementally or use a precomputed table
    
    # Greedy initial solution: pick highest-weight vertices that don't conflict
    def greedy_solution():
        avail = all_mask
        total = 0
        mask = 0
        while avail:
            # Pick the vertex with highest weight in avail
            best_v = -1
            best_w = -1
            tmp = avail
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_v = idx
                tmp ^= lsb
            if best_v == -1:
                break
            mask |= (1 << best_v)
            total += best_w
            avail = avail & can_coexist[best_v] & ~(1 << best_v)
        return total, mask
    
    # Try multiple greedy orders for a better initial bound
    best_init_sum = 0
    best_init_mask = 0
    
    # Standard greedy (highest weight first)
    s, m = greedy_solution()
    best_init_sum = s
    best_init_mask = m
    
    # Greedy by highest weight / degree ratio
    def greedy_ratio():
        avail = all_mask
        total = 0
        mask = 0
        while avail:
            best_v = -1
            best_ratio = -1
            tmp = avail
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                deg = bin(neighbors[idx] & avail).count('1')
                ratio = weights[idx] / (deg + 1)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_v = idx
                tmp ^= lsb
            if best_v == -1:
                break
            mask |= (1 << best_v)
            total += best_v + 1
            avail = avail & can_coexist[best_v] & ~(1 << best_v)
        return total, mask
    
    s2, m2 = greedy_ratio()
    if s2 > best_init_sum:
        best_init_sum = s2
        best_init_mask = m2
    
    # Greedy: try including each vertex first, then fill greedily
    for start_v in range(N):
        avail = all_mask & can_coexist[start_v] & ~(1 << start_v)
        total = weights[start_v]
        mask = (1 << start_v)
        while avail:
            best_v = -1
            best_w = -1
            tmp = avail
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_v = idx
                tmp ^= lsb
            if best_v == -1:
                break
            mask |= (1 << best_v)
            total += best_v + 1
            avail = avail & can_coexist[best_v] & ~(1 << best_v)
        if total > best_init_sum:
            best_init_sum = total
            best_init_mask = mask
        if time.time() > deadline - 60:
            break
    
    best_sum = [best_init_sum]
    best_mask = [best_init_mask]
    
    # Greedy clique cover upper bound on the available set
    # The conflict graph edges represent "cannot coexist" pairs.
    # A clique in the conflict graph = a set of mutually non-coprime vertices.
    # In such a clique, at most 1 vertex can be in our independent set.
    # So the sum of weights of a clique is an upper bound on what we can take from it.
    # Greedy coloring of the conflict graph gives a clique cover: each color class is a clique.
    # Sum of (max weight in each color class) is an upper bound.
    
    def clique_cover_bound(avail):
        """Compute upper bound using greedy clique cover on conflict graph restricted to avail."""
        if avail == 0:
            return 0
        
        # Greedy coloring of the conflict graph on avail
        # Each color class is a set of vertices that are all pairwise connected in conflict graph
        # i.e., a clique in the conflict graph = independent set in the coprime graph
        # Wait, I need to think about this more carefully.
        # 
        # We want an independent set in the coprime graph = no two vertices share a factor > 1.
        # The conflict graph has edges between non-coprime pairs.
        # We want an independent set in the conflict graph (no edges between selected vertices).
        # 
        # Upper bound via clique cover: partition vertices into cliques of the conflict graph.
        # In each clique, at most one vertex can be in our independent set.
        # So upper bound = sum over cliques of (max weight in that clique).
        # 
        # Greedy clique cover = greedy coloring of the complement... no.
        # Actually, a clique cover of graph G = coloring of complement of G.
        # But we can just do greedy: repeatedly find a maximal clique, remove it.
        # Or equivalently, greedy coloring where adjacent vertices get different colors,
        # and each color class forms an independent set in G = clique in complement.
        # 
        # Let me just do: repeatedly pick a vertex, form a clique with all its uncolored neighbors,
        # remove them, repeat. Each group is a clique in the conflict graph.
        
        remaining = avail
        bound = 0
        while remaining:
            # Pick vertex with highest weight to start a clique
            best_v = -1
            best_w = -1
            tmp = remaining
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_v = idx
                tmp ^= lsb
            
            if best_v == -1:
                break
            
            # Greedily add vertices to this clique: must be adjacent to all in clique
            clique = 1 << best_v
            clique_sum = weights[best_v]
            # Candidates: neighbors of best_v in remaining
            candidates = neighbors[best_v] & remaining
            while candidates:
                # Pick highest weight candidate that is adjacent to all in clique
                # For simplicity, just pick highest weight neighbor of best_v
                # (this gives a star, not necessarily a clique, but it's still a valid upper bound
                #  if we treat it as: the clique contains best_v and these candidates,
                #  but they might not all be mutually adjacent... so this isn't a valid clique)
                # 
                # Better: just use best_v alone as a "clique" of size 1, or
                # find a proper maximal clique greedily.
                # 
                # For a tighter bound, let's do proper greedy clique:
                # Start with best_v, then repeatedly add the highest-weight vertex
                # that is adjacent to ALL current clique members.
                break  # skip for now, use simpler approach
            
            # Simple valid bound: each vertex is its own "clique" of size 1
            # bound += max weight in this "group" = weight of best_v
            # But that's just sum of all weights, which is weak.
            
            # Better simple bound: group by a common prime factor
            # Actually, let's use the standard approach:
            # Greedy coloring of the conflict graph. Each color class is an independent set
            # in the conflict graph = a clique in the coprime graph... no.
            # 
            # Independent set in conflict graph = set with no conflict edges = our desired set.
            # Upper bound: if we can partition V into k independent sets of the conflict graph,
            # that doesn't directly help.
            # 
            # The standard MWIS bound: color the graph, bound = sum of max weight per color.
            # Here we color the CONFLICT graph. Each color class is an independent set in conflict graph.
            # That's not what we want for a bound.
            # 
            # For MWIS on graph G: upper bound = sum over color classes of (max weight in class)
            # where the coloring is of G itself. Because each color class is an independent set in G,
            # and we can pick at most... no, that's for clique cover.
            # 
            # Let me reconsider. For Maximum Weight Independent Set on graph G:
            # - A clique in G: at most 1 vertex from each clique can be in the independent set.
            # - If we partition V into cliques C1, C2, ..., Ck of G, then
            #   bound = sum_i (max weight in Ci).
            # - Greedy clique cover: repeatedly find a maximal clique, remove it.
            #   This is equivalent to greedy coloring of the complement of G.
            # 
            # So I need to find cliques in the conflict graph.
            # A clique in the conflict graph = set of vertices all pairwise non-coprime.
            
            # Let's do: pick best_v, then greedily build a clique
            # A vertex can join if it's adjacent to all current clique members.
            new_clique = 1 << best_v
            new_clique_max = weights[best_v]
            candidates = neighbors[best_v] & remaining
            while candidates:
                # Find highest weight candidate adjacent to all in new_clique
                best_cand = -1
                best_cand_w = -1
                tmp = candidates
                while tmp:
                    lsb = tmp & (-tmp)
                    idx = lsb.bit_length() - 1
                    # Check if idx is adjacent to all in new_clique
                    ok = True
                    nq = new_clique
                    while nq:
                        lsb2 = nq & (-nq)
                        idx2 = lsb2.bit_length() - 1
                        if not (neighbors[idx] & (1 << idx2)):
                            ok = False
                            break
                        nq ^= lsb2
                    if ok and weights[idx] > best_cand_w:
                        best_cand_w = weights[idx]
                        best_cand = idx
                    tmp ^= lsb
                if best_cand == -1:
                    break
                new_clique |= (1 << best_cand)
                new_clique_max = max(new_clique_max, best_cand_w)
                # Update candidates: must be adjacent to best_cand too
                candidates &= neighbors[best_cand]
            
            bound += new_clique_max
            remaining &= ~new_clique
        
        return bound
    
    # Branch and bound
    def branch_and_bound(available, current_sum, current_mask):
        if time.time() > deadline:
            return
        
        if available == 0:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_mask[0] = current_mask
            return
        
        # Upper bound using clique cover
        ub = current_sum + clique_cover_bound(available)
        if ub <= best_sum[0]:
            return
        
        # Select vertex: highest weight among available
        best_v = -1
        best_w = -1
        tmp = available
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            if weights[idx] > best_w:
                best_w = weights[idx]
                best_v = idx
            tmp ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_available = available & can_coexist[best_v] & ~(1 << best_v)
        branch_and_bound(new_available, current_sum + weights[best_v], current_mask | (1 << best_v))
        
        # Branch 2: exclude best_v
        new_available2 = available & ~(1 << best_v)
        branch_and_bound(new_available2, current_sum, current_mask)
    
    branch_and_bound(all_mask, 0, 0)
    
    # Extract the set
    result_mask = best_mask[0]
    result = []
    for i in range(N):
        if result_mask & (1 << i):
            result.append(i + 1)
    
    # Ensure 1 is included if possible (it's coprime with everything)
    if 1 not in result:
        result.append(1)
    
    return [int(x) for x in result]
