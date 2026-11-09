import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # 1. Generate Paley(37)
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n0 = q
    adj = [0] * n0
    for i in range(n0):
        for j in range(i + 1, n0):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    current_n = n0
    
    # Helper to count K4 in subgraph induced by mask
    def count_k4_in_mask(adj, mask, n):
        count = 0
        # Get vertices in mask
        verts = [v for v in range(n) if (mask >> v) & 1]
        if len(verts) < 4:
            return 0
        # Iterate 4-combinations
        for combo in itertools.combinations(verts, 4):
            u, v, w, x = combo
            if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[u] >> x) & 1 and \
               (adj[v] >> w) & 1 and (adj[v] >> x) & 1 and \
               (adj[w] >> x) & 1:
                count += 1
        return count

    # Helper to count Independent 4-set in subgraph induced by mask
    def count_i4_in_mask(adj, mask, n):
        count = 0
        verts = [v for v in range(n) if (mask >> v) & 1]
        if len(verts) < 4:
            return 0
        for combo in itertools.combinations(verts, 4):
            u, v, w, x = combo
            if not ((adj[u] >> v) & 1) and not ((adj[u] >> w) & 1) and not ((adj[u] >> x) & 1) and \
               not ((adj[v] >> w) & 1) and not ((adj[v] >> x) & 1) and \
               not ((adj[w] >> x) & 1):
                count += 1
        return count

    def evaluate_extension(adj, n, mask):
        # mask is neighbors of new vertex n
        # Non-neighbors are ~mask & ((1 << n) - 1)
        non_mask = ((1 << n) - 1) ^ mask
        return count_k4_in_mask(adj, mask, n) + count_i4_in_mask(adj, non_mask, n)

    def try_add_vertex(adj, n, time_limit):
        # Try to find a mask for new vertex n
        best_mask = None
        best_cost = float('inf')
        
        all_vertices_mask = (1 << n) - 1
        
        for restart in range(20):
            if time.time() > time_limit:
                break
                
            # Initialize mask
            # Try different densities
            density = random.choice([0.3, 0.5, 0.7])
            mask = 0
            for i in range(n):
                if random.random() < density:
                    mask |= (1 << i)
            
            curr_cost = evaluate_extension(adj, n, mask)
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_mask = mask
                if curr_cost == 0:
                    return mask
            
            # Hill Climbing
            for step in range(2000):
                if time.time() > time_limit:
                    break
                    
                # Flip random bit
                i = random.randint(0, n - 1)
                new_mask = mask ^ (1 << i)
                new_cost = evaluate_extension(adj, n, new_mask)
                
                if new_cost <= curr_cost:
                    mask = new_mask
                    curr_cost = new_cost
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_mask = mask
                        if curr_cost == 0:
                            return mask
                # else revert (mask unchanged)
                
        return best_mask if best_cost == 0 else None

    # Attempt to extend
    target_n = 42
    for new_v in range(n0, target_n):
        if time.time() > deadline - 10:
            break
            
        # Set time limit for this vertex addition
        # Remaining time / remaining vertices
        remaining_time = deadline - time.time()
        remaining_verts = target_n - new_v
        per_vert_time = remaining_time / max(1, remaining_verts)
        
        # Allocate most of the time to search
        search_limit = time.time() + per_vert_time * 0.8
        
        mask = try_add_vertex(adj, new_v, search_limit)
        
        if mask is not None:
            # Add vertex new_v with neighbors in mask
            adj.append(0) # placeholder
            for i in range(new_v):
                if (mask >> i) & 1:
                    adj[i] |= (1 << new_v)
                    adj[new_v] |= (1 << i)
            current_n = new_v + 1
        else:
            # Failed to extend
            break
            
    # Final check and return
    # We should verify the final graph is valid
    # For efficiency, we assume the incremental check was correct, 
    # but a final check is safer if time permits.
    
    if time.time() < deadline - 5:
        # Quick check for K5 or I5 in final graph
        # This is expensive, so only do if n is small or time is ample
        # Given n<=42, checking all 5-sets is ~1M checks.
        # Let's do it.
        is_valid = True
        for v1 in range(current_n):
            for v2 in range(v1+1, current_n):
                for v3 in range(v2+1, current_n):
                    for v4 in range(v3+1, current_n):
                        for v5 in range(v4+1, current_n):
                            # Check K5
                            is_k5 = True
                            is_indep = True
                            pairs = [(v1,v2), (v1,v3), (v1,v4), (v1,v5), (v2,v3), (v2,v4), (v2,v5), (v3,v4), (v3,v5), (v4,v5)]
                            for u, v in pairs:
                                if (adj[u] >> v) & 1:
                                    is_indep = False
                                else:
                                    is_k5 = False
                                if not is_k5 and not is_indep:
                                    break
                            if is_k5 or is_indep:
                                is_valid = False
                                break
                        if not is_valid: break
                    if not is_valid: break
                if not is_valid: break
            if not is_valid: break
            
        if not is_valid:
            # Fallback to Paley 37
            return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

    edges = []
    for i in range(current_n):
        for j in range(i+1, current_n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
                
    return current_n, edges
