import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 295.0  # 5s safety margin

    # Base: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    base_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    base_n = q

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def adj_to_edges(n, adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges

    def count_violations_fast(n, adj):
        """Count K5 and independent 5 violations using bitset operations.
        Returns (k5_count, indep5_count) - fast early exit if either > 0."""
        k5_count = 0
        indep5_count = 0
        
        full_mask = (1 << n) - 1
        
        # Check K5: for each i, look for K4 in N(i)
        for i in range(n):
            ni = adj[i]
            # Get neighbors of i as list
            neighbors = []
            mask = ni
            while mask:
                lsb = mask & (-mask)
                j = lsb.bit_length() - 1
                neighbors.append(j)
                mask ^= lsb
            
            # For each pair (j, k) in neighbors of i, check if j-k edge exists
            # and if there's an edge in N(i) ∩ N(j) ∩ N(k)
            for idx_j in range(len(neighbors)):
                j = neighbors[idx_j]
                for idx_k in range(idx_j + 1, len(neighbors)):
                    k = neighbors[idx_k]
                    if not ((adj[j] >> k) & 1):
                        continue
                    # j and k are connected, both neighbors of i
                    # Find l, m in N(i) ∩ N(j) ∩ N(k) with l-m edge
                    common = ni & adj[j] & adj[k]
                    # Remove i, j, k from consideration
                    common &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                    # Find an edge in common
                    mask_l = common
                    while mask_l:
                        lsb_l = mask_l & (-mask_l)
                        l = lsb_l.bit_length() - 1
                        mask_l ^= lsb_l
                        # Check if l has a neighbor in common
                        mask_m = common & adj[l]
                        if mask_m:
                            k5_count += 1
                            if k5_count > 10:  # Early exit for speed
                                return (k5_count, indep5_count)
        
        if k5_count > 0:
            return (k5_count, indep5_count)
        
        # Check independent 5: for each i, look for independent 4 in non-N(i)
        for i in range(n):
            nonni = full_mask & ~(adj[i] | (1 << i))
            # Get non-neighbors of i as list
            non_neighbors = []
            mask = nonni
            while mask:
                lsb = mask & (-mask)
                j = lsb.bit_length() - 1
                non_neighbors.append(j)
                mask ^= lsb
            
            # For each pair (j, k) in non-neighbors of i, check if j-k is a non-edge
            # and if there's a non-edge in nonN(i) ∩ nonN(j) ∩ nonN(k)
            for idx_j in range(len(non_neighbors)):
                j = non_neighbors[idx_j]
                for idx_k in range(idx_j + 1, len(non_neighbors)):
                    k = non_neighbors[idx_k]
                    if (adj[j] >> k) & 1:
                        continue  # j-k edge exists, not independent
                    # j and k are non-adjacent, both non-neighbors of i
                    # Find l, m in nonN(i) ∩ nonN(j) ∩ nonN(k) with l-m non-edge
                    nonnj = full_mask & ~(adj[j] | (1 << j))
                    nonnk = full_mask & ~(adj[k] | (1 << k))
                    common = nonni & nonnj & nonnk
                    # Remove i, j, k from consideration
                    common &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                    # Find a non-edge in common
                    mask_l = common
                    while mask_l:
                        lsb_l = mask_l & (-mask_l)
                        l = lsb_l.bit_length() - 1
                        mask_l ^= lsb_l
                        # Check if l has a non-neighbor in common
                        mask_m = common & ~adj[l]
                        mask_m &= full_mask
                        if mask_m:
                            indep5_count += 1
                            if indep5_count > 10:  # Early exit for speed
                                return (k5_count, indep5_count)
        
        return (k5_count, indep5_count)

    def is_valid(n, adj):
        """Check if graph has no K5 and no independent 5."""
        k5, i5 = count_violations_fast(n, adj)
        return k5 == 0 and i5 == 0

    # Try to construct a 2-fold cover of Paley graph on 19 vertices
    # This creates a 38-vertex graph
    def construct_2fold_paley19():
        """Construct 2-fold cover of Paley graph on 19 vertices."""
        q19 = 19
        residues19 = {(x * x) % q19 for x in range(1, q19)}
        
        # Create two copies of 19 vertices: 0-18 and 19-37
        n = 38
        adj = [0] * n
        
        # Within each copy, use Paley graph edges
        for copy in range(2):
            offset = copy * 19
            for i in range(19):
                for j in range(i + 1, 19):
                    if (j - i) % q19 in residues19:
                        u = offset + i
                        v = offset + j
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # Between copies: connect vertex i in copy 0 to vertex j in copy 1
        # if i and j are adjacent in Paley(19)
        for i in range(19):
            for j in range(19):
                if i != j and (j - i) % q19 in residues19:
                    u = i  # in copy 0
                    v = 19 + j  # in copy 1
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        return n, adj

    # Local search with penalty-based objective
    def local_search_penalty(n, adj, time_limit, max_iters=100000):
        """Local search using penalty = k5_count + indep5_count as objective."""
        current_adj = adj[:]
        
        # Precompute full mask
        full_mask = (1 << n) - 1
        
        def get_penalty(adj):
            """Return (k5_count, indep5_count, total_penalty)."""
            k5 = 0
            i5 = 0
            
            # Check K5
            for i in range(n):
                ni = adj[i]
                neighbors = []
                mask = ni
                while mask:
                    lsb = mask & (-mask)
                    j = lsb.bit_length() - 1
                    neighbors.append(j)
                    mask ^= lsb
                
                for idx_j in range(len(neighbors)):
                    j = neighbors[idx_j]
                    for idx_k in range(idx_j + 1, len(neighbors)):
                        k = neighbors[idx_k]
                        if not ((adj[j] >> k) & 1):
                            continue
                        common = ni & adj[j] & adj[k]
                        common &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                        mask_l = common
                        while mask_l:
                            lsb_l = mask_l & (-mask_l)
                            l = lsb_l.bit_length() - 1
                            mask_l ^= lsb_l
                            mask_m = common & adj[l]
                            if mask_m:
                                k5 += 1
                                if k5 > 20:
                                    return (k5, i5, k5 + i5)
            
            if k5 > 0:
                return (k5, i5, k5 + i5)
            
            # Check independent 5
            for i in range(n):
                nonni = full_mask & ~(adj[i] | (1 << i))
                non_neighbors = []
                mask = nonni
                while mask:
                    lsb = mask & (-mask)
                    j = lsb.bit_length() - 1
                    non_neighbors.append(j)
                    mask ^= lsb
                
                for idx_j in range(len(non_neighbors)):
                    j = non_neighbors[idx_j]
                    for idx_k in range(idx_j + 1, len(non_neighbors)):
                        k = non_neighbors[idx_k]
                        if (adj[j] >> k) & 1:
                            continue
                        nonnj = full_mask & ~(adj[j] | (1 << j))
                        nonnk = full_mask & ~(adj[k] | (1 << k))
                        common = nonni & nonnj & nonnk
                        common &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                        mask_l = common
                        while mask_l:
                            lsb_l = mask_l & (-mask_l)
                            l = lsb_l.bit_length() - 1
                            mask_l ^= lsb_l
                            mask_m = common & ~adj[l]
                            mask_m &= full_mask
                            if mask_m:
                                i5 += 1
                                if i5 > 20:
                                    return (k5, i5, k5 + i5)
            
            return (k5, i5, k5 + i5)
        
        best_adj = current_adj[:]
        _, _, best_penalty = get_penalty(best_adj)
        
        if best_penalty == 0:
            return best_adj
        
        no_improve_count = 0
        iter_count = 0
        current_penalty = best_penalty
        
        while time.time() < time_limit and iter_count < max_iters:
            iter_count += 1
            
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Flip
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
            
            _, _, new_penalty = get_penalty(current_adj)
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_adj = current_adj[:]
                    no_improve_count = 0
                    if best_penalty == 0:
                        return best_adj
                else:
                    no_improve_count += 1
            else:
                # Revert
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
                no_improve_count += 1
            
            # Restart if stuck
            if no_improve_count > 500:
                # Perturb: flip a few random edges
                for _ in range(5):
                    pu = random.randint(0, n - 1)
                    pv = random.randint(0, n - 1)
                    if pu == pv:
                        continue
                    if pu > pv:
                        pu, pv = pv, pu
                    current_adj[pu] ^= (1 << pv)
                    current_adj[pv] ^= (1 << pu)
                _, _, current_penalty = get_penalty(current_adj)
                no_improve_count = 0
        
        return best_adj if best_penalty == 0 else None

    # First, verify the base Paley 37 is valid
    base_adj = build_adj(base_n, base_edges)
    if not is_valid(base_n, base_adj):
        # Should not happen, but fallback
        return base_n, base_edges
    
    # Try the 2-fold cover of Paley(19) to get 38 vertices
    n38, adj38 = construct_2fold_paley19()
    if is_valid(n38, adj38):
        edges38 = adj_to_edges(n38, adj38)
        return n38, edges38
    
    # If 38-vertex 2-fold cover is invalid, try local search to fix it
    time_slice = min(120.0, deadline - time.time() - 10)
    if time_slice > 10:
        end_time = time.time() + time_slice
        result = local_search_penalty(n38, adj38, end_time)
        if result is not None:
            edges38 = adj_to_edges(n38, result)
            return n38, edges38
    
    # Try growing from Paley 37 to 38, 39, 40
    for target_n in [38, 39, 40]:
        if time.time() > deadline - 30:
            break
        
        found = False
        
        # Multiple attempts with different random seeds for new vertex connections
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            # Initialize: start with Paley 37, add new vertices
            init_adj = base_adj[:]
            init_adj.extend([0] * (target_n - base_n))
            
            # Connect new vertices to existing ones
            for v in range(base_n, target_n):
                # Connect to ~half of existing vertices
                for u in range(v):
                    if random.random() < 0.5:
                        init_adj[u] |= (1 << v)
                        init_adj[v] |= (1 << u)
            
            # Run local search
            time_slice = min(20.0, deadline - time.time() - 10)
            if time_slice < 5:
                break
            end_time = time.time() + time_slice
            
            result = local_search_penalty(target_n, init_adj, end_time)
            
            if result is not None:
                # Valid (5,5)-free graph found
                edges = adj_to_edges(target_n, result)
                return target_n, edges
    
    # Fallback to seed
    return base_n, base_edges
