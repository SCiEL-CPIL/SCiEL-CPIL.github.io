import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # Target n
    n_target = 42
    
    # Helper to generate Paley 37 edges
    def get_paley_edges(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return edges

    paley_edges = get_paley_edges(37)
    
    # Function to count edges in a subgraph defined by a bitmask
    # adj_mask is a list of ints
    def count_edges_in_mask(mask, adj_mask):
        count = 0
        m = mask
        while m:
            # Extract lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            m ^= lsb
            # Count bits in intersection of adj_mask[idx] and remaining m
            count += (adj_mask[idx] & m).bit_count()
        return count

    # Function to calculate total cost (K5 + Ind5) for a graph represented by adj_mask
    # This is used for final verification and initial cost
    def calculate_total_cost(adj_mask, n):
        k5_count = 0
        ind5_count = 0
        # Iterate over all 5-combinations
        # For performance, we can use the property that a K5 is a set of 5 vertices 
        # where each is connected to all others.
        # With bitmasks, for a combination (v1, v2, v3, v4, v5):
        # Check if v5 is in adj[v1] & adj[v2] & adj[v3] & adj[v4]
        # Check if v5 is NOT in adj[v1] | adj[v2] | adj[v3] | adj[v4] for independent
        
        # This is O(C(n,5)) which is ~850k for n=42.
        # 850k iterations in Python might take ~1s.
        
        for comb in itertools.combinations(range(n), 5):
            v1, v2, v3, v4, v5 = comb
            
            # K5 check
            mask = adj_mask[v1] & adj_mask[v2] & adj_mask[v3] & adj_mask[v4]
            if mask & (1 << v5):
                k5_count += 1
                
            # Independent set check
            # v5 must not be connected to any of v1..v4
            # i.e., v5 is not in adj[v1] | adj[v2] | adj[v3] | adj[v4]
            union_mask = adj_mask[v1] | adj_mask[v2] | adj_mask[v3] | adj_mask[v4]
            if not (union_mask & (1 << v5)):
                ind5_count += 1
                
        return k5_count + ind5_count

    # Initialize graph
    # Start with Paley 37
    adj_mask = [0] * n_target
    all_vertices_mask = (1 << n_target) - 1
    
    for u, v in paley_edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Initialize vertices 37-41 randomly
    random.seed(42)
    for i in range(37, n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
        for k in range(i):
            if random.random() < 0.5:
                adj_mask[i] |= (1 << k)
                adj_mask[k] |= (1 << i)

    # Precompute common neighbor masks for all pairs?
    # common_neigh_mask[u][v] = adj_mask[u] & adj_mask[v] & ~(1<<u) & ~(1<<v)
    # We don't need to store this if we can compute it on the fly.
    # adj_mask[u] & adj_mask[v] is fast.
    
    current_cost = calculate_total_cost(adj_mask, n_target)
    best_cost = current_cost
    best_adj_mask = adj_mask[:]
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    steps = 0
    last_check_time = time.time()
    
    while time.time() < deadline:
        # Pick random edge to flip
        u = random.randint(0, n_target - 1)
        v = random.randint(0, n_target - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        edge_exists = bool(adj_mask[u] & (1 << v))
        
        # Calculate Delta Cost
        # Common neighbors of u and v
        common_neigh = adj_mask[u] & adj_mask[v]
        # Remove u and v from common_neigh (they are not neighbors of themselves)
        common_neigh &= ~(1 << u)
        common_neigh &= ~(1 << v)
        
        # Triangles in common_neigh
        # This is the number of edges in the subgraph induced by common_neigh
        delta_k5 = count_edges_in_mask(common_neigh, adj_mask)
        
        # Common non-neighbors of u and v
        # Nodes w such that w is not neighbor of u AND w is not neighbor of v
        # ~(adj_mask[u] | adj_mask[v]) & all_vertices_mask
        common_non_neigh = ~(adj_mask[u] | adj_mask[v]) & all_vertices_mask
        common_non_neigh &= ~(1 << u)
        common_non_neigh &= ~(1 << v)
        
        # Independent triangles in common_non_neigh
        # This is the number of edges in the COMPLEMENT graph induced by common_non_neigh
        # Or, number of non-edges in G induced by common_non_neigh
        # count_edges_in_mask counts edges in G.
        # We want to count edges in G_bar, which is (total_pairs - edges_in_G)
        # Total pairs in a set of size k is k*(k-1)/2
        k = common_non_neigh.bit_count()
        total_pairs = k * (k - 1) // 2
        edges_in_g = count_edges_in_mask(common_non_neigh, adj_mask)
        edges_in_g_bar = total_pairs - edges_in_g
        
        delta_ind = edges_in_g_bar
        
        if edge_exists:
            # Removing edge
            # We lose K5s containing this edge: -delta_k5
            # We gain Ind5s containing this edge (because it was blocking them): +delta_ind
            delta_cost = -delta_k5 + delta_ind
        else:
            # Adding edge
            # We gain K5s: +delta_k5
            # We lose Ind5s: -delta_ind
            delta_cost = delta_k5 - delta_ind
            
        # Acceptance
        if delta_cost < 0 or random.random() < pow(2.71828, -delta_cost / T):
            # Apply move
            if edge_exists:
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
            current_cost += delta_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj_mask = adj_mask[:]
                
                if best_cost == 0:
                    # Verify!
                    # It's possible the incremental logic has a bug or floating point issue (though we use ints)
                    # But it's safer to verify before returning.
                    # Verification takes ~1s. We have time.
                    if calculate_total_cost(best_adj_mask, n_target) == 0:
                        edges = []
                        for i in range(n_target):
                            for j in range(i + 1, n_target):
                                if best_adj_mask[i] & (1 << j):
                                    edges.append((i, j))
                        return n_target, edges

        T *= cooling
        steps += 1
        
        # Periodic full check to correct any drift if incremental logic is slightly off?
        # The incremental logic is exact for integer counts, so drift shouldn't happen.
        # But if we want to be extra safe, we can occasionally recompute current_cost.
        if steps % 1000 == 0:
            current_cost = calculate_total_cost(adj_mask, n_target)
            if current_cost < best_cost:
                best_cost = current_cost
                best_adj_mask = adj_mask[:]
                if best_cost == 0:
                     if calculate_total_cost(best_adj_mask, n_target) == 0:
                        edges = []
                        for i in range(n_target):
                            for j in range(i + 1, n_target):
                                if best_adj_mask[i] & (1 << j):
                                    edges.append((i, j))
                        return n_target, edges

    # If we didn't find a valid 42-graph, check if best_adj_mask is valid (cost 0)
    if best_cost == 0:
        if calculate_total_cost(best_adj_mask, n_target) == 0:
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if best_adj_mask[i] & (1 << j):
                        edges.append((i, j))
            return n_target, edges

    # Fallback: Return Paley 37
    return 37, paley_edges
