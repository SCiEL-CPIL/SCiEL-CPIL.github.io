# EVOLVE-BLOCK-START
import numpy as np

def sign_edges(graph, budget):
    """
    Assign signs to edges of a prism graph to minimize the spectral radius
    of the signed adjacency matrix, while ensuring the 2-lift is connected.
    
    Strategy:
    1. Construct a sign vector that enforces -1 product on fundamental cycles
       where possible (analytic balancing).
    2. For even m, this satisfies all constraints and yields minimal radius.
    3. For odd m, this satisfies ladder constraints and one cycle constraint.
    4. Refine with local search if time permits (though analytic is likely optimal).
    """
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Classify edges into Top, Bottom, Matching
    # Top: u < m, v < m
    # Bottom: u >= m, v >= m
    # Match: one < m, one >= m
    
    top_idx = {}
    bot_idx = {}
    match_idx = {}
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            # Ensure consistent ordering for mapping? 
            # The edge is undirected, but we store index.
            # We need to know which vertex is i and which is i+1 for cycle products?
            # Actually, for product, order doesn't matter.
            # But for construction, we need to map edge to 'position' i in the cycle.
            # Vertices are 0..m-1. Edge (i, i+1)%m.
            # Let's store the smaller vertex or just the pair.
            top_idx[idx] = (u, v)
        elif u >= m and v >= m:
            bot_idx[idx] = (u-m, v-m)
        else:
            if u < m:
                match_idx[idx] = u
            else:
                match_idx[idx] = v - m

    def build_signed_adjacency(signs):
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
        return A

    def spectral_radius(signs):
        A = build_signed_adjacency(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_connected_fast(signs):
        """
        Check connectivity by verifying that the sign vector is not a coboundary.
        A sign vector is a coboundary iff product of signs around every cycle is +1.
        We check fundamental cycles.
        """
        # Calculate products
        prod_top = 1
        for idx in top_idx:
            prod_top *= signs[idx]
            
        prod_bot = 1
        for idx in bot_idx:
            prod_bot *= signs[idx]
            
        # Ladder cycles: i -> i+1 -> m+i+1 -> m+i -> i
        # Edges: top(i), match(i+1), bottom(i), match(i)
        
        # Map indices for ladder calculation
        # We need to find the edge index for top(i, i+1), bottom(i, i+1), match(i)
        
        top_map = {}
        for idx, (u, v) in top_idx.items():
            top_map[(u, v)] = idx
            top_map[(v, u)] = idx
            
        bot_map = {}
        for idx, (u, v) in bot_idx.items():
            bot_map[(u, v)] = idx
            bot_map[(v, u)] = idx
            
        match_map = {}
        for idx, i in match_idx.items():
            match_map[i] = idx

        for i in range(m):
            i_next = (i + 1) % m
            
            try:
                t_idx = top_map[(i, i_next)]
                b_idx = bot_map[(i, i_next)]
                m_idx_i = match_map[i]
                m_idx_next = match_map[i_next]
            except KeyError:
                # Fallback: if map fails, assume connected to be safe? 
                # Or try reversed keys.
                try:
                    t_idx = top_map[(i_next, i)]
                    b_idx = bot_map[(i_next, i)]
                except KeyError:
                    return True
            
            prod_ladder = signs[t_idx] * signs[m_idx_next] * signs[b_idx] * signs[m_idx_i]
            if prod_ladder == -1:
                return True # At least one cycle is -1, so not a coboundary, so connected
        
        if prod_top == -1 or prod_bot == -1:
            return True
            
        return False # All cycles +1, disconnected

    def local_search(signs, max_iters=200):
        signs = signs.copy()
        current_radius = spectral_radius(signs)
        
        if not is_connected_fast(signs):
            return None

        for it in range(max_iters):
            improved = False
            order = np.random.permutation(num_edges)
            
            for idx in order:
                new_signs = signs.copy()
                new_signs[idx] = -new_signs[idx]
                
                if not is_connected_fast(new_signs):
                    continue
                
                r = spectral_radius(new_signs)
                if r < current_radius - 1e-12:
                    signs = new_signs
                    current_radius = r
                    improved = True
                    break # Restart search from new state
            
            if not improved:
                break
        
        return signs, current_radius

    # Construct analytic initialization
    signs = np.ones(num_edges, dtype=int)
    
    if m % 2 == 0:
        # Even m: Satisfy all constraints
        # m_i = (-1)^i
        # t_i = 1, except t_0 = -1
        # b_i = 1, except b_0 = -1
        
        # Assign matching signs
        for idx, i in match_idx.items():
            if i % 2 == 1:
                signs[idx] = -1
            else:
                signs[idx] = 1
        
        # Assign top signs
        # Find edge index for top edge (0, 1)
        # It might be (0,1) or (1,0)
        target_top_0 = None
        for idx, (u, v) in top_idx.items():
            if (u == 0 and v == 1) or (u == 1 and v == 0):
                target_top_0 = idx
                break
        if target_top_0 is not None:
            signs[target_top_0] = -1
        else:
            # Fallback if edge not found (should not happen)
            # Just flip first top edge
            if top_idx:
                signs[next(iter(top_idx))] = -1

        # Assign bottom signs
        target_bot_0 = None
        for idx, (u, v) in bot_idx.items():
            if (u == 0 and v == 1) or (u == 1 and v == 0):
                target_bot_0 = idx
                break
        if target_bot_0 is not None:
            signs[target_bot_0] = -1
        else:
            if bot_idx:
                signs[next(iter(bot_idx))] = -1

    else:
        # Odd m: 
        # m_i = 1
        # t_i = 1
        # b_i = -1
        # This gives Ladders -1, Top 1, Bottom -1
        
        for idx in bot_idx:
            signs[idx] = -1
        # Top and Match remain 1

    # Verify connectivity
    if not is_connected_fast(signs):
        # If somehow disconnected (shouldn't be for this construction), fallback
        # Flip one edge to make it connected
        for idx in range(num_edges):
            s_try = signs.copy()
            s_try[idx] = -s_try[idx]
            if is_connected_fast(s_try):
                signs = s_try
                break

    # Refine with local search (short budget)
    # The analytic solution is likely optimal or near-optimal.
    # Local search might find a better one for odd m if the analytic one isn't perfectly balanced.
    result = local_search(signs, max_iters=100)
    if result is not None:
        final_signs, radius = result
        return final_signs
    
    return signs
# EVOLVE-BLOCK-END
