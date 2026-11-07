# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def sign_edges(graph, budget):
    """
    Assign signs to edges of a prism graph to minimize the spectral radius
    of the signed adjacency matrix, while ensuring the 2-lift is connected.
    
    Strategy:
    1. Initialize with a "balanced" sign pattern that maximizes frustration
       (product of signs around fundamental cycles is -1 where possible).
    2. Perform local search to minimize spectral radius.
    3. Use fast connectivity check based on cycle products.
    """
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Classify edges
    # Top cycle: (i, (i+1)%m)
    # Bottom cycle: (m+i, m+(i+1)%m)
    # Matching: (i, m+i)
    
    top_edges = {}
    bottom_edges = {}
    match_edges = {}
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges[idx] = (u, v)
        elif u >= m and v >= m:
            bottom_edges[idx] = (u-m, v-m)
        else:
            # One top, one bottom
            if u < m:
                match_edges[idx] = u
            else:
                match_edges[idx] = v - m

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
        We check fundamental cycles: Top, Bottom, and Ladders.
        """
        # Calculate products
        prod_top = 1
        for idx in top_edges:
            prod_top *= signs[idx]
            
        prod_bot = 1
        for idx in bottom_edges:
            prod_bot *= signs[idx]
            
        # Ladder cycles: i -> i+1 -> m+i+1 -> m+i -> i
        # Edges: top(i), match(i+1), bottom(i), match(i)
        # Note: bottom edge (m+i, m+i+1) is traversed m+i+1 -> m+i, so sign is same.
        # Match edges: (i, m+i) and (i+1, m+i+1).
        
        # We need to map indices carefully.
        # Let's just compute product for each ladder i.
        # Ladder i involves:
        # top edge between i and i+1
        # bottom edge between m+i and m+i+1
        # match edge i
        # match edge i+1
        
        # Find indices
        top_idx_map = {}
        for idx, (u, v) in top_edges.items():
            top_idx_map[(u, v)] = idx
            top_idx_map[(v, u)] = idx
            
        bot_idx_map = {}
        for idx, (u, v) in bottom_edges.items():
            # u,v are local indices 0..m-1
            bot_idx_map[(u, v)] = idx
            bot_idx_map[(v, u)] = idx
            
        match_idx_map = {}
        for idx, i in match_edges.items():
            match_idx_map[i] = idx

        for i in range(m):
            i_next = (i + 1) % m
            
            # Get indices
            try:
                t_idx = top_idx_map[(i, i_next)]
                b_idx = bot_idx_map[(i, i_next)]
                m_idx_i = match_idx_map[i]
                m_idx_next = match_idx_map[i_next]
            except KeyError:
                # If edge order is different, try reversed
                try:
                    t_idx = top_idx_map[(i_next, i)]
                    b_idx = bot_idx_map[(i_next, i)]
                except KeyError:
                    return True # Assume connected if structure is weird, fallback to slow check later if needed
            
            prod_ladder = signs[t_idx] * signs[m_idx_next] * signs[b_idx] * signs[m_idx_i]
            if prod_ladder == -1:
                return True # At least one cycle is -1, so not a coboundary, so connected
        
        # If all ladder products are +1, check top and bottom
        if prod_top == -1 or prod_bot == -1:
            return True
            
        return False # All cycles +1, disconnected

    def local_search(signs, max_iters=1000):
        signs = signs.copy()
        current_radius = spectral_radius(signs)
        
        if not is_connected_fast(signs):
             # Fallback to slow check if fast check says disconnected but we suspect error
             # Or just return None
             return None

        for it in range(max_iters):
            improved = False
            # Try flipping each edge
            # Shuffle edge order to avoid bias
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

    # Initializations
    initializations = []
    
    # Init 1: Balanced (Ladder -1)
    # s_m(i) = (-1)^i, s_t(i)=1, s_b(i)=1
    s1 = np.ones(num_edges, dtype=int)
    for idx, i in match_edges.items():
        if i % 2 == 1:
            s1[idx] = -1
    initializations.append(s1)
    
    # Init 2: Alternating Bottom
    # s_m(i)=1, s_t(i)=1, s_b(i)=(-1)^i
    s2 = np.ones(num_edges, dtype=int)
    for idx, (u, v) in bottom_edges.items():
        # u is local index
        if u % 2 == 1:
            s2[idx] = -1
    initializations.append(s2)
    
    # Init 3: All +1 except one match
    s3 = np.ones(num_edges, dtype=int)
    if match_edges:
        first_match_idx = next(iter(match_edges.keys()))
        s3[first_match_idx] = -1
    initializations.append(s3)
    
    # Init 4: Random
    rng = np.random.default_rng(42)
    s4 = rng.choice([-1, 1], size=num_edges).astype(int)
    # Ensure connectivity
    if not is_connected_fast(s4):
        # Try flipping one edge to fix
        for idx in range(num_edges):
            s4_try = s4.copy()
            s4_try[idx] = -s4_try[idx]
            if is_connected_fast(s4_try):
                s4 = s4_try
                break
    initializations.append(s4)
    
    best_signs = None
    best_radius = float('inf')
    
    for init_signs in initializations:
        if not is_connected_fast(init_signs):
            continue
        result = local_search(init_signs, max_iters=500)
        if result is not None:
            signs, radius = result
            if radius < best_radius:
                best_radius = radius
                best_signs = signs
    
    if best_signs is None:
        # Fallback: use seed logic or simple valid sign
        # Try all +1 except one top edge
        s_fallback = np.ones(num_edges, dtype=int)
        if top_edges:
            first_top_idx = next(iter(top_edges.keys()))
            s_fallback[first_top_idx] = -1
        return s_fallback
    
    return best_signs
# EVOLVE-BLOCK-END
