"""Program for spectral graph lifts task."""
import numpy as np

# EVOLVE-BLOCK-START
def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    # Build adjacency structure
    # Two m-cycles: vertices 0..m-1 (top) and m..2m-1 (bottom)
    # Top cycle: (i, i+1) for i in 0..m-1 (with wrap-around)
    # Bottom cycle: (m+i, m+i+1) for i in 0..m-1 (with wrap-around)
    # Matching: (i, m+i) for i in 0..m-1
    
    # We'll assign signs to each edge
    # The signed adjacency matrix A has entries:
    # A[u,v] = sign for edge (u,v), and A[v,u] = sign for edge (u,v)
    # (symmetric)
    
    # Spectral radius = max |eigenvalue| of A
    
    # Strategy: Use continuous phase optimization
    # Parameterize each vertex with a complex unit vector z_v = exp(i * theta_v)
    # Edge sign: sign(Re(z_u * conj(z_v)))
    # Optimize theta to minimize spectral radius
    
    # But wait - we need the lift to be connected. Let's first find good signs,
    # then verify connectivity.
    
    # Actually, let's think about this more carefully.
    # The signed adjacency matrix of the base graph determines the eigenvalues
    # of the lift. The lift is 2n vertices, 3-regular.
    
    # For a prism graph P_m:
    # - Top cycle edges: m edges
    # - Bottom cycle edges: m edges  
    # - Matching edges: m edges
    # Total: 3m edges = 3*(n/2) edges. Since n = 2m, that's 3n/2 edges.
    
    # Let me construct the signed adjacency matrix directly.
    
    def build_signed_adj(signs):
        """Build signed adjacency matrix from edge signs."""
        A = np.zeros((n, n))
        for (u, v), s in zip(edges, signs):
            A[u, v] = s
            A[v, u] = s
        return A
    
    def spectral_radius(A):
        """Compute spectral radius (max abs eigenvalue)."""
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            return True, radius
        except Exception:
            return False, None
    
    # Phase-based optimization
    # Each vertex v has a phase theta_v in [0, 2*pi)
    # Edge (u,v) sign = sign(Re(exp(i*theta_u) * exp(-i*theta_v)))
    #                = sign(Re(exp(i*(theta_u - theta_v))))
    #                = sign(cos(theta_u - theta_v))
    
    # So the sign depends on the phase difference.
    # If |theta_u - theta_v| < pi/2 or > 3pi/2, sign = +1
    # Otherwise sign = -1
    
    # We want to minimize the spectral radius of the resulting signed adjacency.
    
    # Let's use gradient-free optimization since the sign function is discontinuous.
    # Instead, we can use a smooth approximation: 
    # f(theta_u, theta_v) = cos(theta_u - theta_v) instead of sign(cos(...))
    # Then the "signed" adjacency becomes a real symmetric matrix with entries
    # cos(theta_u - theta_v) for edges, and we minimize its spectral radius.
    
    # But the actual sign is sign(cos(...)), so we need to be careful.
    
    # Alternative approach: directly search over sign assignments using simulated
    # annealing or other methods, with the spectral radius as the objective.
    
    # Given the small size (m = 4,5,7,9, so n = 8,10,14,18, and 3m = 12,15,21,27 edges),
    # we can do a more exhaustive search.
    
    # Let's try: start with all +1, then use local search (flip individual edges)
    # to minimize spectral radius, with occasional random restarts.
    
    # First, let's establish a good baseline with the seed approach.
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    # Try the seed's budget random samples
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=len(edges))
        valid, radius = is_valid_lift(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # Now do local search from the best found
    if best_signs is not None:
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        # Local search: try flipping each edge
        for iteration in range(200):  # Limit iterations for time
            improved = False
            for i in range(len(edges)):
                # Flip edge i
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                valid, radius = is_valid_lift(current_signs)
                
                if valid and radius < current_radius:
                    current_radius = radius
                    improved = True
                    if radius < best_radius:
                        best_radius = radius
                        best_signs = current_signs.copy()
                else:
                    current_signs[i] = old_sign
            
            if not improved:
                break
    
    # Also try some structured patterns
    # Pattern 1: all +1
    signs_all_plus = np.ones(len(edges), dtype=int)
    valid, radius = is_valid_lift(signs_all_plus)
    if valid and radius < best_radius:
        best_radius = radius
        best_signs = signs_all_plus.copy()
    
    # Pattern 2: alternating signs on cycles
    # Top cycle: + - + - ...
    # Bottom cycle: + - + - ...
    # Matching: +1
    for pattern_type in range(4):
        signs = np.ones(len(edges), dtype=int)
        for idx, (u, v) in enumerate(edges):
            if u < m and v < m:  # Top cycle
                # Determine position in cycle
                if v == (u + 1) % m:
                    signs[idx] = 1 if (u % 2 == pattern_type % 2) else -1
            elif u >= m and v >= m:  # Bottom cycle
                u2 = u - m
                v2 = v - m
                if v2 == (u2 + 1) % m:
                    signs[idx] = 1 if (u2 % 2 == pattern_type % 2) else -1
            else:  # Matching
                signs[idx] = 1
        
        valid, radius = is_valid_lift(signs)
        if valid and radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    # Try more random restarts with local search
    for restart in range(10):
        signs = rng.choice([-1, 1], size=len(edges))
        valid, radius = is_valid_lift(signs)
        if not valid:
            continue
        
        current_signs = signs.copy()
        current_radius = radius
        
        # Local search
        for iteration in range(50):
            improved = False
            for i in range(len(edges)):
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                valid2, radius2 = is_valid_lift(current_signs)
                
                if valid2 and radius2 < current_radius:
                    current_radius = radius2
                    improved = True
                    if radius2 < best_radius:
                        best_radius = radius2
                        best_signs = current_signs.copy()
                else:
                    current_signs[i] = old_sign
            
            if not improved:
                break
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(len(edges), dtype=int)
    
    return best_signs.astype(int)
# EVOLVE-BLOCK-END
