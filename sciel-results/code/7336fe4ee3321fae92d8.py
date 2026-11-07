import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy: Exploit the algebraic structure of the prism graph.
    The spectrum depends only on:
      P1 = product of signs on cycle 1
      P2 = product of signs on cycle 2  
      S  = sum of signs on matching edges
    For each (P1, P2, S), we construct the optimal sign assignment and check connectivity.
    """
    
    deadline = time.time() + 85.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Identify edge groups based on prism structure
    # Cycle 1: edges 0 to m-1
    # Cycle 2: edges m to 2m-1
    # Matching: edges 2m to 3m-1
    cycle1_edges = list(range(m))
    cycle2_edges = list(range(m, 2*m))
    matching_edges = list(range(2*m, 3*m))
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    # Precompute adjacency for connectivity check
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    def is_lift_connected(signs):
        """Check if the 2-lift is connected (not a coboundary)."""
        f = [-1] * n
        f[0] = 0
        queue = deque([0])
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        return True
        return False

    def get_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        S = np.zeros((n, n), dtype=np.float64)
        S[u_arr, v_arr] = signs.astype(float)
        S[v_arr, u_arr] = signs.astype(float)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    def make_signs(P1, P2, S):
        """
        Construct a sign assignment with:
        - Product of cycle 1 signs = P1 (+1 or -1)
        - Product of cycle 2 signs = P2 (+1 or -1)
        - Sum of matching signs = S (integer from -m to m)
        
        For cycles: to minimize spectral radius, we want signs as "balanced" as possible.
        For matching: exactly (m+S)/2 edges are +1, (m-S)/2 edges are -1.
        """
        signs = np.ones(num_edges, dtype=np.int8)
        
        # Cycle 1: m edges, product = P1
        # Optimal: if P1 == 1, use (m//2)+1 and (m//2)-1 negatives (or all +1 if m even)
        # If P1 == -1, use different pattern
        num_neg_c1 = 0
        if P1 == -1:
            num_neg_c1 = 1  # At least one -1
        # For even m, to get product +1: even number of -1s. Best is 0 or 2, etc.
        # For odd m, to get product +1: even number of -1s. Best is 0, 2, etc.
        # To get product -1: odd number of -1s. Best is 1, 3, etc.
        # The spectral radius is minimized when the cycle signing is "most balanced"
        # which means the number of -1s is as close to m/2 as possible while satisfying the product constraint.
        
        # For cycle 1:
        if P1 == 1:
            # Even number of -1s
            if m % 2 == 0:
                # Can have 0, 2, 4, ... -1s. Best is 0 (all +1) or 2
                # All +1 gives product +1 and is most balanced
                num_neg_c1 = 0
            else:
                # m is odd, even number of -1s: 0, 2, 4, ...
                # Best is 0 (all +1)
                num_neg_c1 = 0
        else:  # P1 == -1
            # Odd number of -1s: 1, 3, 5, ...
            if m % 2 == 1:
                # m is odd, odd number of -1s: 1, 3, ...
                # Best is 1
                num_neg_c1 = 1
            else:
                # m is even, odd number of -1s: 1, 3, ...
                # Best is 1
                num_neg_c1 = 1
        
        # Assign -1 to the first num_neg_c1 edges of cycle 1
        for i in range(num_neg_c1):
            signs[cycle1_edges[i]] = -1
        
        # Cycle 2: similar logic
        num_neg_c2 = 0
        if P2 == 1:
            num_neg_c2 = 0
        else:  # P2 == -1
            num_neg_c2 = 1
        
        for i in range(num_neg_c2):
            signs[cycle2_edges[i]] = -1
        
        # Matching: exactly (m+S)/2 edges are +1, (m-S)/2 edges are -1
        num_plus_match = (m + S) // 2
        num_neg_match = m - num_plus_match
        
        # Assign -1 to the first num_neg_match edges of matching
        for i in range(num_neg_match):
            signs[matching_edges[i]] = -1
        
        return signs

    best_signs = None
    best_radius = np.inf
    
    # Enumerate all possible (P1, P2, S) combinations
    # P1, P2 in {+1, -1}
    # S in {-m, -m+2, ..., m-2, m} (step 2 since m+S must be even)
    
    for P1 in [1, -1]:
        for P2 in [1, -1]:
            for S in range(-m, m+1, 2):
                if time.time() > deadline - 2.0:
                    break
                
                signs = make_signs(P1, P2, S)
                
                if not is_lift_connected(signs):
                    continue
                
                radius = get_radius(signs)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
    
    # If the structured approach didn't find a good solution, fall back to local search
    if best_signs is None or best_radius > 2.5:
        # Run local search from the best found solution or a random start
        if best_signs is None:
            best_signs = np.ones(num_edges, dtype=np.int8)
        
        # Try local search to improve
        current_signs = best_signs.copy()
        if not is_lift_connected(current_signs):
            for _ in range(100):
                idx = np.random.randint(num_edges)
                current_signs[idx] *= -1
                if is_lift_connected(current_signs):
                    break
            if not is_lift_connected(current_signs):
                return best_signs
        
        current_radius = get_radius(current_signs)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 5.0:
                    break
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                if not is_lift_connected(current_signs):
                    current_signs[i] = old_sign
                    continue
                new_radius = get_radius(current_signs)
                if new_radius < current_radius - 1e-9:
                    current_radius = new_radius
                    improved = True
                else:
                    current_signs[i] = old_sign
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()
    
    return best_signs
