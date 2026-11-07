import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Precompute edge structure
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    # Identify edge types: bottom cycle, top cycle, matching
    bottom_cycle_edges = []
    top_cycle_edges = []
    matching_edges = []
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_cycle_edges.append(i)
        elif u >= m and v >= m:
            top_cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    def build_signed_adj(signs):
        A = np.zeros((n, n))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        return A
    
    def spectral_radius(signs):
        A = build_signed_adj(signs)
        eigvals = np.linalg.eigvalsh(A)
        return max(abs(eigvals[0]), abs(eigvals[-1]))
    
    def power_iteration_radius(signs, max_iter=100, tol=1e-10):
        """Fast spectral radius estimate via power iteration on |A|."""
        A = build_signed_adj(signs)
        x = np.ones(n) / np.sqrt(n)
        for _ in range(max_iter):
            x_new = A @ x
            norm = np.linalg.norm(x_new)
            if norm < 1e-15:
                return 0.0
            x_new = x_new / norm
            if np.linalg.norm(x_new - x) < tol:
                return norm
            x = x_new
        return np.linalg.norm(A @ x)
    
    def is_connected_lift(signs):
        """Check if the 2-lift is connected using union-find."""
        num_lift_vertices = 2 * n
        parent = list(range(num_lift_vertices))
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        
        for idx in range(num_edges):
            u = edge_u[idx]
            v = edge_v[idx]
            s = signs[idx]
            if s == 1:
                union(u * 2, v * 2)
                union(u * 2 + 1, v * 2 + 1)
            else:
                union(u * 2, v * 2 + 1)
                union(u * 2 + 1, v * 2)
        
        root = find(0)
        for i in range(1, num_lift_vertices):
            if find(i) != root:
                return False
        return True
    
    def local_search(signs, max_iterations=50, use_power=True):
        """Flip individual signs to reduce spectral radius."""
        current_signs = signs.copy()
        if use_power:
            current_metric = power_iteration_radius(current_signs)
        else:
            current_metric = spectral_radius(current_signs)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            improved = False
            for idx in range(num_edges):
                trial_signs = current_signs.copy()
                trial_signs[idx] *= -1
                if not is_connected_lift(trial_signs):
                    continue
                if use_power:
                    trial_metric = power_iteration_radius(trial_signs)
                else:
                    trial_metric = spectral_radius(trial_signs)
                if trial_metric < current_metric - 1e-10:
                    current_signs = trial_signs
                    current_metric = trial_metric
                    improved = True
                    break
            if not improved:
                break
        
        return current_signs
    
    def make_signing_from_pattern(bottom_pattern, top_pattern, match_pattern):
        """Construct signing from cycle patterns and matching pattern."""
        signs = np.ones(num_edges, dtype=int)
        for j, idx in enumerate(bottom_cycle_edges):
            signs[idx] = bottom_pattern[j % len(bottom_pattern)]
        for j, idx in enumerate(top_cycle_edges):
            signs[idx] = top_pattern[j % len(top_pattern)]
        for j, idx in enumerate(matching_edges):
            signs[idx] = match_pattern[j % len(match_pattern)]
        return signs
    
    best_signs = None
    best_radius = float('inf')
    
    def try_candidate(signs):
        nonlocal best_signs, best_radius
        if time.time() > deadline - 2:
            return
        if not is_connected_lift(signs):
            return
        r = spectral_radius(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    # Phase 1: Structured search over frustration patterns
    # For prism graphs, key insight: the spectral radius depends on the
    # frustration (product of signs) around each cycle and the matching.
    
    # Try all possible numbers of -1s in each cycle, distributed in different ways
    for num_neg_bottom in range(0, m + 1):
        if time.time() > deadline - 10:
            break
        for num_neg_top in range(0, m + 1):
            if time.time() > deadline - 10:
                break
            # Try different distributions of negative signs
            for dist_type in ['consecutive', 'alternating', 'sparse']:
                if time.time() > deadline - 10:
                    break
                
                bottom_pattern = np.ones(m, dtype=int)
                top_pattern = np.ones(m, dtype=int)
                
                if num_neg_bottom > 0:
                    if dist_type == 'consecutive':
                        for j in range(num_neg_bottom):
                            bottom_pattern[j] = -1
                    elif dist_type == 'alternating':
                        for j in range(0, m, 2):
                            if num_neg_bottom > 0 and j < m:
                                bottom_pattern[j] = -1
                                num_neg_bottom -= 1
                                if num_neg_bottom == 0:
                                    break
                    else:  # sparse
                        step = max(1, m // max(1, num_neg_bottom))
                        for j in range(0, m, step):
                            if num_neg_bottom > 0:
                                bottom_pattern[j] = -1
                                num_neg_bottom -= 1
                
                if num_neg_top > 0:
                    if dist_type == 'consecutive':
                        for j in range(num_neg_top):
                            top_pattern[j] = -1
                    elif dist_type == 'alternating':
                        for j in range(0, m, 2):
                            if num_neg_top > 0 and j < m:
                                top_pattern[j] = -1
                                num_neg_top -= 1
                                if num_neg_top == 0:
                                    break
                    else:  # sparse
                        step = max(1, m // max(1, num_neg_top))
                        for j in range(0, m, step):
                            if num_neg_top > 0:
                                top_pattern[j] = -1
                                num_neg_top -= 1
                
                # Try different matching patterns
                for match_type in ['all_plus', 'all_minus', 'alternating']:
                    if time.time() > deadline - 10:
                        break
                    
                    match_pattern = np.ones(m, dtype=int)
                    if match_type == 'all_minus':
                        match_pattern[:] = -1
                    elif match_type == 'alternating':
                        match_pattern[1::2] = -1
                    
                    signs = make_signing_from_pattern(
                        bottom_pattern.tolist(), 
                        top_pattern.tolist(), 
                        match_pattern.tolist()
                    )
                    try_candidate(signs)
    
    # Phase 2: Random search with bias toward frustration
    rng = np.random.default_rng(3)
    num_random_trials = 500 if m <= 7 else 2000
    
    for trial in range(num_random_trials):
        if time.time() > deadline - 15:
            break
        
        # Bias: more likely to have odd number of -1s (frustrated cycles)
        signs = np.ones(num_edges, dtype=int)
        
        # Randomly flip edges with probability that favors frustration
        for idx in range(num_edges):
            if rng.random() < 0.3:
                signs[idx] = -1
        
        try_candidate(signs)
    
    # Phase 3: Local search from best candidates
    if best_signs is not None:
        # Quick polish with power iteration
        if time.time() < deadline - 15:
            polished = local_search(best_signs, max_iterations=30, use_power=True)
            r = spectral_radius(polished)
            if r < best_radius:
                best_signs = polished
                best_radius = r
        
        # Final high-precision polish with exact spectral radius
        if time.time() < deadline - 8:
            polished = local_search(best_signs, max_iterations=20, use_power=False)
            r = spectral_radius(polished)
            if r < best_radius - 1e-12:
                best_signs = polished
                best_radius = r
    
    # Phase 4: Try a few more structured starting points
    if time.time() < deadline - 20:
        # All +1
        all_plus = np.ones(num_edges, dtype=int)
        if is_connected_lift(all_plus):
            polished = local_search(all_plus, max_iterations=30, use_power=True)
            r = spectral_radius(polished)
            if r < best_radius:
                best_signs = polished
                best_radius = r
    
    if time.time() < deadline - 15:
        # Alternating pattern on all edges
        alt = np.ones(num_edges, dtype=int)
        for i in range(0, num_edges, 2):
            alt[i] = -1
        if is_connected_lift(alt):
            polished = local_search(alt, max_iterations=30, use_power=True)
            r = spectral_radius(polished)
            if r < best_radius:
                best_signs = polished
                best_radius = r
    
    if time.time() < deadline - 10:
        # Frustrated cycles: one -1 in each cycle, all matching +1
        signs = np.ones(num_edges, dtype=int)
        if len(bottom_cycle_edges) > 0:
            signs[bottom_cycle_edges[0]] = -1
        if len(top_cycle_edges) > 0:
            signs[top_cycle_edges[0]] = -1
        if is_connected_lift(signs):
            polished = local_search(signs, max_iterations=30, use_power=True)
            r = spectral_radius(polished)
            if r < best_radius:
                best_signs = polished
                best_radius = r
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=int)
    
    return best_signs
