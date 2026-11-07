import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Classify edges: top cycle, bottom cycle, matching
    top_edges = []
    bottom_edges = []
    match_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bottom_edges.append(idx)
        else:
            match_edges.append(idx)
    
    def build_signed(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            S[u, v] = s
            S[v, u] = s
        return S
    
    def spectral_radius(signs):
        S = build_signed(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))
    
    # Analytical approach: for a prism graph, if we know the cycle frustrations
    # and matching signs, we can compute eigenvalues analytically.
    # 
    # Key insight: The spectrum depends on:
    # - Product of signs around top cycle (frustration_top)
    # - Product of signs around bottom cycle (frustration_bottom)  
    # - The matching edge signs
    #
    # For the analytical formula, we need to consider the structure more carefully.
    # Let's use a parameterization:
    # - top_cycle_signs: signs on top cycle edges (product = f_top)
    # - bottom_cycle_signs: signs on bottom cycle edges (product = f_bottom)
    # - match_signs: signs on matching edges
    #
    # The eigenvalues of the signed adjacency matrix can be found by considering
    # the Fourier modes on each cycle.
    
    # Let's try a different approach: exploit that the graph has a specific structure
    # and do smart local search with analytical evaluation where possible.
    
    def evaluate_fast(signs):
        """Fast spectral radius computation."""
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            S[u, v] = s
            S[v, u] = s
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))
    
    # Strategy: 
    # 1. Start with structured patterns based on cycle frustrations
    # 2. Use local search (single and pair flips)
    # 3. Use continuous relaxation to escape local minima
    
    best_signs = None
    best_obj = float('inf')
    
    def update_best(signs, obj):
        nonlocal best_signs, best_obj
        if obj < best_obj - 1e-14:
            best_obj = obj
            best_signs = signs.copy()
    
    # Generate structured initial patterns
    # The key variables are:
    # - frustration of top cycle (product of signs)
    # - frustration of bottom cycle
    # - pattern of matching signs
    
    def make_pattern(top_frustration, bottom_frustration, match_pattern):
        """
        Create a sign pattern given:
        - top_frustration: +1 or -1 (product of top cycle signs)
        - bottom_frustration: +1 or -1 (product of bottom cycle signs)
        - match_pattern: 'all_plus', 'all_minus', 'alternating', 'alternating_shift'
        """
        signs = np.ones(num_edges, dtype=np.int8)
        
        # Top cycle: make product = top_frustration
        # Use all +1 except one edge if frustration = -1
        if top_frustration == -1:
            # Flip one edge in top cycle
            if top_edges:
                signs[top_edges[0]] = -1
        # Bottom cycle
        if bottom_frustration == -1:
            if bottom_edges:
                signs[bottom_edges[0]] = -1
        
        # Matching edges
        if match_pattern == 'all_plus':
            pass
        elif match_pattern == 'all_minus':
            for idx in match_edges:
                signs[idx] = -1
        elif match_pattern == 'alternating':
            for i, idx in enumerate(match_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        elif match_pattern == 'alternating_shift':
            for i, idx in enumerate(match_edges):
                signs[idx] = -1 if i % 2 == 0 else 1
        
        return signs
    
    # Try all combinations of frustrations and matching patterns
    patterns = []
    for tf in [1, -1]:
        for bf in [1, -1]:
            for mp in ['all_plus', 'all_minus', 'alternating', 'alternating_shift']:
                patterns.append(make_pattern(tf, bf, mp))
    
    # Also try some random patterns
    rng = np.random.default_rng(3)
    for _ in range(10):
        patterns.append(rng.choice([-1, 1], size=num_edges).astype(np.int8))
    
    # Local search function
    def local_search(signs, max_time=5.0):
        """Perform local search with single and pair flips."""
        cur_signs = signs.copy()
        cur_obj = evaluate_fast(cur_signs)
        update_best(cur_signs, cur_obj)
        
        start_time = time.time()
        improved = True
        
        while improved and time.time() - start_time < max_time:
            improved = False
            best_flip_obj = cur_obj
            best_flip = -1
            best_pair = (-1, -1)
            best_pair_obj = cur_obj
            
            # Try single flips
            for e in range(num_edges):
                old_s = cur_signs[e]
                cur_signs[e] = -old_s
                obj = evaluate_fast(cur_signs)
                cur_signs[e] = old_s
                
                if obj < best_flip_obj - 1e-14:
                    best_flip_obj = obj
                    best_flip = e
            
            # Try pair flips
            for e1 in range(num_edges):
                if time.time() - start_time > max_time:
                    break
                for e2 in range(e1 + 1, num_edges):
                    if time.time() - start_time > max_time:
                        break
                    s1_old = cur_signs[e1]
                    s2_old = cur_signs[e2]
                    cur_signs[e1] = -s1_old
                    cur_signs[e2] = -s2_old
                    obj = evaluate_fast(cur_signs)
                    cur_signs[e1] = s1_old
                    cur_signs[e2] = s2_old
                    
                    if obj < best_pair_obj - 1e-14:
                        best_pair_obj = obj
                        best_pair = (e1, e2)
            
            # Apply best improvement
            if best_flip >= 0:
                cur_signs[best_flip] = -cur_signs[best_flip]
                cur_obj = best_flip_obj
                update_best(cur_signs, cur_obj)
                improved = True
            elif best_pair[0] >= 0:
                cur_signs[best_pair[0]] = -cur_signs[best_pair[0]]
                cur_signs[best_pair[1]] = -cur_signs[best_pair[1]]
                cur_obj = best_pair_obj
                update_best(cur_signs, cur_obj)
                improved = True
        
        return cur_signs, cur_obj
    
    # Run local search on each pattern
    for pattern in patterns:
        if time.time() > deadline - 10.0:
            break
        local_search(pattern, max_time=2.0)
    
    # Continuous relaxation to escape local minima
    if best_signs is not None and time.time() < deadline - 15.0:
        def smooth_obj(phases, beta=50.0):
            """Smooth surrogate for spectral radius."""
            S = np.zeros((n, n), dtype=np.float64)
            for idx, (u, v) in enumerate(edges):
                s = np.cos(phases[idx])
                S[u, v] = s
                S[v, u] = s
            ev = np.linalg.eigvalsh(S)
            abs_ev = np.abs(ev)
            max_abs = np.max(abs_ev)
            soft_max = max_abs + np.log(np.sum(np.exp(beta * (abs_ev - max_abs)))) / beta
            return float(soft_max)
        
        def smooth_grad(phases, beta=50.0, eps=1e-5):
            grad = np.zeros_like(phases)
            for i in range(len(phases)):
                phases_p = phases.copy()
                phases_p[i] += eps
                val_p = smooth_obj(phases_p, beta)
                phases_p[i] -= 2 * eps
                val_m = smooth_obj(phases_p, beta)
                grad[i] = (val_p - val_m) / (2 * eps)
            return grad
        
        # Multiple initializations
        inits = []
        # From best signs
        init1 = np.where(best_signs == 1, 0.0, np.pi)
        inits.append(init1)
        # Perturbed
        rng2 = np.random.default_rng(42)
        inits.append(init1 + rng2.normal(0, 0.5, size=num_edges))
        # Random
        inits.append(rng2.uniform(0, 2 * np.pi, size=num_edges))
        
        for init in inits:
            if time.time() > deadline - 5.0:
                break
            try:
                result = scipy_minimize(
                    lambda p: smooth_obj(p, beta=50.0),
                    init,
                    jac=lambda p: smooth_grad(p, beta=50.0),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                new_signs = np.where(result.x < np.pi / 2, 1, -1).astype(np.int8)
                new_obj = evaluate_fast(new_signs)
                if new_obj < best_obj - 1e-14:
                    best_obj = new_obj
                    best_signs = new_signs.copy()
                    # Polish with local search
                    best_signs, best_obj = local_search(best_signs, max_time=2.0)
            except Exception:
                pass
    
    # Final exhaustive check on best solution
    if best_signs is not None and time.time() < deadline - 2.0:
        # Multiple rounds of single flips
        for _ in range(3):
            if time.time() > deadline - 1.0:
                break
            improved = True
            while improved:
                improved = False
                for e in range(num_edges):
                    if time.time() > deadline - 0.5:
                        break
                    old_s = best_signs[e]
                    best_signs[e] = -old_s
                    obj = evaluate_fast(best_signs)
                    if obj < best_obj - 1e-14:
                        best_obj = obj
                        improved = True
                    else:
                        best_signs[e] = old_s
    
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
    
    return best_signs.astype(np.int8)
