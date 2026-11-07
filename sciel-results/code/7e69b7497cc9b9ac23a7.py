import time
import numpy as np
from scipy.optimize import minimize as scipy_minimize

def sign_edges(graph, budget):
    deadline = time.time() + 82.0  # Safety margin for 90s limit
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Build index mapping for fast signed adjacency construction
    edge_pos = {}
    for idx, (u, v) in enumerate(edges):
        edge_pos[(u, v)] = idx
        edge_pos[(v, u)] = idx

    def build_signed(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            S[u, v] = s
            S[v, u] = s
        return S
    
    def objective(signs):
        S = build_signed(signs)
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))
    
    def objective_fast(signs, S_cache=None):
        """Compute spectral radius, optionally reusing a cached matrix."""
        if S_cache is None:
            S = build_signed(signs)
        else:
            S = S_cache
        ev = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(ev)))
    
    def flip_and_evaluate(signs, e, S):
        """Flip edge e in signs, update S in-place, compute radius, then undo."""
        u, v = edges[e]
        old_s = signs[e]
        new_s = -old_s
        signs[e] = new_s
        S[u, v] = new_s
        S[v, u] = new_s
        ev = np.linalg.eigvalsh(S)
        radius = float(np.max(np.abs(ev)))
        # Undo
        signs[e] = old_s
        S[u, v] = old_s
        S[v, u] = old_s
        return radius
    
    def smooth_spectral_radius(phases, beta=30.0):
        """
        Smooth surrogate: minimize max |lambda| of signed adjacency where
        edge signs are cos(phases). This is differentiable and allows gradient
        descent to find good sign patterns.
        """
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = np.cos(phases[idx])
            S[u, v] = s
            S[v, u] = s
        ev = np.linalg.eigvalsh(S)
        # Use soft-max of |eigenvalues| for smoothness
        abs_ev = np.abs(ev)
        # log-sum-exp trick for stability
        max_abs = np.max(abs_ev)
        soft_max = max_abs + np.log(np.sum(np.exp(beta * (abs_ev - max_abs)))) / beta
        return float(soft_max)
    
    def smooth_gradient(phases, beta=30.0):
        """Numerical gradient of smooth spectral radius."""
        eps = 1e-5
        grad = np.zeros_like(phases)
        for i in range(len(phases)):
            phases_p = phases.copy()
            phases_p[i] += eps
            val_p = smooth_spectral_radius(phases_p, beta)
            phases_p[i] -= 2 * eps
            val_m = smooth_spectral_radius(phases_p, beta)
            grad[i] = (val_p - val_m) / (2 * eps)
        return grad
    
    # Identify edge types: Top Cycle, Bottom Cycle, Matching
    # Vertices 0..m-1 are top, m..2m-1 are bottom
    m = n // 2
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
            
    def get_pattern(pattern_id):
        signs = np.ones(num_edges, dtype=np.int8)
        if pattern_id == 0:
            # All +1
            pass
        elif pattern_id == 1:
            # Alternating cycles, +1 matching
            for i, idx in enumerate(top_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
            for i, idx in enumerate(bottom_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        elif pattern_id == 2:
            # +1 cycles, Alternating matching
            for i, idx in enumerate(match_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        elif pattern_id == 3:
            # Alternating cycles (shifted), Alternating matching
            for i, idx in enumerate(top_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
            for i, idx in enumerate(bottom_edges):
                signs[idx] = -1 if i % 2 == 0 else 1
            for i, idx in enumerate(match_edges):
                signs[idx] = 1 if i % 2 == 0 else -1
        return signs

    best_signs = None
    best_obj = float('inf')
    
    def try_improve(signs):
        nonlocal best_signs, best_obj
        cur_signs = signs.copy()
        S_cur = build_signed(cur_signs)
        cur_obj = objective(cur_signs)
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_signs = cur_signs.copy()
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            best_flip = -1
            best_flip_obj = cur_obj
            
            # Greedy single flip using fast evaluation
            for e in range(num_edges):
                o = flip_and_evaluate(cur_signs, e, S_cur)
                if o < best_flip_obj - 1e-12:
                    best_flip_obj = o
                    best_flip = e
            
            if best_flip >= 0:
                # Apply the best flip
                e = best_flip
                u, v = edges[e]
                old_s = cur_signs[e]
                cur_signs[e] = -old_s
                S_cur[u, v] = -old_s
                S_cur[v, u] = -old_s
                cur_obj = best_flip_obj
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_signs = cur_signs.copy()
                improved = True
            else:
                # Try all pair flips to escape local minima
                found = False
                for e1 in range(num_edges):
                    if time.time() > deadline - 2.0:
                        break
                    for e2 in range(e1 + 1, num_edges):
                        if time.time() > deadline - 2.0:
                            break
                        u1, v1 = edges[e1]
                        u2, v2 = edges[e2]
                        s1_old = cur_signs[e1]
                        s2_old = cur_signs[e2]
                        s1_new = -s1_old
                        s2_new = -s2_old
                        # Apply both flips
                        cur_signs[e1] = s1_new
                        cur_signs[e2] = s2_new
                        S_cur[u1, v1] = s1_new
                        S_cur[v1, u1] = s1_new
                        S_cur[u2, v2] = s2_new
                        S_cur[v2, u2] = s2_new
                        o = objective_fast(cur_signs, S_cur)
                        # Undo
                        cur_signs[e1] = s1_old
                        cur_signs[e2] = s2_old
                        S_cur[u1, v1] = s1_old
                        S_cur[v1, u1] = s1_old
                        S_cur[u2, v2] = s2_old
                        S_cur[v2, u2] = s2_old
                        if o < cur_obj - 1e-12:
                            # Apply the improving pair flip
                            cur_signs[e1] = s1_new
                            cur_signs[e2] = s2_new
                            S_cur[u1, v1] = s1_new
                            S_cur[v1, u1] = s1_new
                            S_cur[u2, v2] = s2_new
                            S_cur[v2, u2] = s2_new
                            cur_obj = o
                            if cur_obj < best_obj:
                                best_obj = cur_obj
                                best_signs = cur_signs.copy()
                            found = True
                            improved = True
                            break
                    if improved:
                        break
                if not found:
                    break

    # Start from the best symmetric pattern
    for p in range(4):
        if time.time() > deadline - 5.0:
            break
        pattern = get_pattern(p)
        try_improve(pattern)
        
    # If time remains, do random restarts
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(3)
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            try_improve(signs)
    
    # Continuous phase-space optimization to escape discrete local minima
    if best_signs is not None and time.time() < deadline - 8.0:
        # Multiple initializations for phase-space optimization
        initializations = []
        
        # Init 1: From current best signs
        init1 = np.where(best_signs == 1, 0.0, np.pi)
        initializations.append(init1)
        
        # Init 2: Small perturbation around best
        rng_phase = np.random.default_rng(42)
        init2 = init1 + rng_phase.normal(0, 0.3, size=num_edges)
        initializations.append(init2)
        
        # Init 3: Random phases
        init3 = rng_phase.uniform(0, 2 * np.pi, size=num_edges)
        initializations.append(init3)
        
        for init_phases in initializations:
            if time.time() > deadline - 4.0:
                break
            try:
                result = scipy_minimize(
                    lambda p: smooth_spectral_radius(p, beta=30.0),
                    init_phases,
                    jac=lambda p: smooth_gradient(p, beta=30.0),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                # Round back to signs
                new_signs = np.where(result.x < np.pi / 2, 1, -1).astype(np.int8)
                new_signs[new_signs > 0] = 1
                new_signs[new_signs < 0] = -1
                new_obj = objective(new_signs)
                if new_obj < best_obj - 1e-12:
                    best_obj = new_obj
                    best_signs = new_signs.copy()
                    # Polish with discrete local search
                    try_improve(best_signs.copy())
            except Exception:
                pass

    # Final polish: exhaustive single and pair flip check on best solution
    if best_signs is not None and time.time() < deadline - 1.0:
        S_best = build_signed(best_signs)
        obj_best = objective(best_signs)
        # Single flips - apply improvements
        for e in range(num_edges):
            if time.time() > deadline - 0.5:
                break
            o = flip_and_evaluate(best_signs, e, S_best)
            if o < obj_best - 1e-12:
                # Apply the flip
                u, v = edges[e]
                old_s = best_signs[e]
                best_signs[e] = -old_s
                S_best[u, v] = -old_s
                S_best[v, u] = -old_s
                obj_best = o
        # Pair flips - apply improvements
        for e1 in range(num_edges):
            if time.time() > deadline - 0.5:
                break
            for e2 in range(e1 + 1, num_edges):
                if time.time() > deadline - 0.5:
                    break
                u1, v1 = edges[e1]
                u2, v2 = edges[e2]
                s1_old = best_signs[e1]
                s2_old = best_signs[e2]
                s1_new = -s1_old
                s2_new = -s2_old
                best_signs[e1] = s1_new
                best_signs[e2] = s2_new
                S_best[u1, v1] = s1_new
                S_best[v1, u1] = s1_new
                S_best[u2, v2] = s2_new
                S_best[v2, u2] = s2_new
                o = objective_fast(best_signs, S_best)
                if o < obj_best - 1e-12:
                    obj_best = o
                else:
                    best_signs[e1] = s1_old
                    best_signs[e2] = s2_old
                    S_best[u1, v1] = s1_old
                    S_best[v1, u1] = s1_old
                    S_best[u2, v2] = s2_old
                    S_best[v2, u2] = s2_old

    if best_signs is None:
        best_signs = np.array([1] * num_edges, dtype=np.int8)
    
    return best_signs.astype(np.int8)
