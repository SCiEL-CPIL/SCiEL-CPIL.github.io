"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _enforce_sum(h):
    """Enforce sum(h) = N/2 and 0<=h<=1 with iterative projection."""
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-16:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    diff = N/2.0 - s
    if abs(diff) > 1e-16:
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _knots_to_h(knot_positions, knot_values):
    """
    Interpolate piecewise-linear from knots to N sample points.
    knot_positions: array of K positions in [0, N-1] (monotonic)
    knot_values: array of K values
    Returns: array of N values
    """
    K = len(knot_positions)
    if K < 2:
        return np.full(N, knot_values[0] if K == 1 else 0.5)
    
    h = np.empty(N)
    # For each sample point, find which segment it falls in
    for i in range(N):
        if i <= knot_positions[0]:
            h[i] = knot_values[0]
        elif i >= knot_positions[-1]:
            h[i] = knot_values[-1]
        else:
            # Find segment
            seg = np.searchsorted(knot_positions, i, side='right') - 1
            seg = max(0, min(seg, K-2))
            t = (i - knot_positions[seg]) / (knot_positions[seg+1] - knot_positions[seg] + 1e-12)
            t = max(0.0, min(1.0, t))
            h[i] = knot_values[seg] * (1-t) + knot_values[seg+1] * t
    return h

def _h_to_knots(h, n_knots):
    """
    Extract knot positions and values from h by identifying transition points.
    Returns (knot_positions, knot_values)
    """
    # Find where h changes significantly
    diffs = np.abs(np.diff(h))
    threshold = 0.05
    change_points = np.where(diffs > threshold)[0]
    
    if len(change_points) == 0:
        # No transitions, use uniform knots
        positions = np.linspace(0, N-1, n_knots)
        values = np.interp(positions, np.arange(N), h)
        return positions, values
    
    # Cluster change points
    clusters = []
    current_cluster = [change_points[0]]
    for cp in change_points[1:]:
        if cp - current_cluster[-1] <= 3:
            current_cluster.append(cp)
        else:
            clusters.append(current_cluster)
            current_cluster = [cp]
    clusters.append(current_cluster)
    
    # Build knot positions from clusters
    knot_pos = [0.0]
    knot_val = [h[0]]
    for cluster in clusters:
        mid = int(np.mean(cluster))
        # Add transition: value before and after
        if mid not in knot_pos:
            knot_pos.append(float(mid))
            knot_val.append(h[mid])
        if mid+1 < N and (mid+1) not in knot_pos:
            knot_pos.append(float(mid+1))
            knot_val.append(h[mid+1])
    if (N-1) not in knot_pos:
        knot_pos.append(float(N-1))
        knot_val.append(h[N-1])
    
    knot_pos = np.array(knot_pos)
    knot_val = np.array(knot_val)
    
    # If we have more knots than desired, subsample
    if len(knot_pos) > n_knots:
        idx = np.linspace(0, len(knot_pos)-1, n_knots, dtype=int)
        knot_pos = knot_pos[idx]
        knot_val = knot_val[idx]
    # If fewer, pad
    while len(knot_pos) < n_knots:
        knot_pos = np.append(knot_pos, N-1)
        knot_val = np.append(knot_val, knot_val[-1])
    
    return knot_pos, knot_val

def _knot_objective(params, n_knots):
    """
    Objective for knot optimization.
    params: [pos_1, ..., pos_{K-1}, val_0, ..., val_{K-1}]
    pos_0 = 0, pos_{K-1} = N-1 are fixed.
    """
    K = n_knots
    pos_internal = params[:K-2]  # K-2 internal positions
    values = params[K-2:]        # K values
    
    # Build full position array
    positions = np.concatenate([[0.0], pos_internal, [float(N-1)]])
    
    # Check monotonicity
    if np.any(np.diff(positions) < 0.5):
        return 1.0 + 10.0 * np.sum(np.maximum(0, 0.5 - np.diff(positions)))
    
    # Clip values to [0,1]
    values = np.clip(values, 0.0, 1.0)
    
    # Interpolate
    h = _knots_to_h(positions, values)
    
    # Enforce sum constraint via scaling
    s = np.sum(h)
    if s > 1e-10:
        scale = (N/2.0) / s
        h_scaled = h * scale
        # If scaling pushes out of bounds, penalize
        if np.any(h_scaled > 1.0) or np.any(h_scaled < 0.0):
            penalty = 100.0 * (np.sum(np.maximum(0, h_scaled - 1.0)) + np.sum(np.maximum(0, -h_scaled)))
            h_scaled = np.clip(h_scaled, 0.0, 1.0)
        else:
            penalty = 0.0
    else:
        h_scaled = np.full(N, 0.5)
        penalty = 0.0
    
    # Compute score
    sc = _score(h_scaled)
    return sc + penalty

def _knots_to_h_vectorized(knot_positions, knot_values):
    """Vectorized version of knots to h."""
    K = len(knot_positions)
    if K < 2:
        return np.full(N, knot_values[0] if K == 1 else 0.5)
    
    x = np.arange(N, dtype=float)
    h = np.interp(x, knot_positions, knot_values)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial 2-block solution
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    h0 = _enforce_sum(h0)
    current_score = _score(h0)
    h_final = h0.copy()
    
    def _try_update(h_candidate):
        nonlocal h_final, current_score
        h_c = _enforce_sum(h_candidate)
        if np.all(h_c >= -1e-12) and np.all(h_c <= 1.0 + 1e-12):
            h_c = np.clip(h_c, 0.0, 1.0)
            s = np.sum(h_c)
            if abs(s - N/2.0) < 1e-8:
                sc = _score(h_c)
                if sc < current_score - 1e-14:
                    h_final = h_c.copy()
                    current_score = sc
                    return True
        return False
    
    # Phase 1: Nelder-Mead on knot representation
    try:
        n_knots = 8
        # Initial knots from 2-block: transition at position half
        knot_pos_init = np.array([0.0, half-2, half-1, half, half+1, half+2, float(N-1)], dtype=float)
        # Pad or trim to n_knots-1 internal + 2 endpoints
        # We need K-2 internal positions
        if len(knot_pos_init) < n_knots:
            extra = np.linspace(0, N-1, n_knots - len(knot_pos_init))
            knot_pos_init = np.sort(np.concatenate([knot_pos_init, extra]))
        elif len(knot_pos_init) > n_knots:
            idx = np.linspace(0, len(knot_pos_init)-1, n_knots, dtype=int)
            knot_pos_init = knot_pos_init[idx]
        
        # Ensure first is 0 and last is N-1
        knot_pos_init[0] = 0.0
        knot_pos_init[-1] = float(N-1)
        
        # Values at knots
        knot_val_init = np.interp(knot_pos_init, np.arange(N), h0)
        
        # Parameter vector: internal positions + all values
        internal_pos = knot_pos_init[1:-1]  # K-2 internal
        params_init = np.concatenate([internal_pos, knot_val_init])
        
        # Nelder-Mead
        def obj_nm(p):
            return _knot_objective(p, n_knots)
        
        result_nm = minimize(
            obj_nm,
            params_init.copy(),
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        
        if result_nm.fun < current_score:
            # Extract h from optimized knots
            p_opt = result_nm.x
            K = n_knots
            pos_int = p_opt[:K-2]
            vals = p_opt[K-2:]
            positions = np.concatenate([[0.0], pos_int, [float(N-1)]])
            values = np.clip(vals, 0.0, 1.0)
            h_opt = _knots_to_h_vectorized(positions, values)
            _try_update(h_opt)
            
    except Exception:
        pass
    
    # Phase 2: SLSQP on full h for local refinement
    try:
        if time.time() < deadline - 30.0:
            def objective_slsqp(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))
            
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N
            
            result_slsqp = minimize(
                objective_slsqp,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 3000, 'ftol': 1e-15, 'disp': False}
            )
            
            if result_slsqp.fun < current_score:
                _try_update(result_slsqp.x)
                
    except Exception:
        pass
    
    # Phase 3: More knots with Nelder-Mead from current best
    try:
        if time.time() < deadline - 30.0:
            n_knots2 = 12
            # Extract knots from current best
            knot_pos2, knot_val2 = _h_to_knots(h_final, n_knots2)
            
            if len(knot_pos2) != n_knots2:
                idx = np.linspace(0, N-1, n_knots2, dtype=int)
                knot_pos2 = np.array(idx, dtype=float)
                knot_val2 = np.interp(knot_pos2, np.arange(N), h_final)
            
            knot_pos2[0] = 0.0
            knot_pos2[-1] = float(N-1)
            
            internal_pos2 = knot_pos2[1:-1]
            params2_init = np.concatenate([internal_pos2, knot_val2])
            
            def obj_nm2(p):
                return _knot_objective(p, n_knots2)
            
            result_nm2 = minimize(
                obj_nm2,
                params2_init.copy(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if result_nm2.fun < current_score:
                p_opt = result_nm2.x
                K = n_knots2
                pos_int = p_opt[:K-2]
                vals = p_opt[K-2:]
                positions = np.concatenate([[0.0], pos_int, [float(N-1)]])
                values = np.clip(vals, 0.0, 1.0)
                h_opt = _knots_to_h_vectorized(positions, values)
                _try_update(h_opt)
                
    except Exception:
        pass
    
    # Phase 4: Multi-start Nelder-Mead with different knot counts
    try:
        if time.time() < deadline - 20.0:
            np.random.seed(123)
            for n_k in [6, 10, 14]:
                if time.time() > deadline - 10.0:
                    break
                
                # Random initial knots
                n_internal = n_k - 2
                pos_rand = np.sort(np.random.uniform(1, N-2, n_internal))
                pos_full = np.concatenate([[0.0], pos_rand, [float(N-1)]])
                val_rand = np.random.uniform(0.2, 0.8, n_k)
                
                params_rand = np.concatenate([pos_full[1:-1], val_rand])
                
                def obj_rand(p):
                    return _knot_objective(p, n_k)
                
                result_rand = minimize(
                    obj_rand,
                    params_rand.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                
                if result_rand.fun < current_score:
                    p_opt = result_rand.x
                    K = n_k
                    pos_int = p_opt[:K-2]
                    vals = p_opt[K-2:]
                    positions = np.concatenate([[0.0], pos_int, [float(N-1)]])
                    values = np.clip(vals, 0.0, 1.0)
                    h_opt = _knots_to_h_vectorized(positions, values)
                    _try_update(h_opt)
                    
    except Exception:
        pass
    
    # Phase 5: Final SLSQP polish
    try:
        if time.time() < deadline - 5.0:
            def objective_polish(h):
                shifted = np.zeros(N)
                shifted[:-1] = h[1:]
                overlap = h * (1.0 - shifted)
                return float(np.max(overlap))
            
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
            bounds = [(0.0, 1.0)] * N
            
            result_final = minimize(
                objective_polish,
                h_final.copy(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 5000, 'ftol': 1e-16, 'disp': False}
            )
            
            if result_final.fun < current_score:
                _try_update(result_final.x)
                
    except Exception:
        pass
    
    h_final = _enforce_sum(h_final)
    return h_final
