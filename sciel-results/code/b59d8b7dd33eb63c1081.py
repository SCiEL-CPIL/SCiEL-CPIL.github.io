"""Improved program for packing 50 equal circles in a unit circle.
Strategy: low-dimensional ring-parameter optimization via differential evolution,
combined with a smooth soft-min objective and high-precision local polish."""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    return min(r_pair, r_bound)

def _softmin_logsumexp(arr, beta):
    """Numerically stable soft-min of an array using log-sum-exp."""
    if arr.size == 0:
        return 0.0
    max_val = np.max(arr)
    if max_val > 1e15:
        return max_val
    log_sum = np.log(np.sum(np.exp(-beta * (arr - max_val))))
    return max_val - np.log(np.sum(np.exp(-beta * (arr - max_val)))) / beta

def _smooth_objective(P, beta):
    """
    Smooth surrogate objective: soft-min of all constraints (pairwise distances
    and boundary distances). Lower is better (we minimize negative soft-min).
    Returns the negative of the soft-min radius.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_constraints = dist / 2.0  # radius from pairwise
    
    dist_origin = np.linalg.norm(P, axis=1)
    boundary_constraints = 1.0 - dist_origin  # radius from boundary
    
    # Combine all constraints
    all_constraints = np.concatenate([pair_constraints.ravel(), boundary_constraints.ravel()])
    # Filter out inf values
    all_constraints = all_constraints[np.isfinite(all_constraints)]
    
    if all_constraints.size == 0:
        return 0.0
    
    soft_min_val = _softmin_logsumexp(all_constraints, beta)
    return -soft_min_val

def _energy(P, r):
    """
    Smooth energy function: sum of squared overlaps for fixed r.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    overlaps_cc = np.maximum(2.0 * r - dist, 0.0)
    
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r - 1.0, 0.0)
    
    return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))

def _energy_grad(P, r):
    """
    Gradient of the energy function.
    """
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    V_cc = np.maximum(2.0 * r - dist, 0.0)
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 0)
    
    W = V_cc * inv_dist
    sum_W = np.sum(W, axis=1)
    term1 = (P.T @ W).T
    term2 = P * sum_W[:, np.newaxis]
    grad += term1 - term2
    
    dist_origin = np.linalg.norm(P, axis=1)
    V_cb = np.maximum(r + dist_origin - 1.0, 0.0)
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 0)
    
    grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad

def _ring_to_points(radii, offsets, ring_counts):
    """Convert ring parameters to point positions."""
    P = np.zeros((N, 2))
    k = len(ring_counts)
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += n_pts
    return P

def _flatten_ring(P, ring_counts):
    """Extract ring parameters from point positions."""
    k = len(ring_counts)
    radii = np.zeros(k)
    offsets = np.zeros(k)
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            count += 1
        else:
            pts = P[count:count+n_pts]
            radii[i] = np.mean(np.linalg.norm(pts, axis=1))
            # Estimate offset from mean angle
            angles = np.arctan2(pts[:, 1], pts[:, 0])
            # Use circular mean
            mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
            offsets[i] = mean_angle
            count += n_pts
    return np.concatenate([radii, offsets])

def _unflatten_ring(params, ring_counts):
    """Convert flattened parameters to point positions."""
    k = len(ring_counts)
    radii = params[:k]
    offsets = params[k:2*k]
    return _ring_to_points(radii, offsets, ring_counts)

def _constraint_push(P, r, iterations=20):
    """
    Push points to satisfy constraints for given r.
    Iteratively resolve overlaps and boundary violations.
    """
    P = P.copy()
    for _ in range(iterations):
        # Fix boundary violations
        norms = np.linalg.norm(P, axis=1)
        viol_b = norms + r > 1.0
        if np.any(viol_b):
            target_norm = 1.0 - r
            P[viol_b] *= target_norm / norms[viol_b, np.newaxis]
        
        # Fix pairwise violations
        diff = P[:, np.newaxis, :] - P[np.newaxis, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        violating = dist < 2.0 * r
        if not np.any(violating):
            break
        
        for i in range(N):
            for j in range(i+1, N):
                if violating[i, j]:
                    d = P[i] - P[j]
                    norm_d = np.linalg.norm(d)
                    if norm_d < 1e-10:
                        d = np.array([1.0, 0.0])
                        norm_d = 1.0
                    unit_d = d / norm_d
                    gap = (2.0 * r - norm_d) / 2.0
                    P[i] += unit_d * gap
                    P[j] -= unit_d * gap
    
    return P

def _local_refine(P_init, r_init, deadline):
    """
    Local refinement using gradient-based optimization with adaptive steps.
    """
    P_curr = P_init.copy()
    r_curr = r_init
    
    # Use scipy's L-BFGS-B for local optimization
    def objective(P_flat):
        P_test = P_flat.reshape(N, 2)
        score = _compute_score(P_test)
        norms = np.linalg.norm(P_test, axis=1)
        outside_penalty = np.sum(np.maximum(norms - 1.0, 0.0)**2)
        return -score + outside_penalty * 100
    
    P_flat = P_curr.flatten()
    
    try:
        result = minimize(objective, P_flat, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-13})
        if result.fun < -r_curr + 1e-10:
            P_opt = result.x.reshape(N, 2)
            score_opt = _compute_score(P_opt)
            if score_opt > r_curr:
                P_curr = P_opt
                r_curr = score_opt
    except:
        pass
    
    # Additional manual refinement pass
    max_iter = 150
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        diff = P_curr[:, np.newaxis, :] - P_curr[np.newaxis, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        forces = np.zeros_like(P_curr)
        
        overlap_cc = np.maximum(2.0 * r_curr - dist, 0.0)
        mask_cc = overlap_cc > 0
        if np.any(mask_cc):
            inv_dist = np.zeros_like(dist)
            np.divide(1.0, dist, out=inv_dist, where=dist > 0)
            weight = np.zeros((N, N))
            weight[mask_cc] = overlap_cc[mask_cc] * inv_dist[mask_cc]
            sum_w = np.sum(weight, axis=1)
            term1 = P_curr * sum_w[:, np.newaxis]
            term2 = (P_curr.T @ weight).T
            forces += 2.0 * (term1 - term2)
        
        dist_origin = np.linalg.norm(P_curr, axis=1)
        overlap_cb = np.maximum(dist_origin + r_curr - 1.0, 0.0)
        mask_cb = overlap_cb > 0
        if np.any(mask_cb):
            unit_vec = P_curr[mask_cb] / dist_origin[mask_cb, np.newaxis]
            forces[mask_cb] -= 2.0 * unit_vec * overlap_cb[mask_cb, np.newaxis]
        
        norm_forces = np.linalg.norm(forces)
        if norm_forces < 1e-15:
            break
        
        best_step = 0
        best_score = r_curr
        for factor in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
            step = factor * norm_forces / (N * N + 1)
            P_new = P_curr + step * forces
            new_score = _compute_score(P_new)
            if new_score > best_score:
                best_score = new_score
                best_step = step
        
        if best_step > 0:
            P_curr = P_curr + best_step * forces
            r_curr = best_score
        else:
            break
            
    return P_curr, r_curr

def _anneal_ring_params(ring_counts, deadline, n_restarts=3):
    """
    Anneal ring parameters using progressive beta annealing with differential evolution.
    """
    k = len(ring_counts)
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k
    
    best_P = None
    best_score = -np.inf
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        def neg_smooth(params):
            if time.time() > deadline - 10.0:
                return 0.0
            P = _unflatten_ring(params, ring_counts)
            # Progressive beta: start high (sharp) and decrease
            progress = (time.time() - _anneal_start) / max(deadline - 20.0 - _anneal_start, 1.0)
            beta = 50.0 * (1.0 - progress) + 5.0
            return _smooth_objective(P, beta)
        
        _anneal_start = time.time()
        
        try:
            res = differential_evolution(neg_smooth, bounds, maxiter=30, popsize=12,
                                         tol=1e-5, seed=42 + restart, polish=True,
                                         workers=1, init='latinhypercube')
            best_params = res.x
            best_P_candidate = _unflatten_ring(best_params, ring_counts)
            score_candidate = _compute_score(best_P_candidate)
            
            if score_candidate > best_score:
                best_score = score_candidate
                best_P = best_P_candidate.copy()
                
        except Exception:
            pass
    
    return best_P, best_score

def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98

def _high_precision_polish(P, r, deadline):
    """
    High-precision polish: iterate between radius increase and constraint projection.
    """
    P_curr = P.copy()
    r_curr = r
    
    for it in range(50):
        if time.time() > deadline - 5.0:
            break
        
        # Try to increase radius slightly
        r_try = r_curr * (1.0 + 1e-5)
        
        # Project points to satisfy constraints at r_try
        P_projected = _constraint_push(P_curr, r_try, iterations=30)
        
        # Check if projection works
        score_projected = _compute_score(P_projected)
        if score_projected >= r_try - 1e-12:
            P_curr = P_projected
            r_curr = r_try
        else:
            # Try smaller increment
            r_try2 = r_curr * (1.0 + 1e-7)
            P_projected2 = _constraint_push(P_curr, r_try2, iterations=30)
            score_projected2 = _compute_score(P_projected2)
            if score_projected2 >= r_try2 - 1e-12:
                P_curr = P_projected2
                r_curr = r_try2
            else:
                break
    
    # Final local refinement at the improved radius
    P_final, r_final = _local_refine(P_curr, r_curr, deadline)
    return P_final, r_final

def _binary_search_radius(P, r_lo, r_hi, deadline, max_iter=80):
    """
    Binary search on radius with constraint projection.
    """
    best_P = P.copy()
    best_r = r_lo
    
    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        
        r_mid = (r_lo + r_hi) / 2.0
        P_test = P.copy()
        
        # Check boundary constraint
        norms = np.linalg.norm(P_test, axis=1)
        boundary_ok = np.all(norms + r_mid <= 1.0 + 1e-12)
        
        # Check pairwise constraint
        diff = P_test[:, np.newaxis, :] - P_test[np.newaxis, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pairwise_ok = np.all(dist >= 2.0 * r_mid - 1e-12)
        
        if boundary_ok and pairwise_ok:
            r_lo = r_mid
            best_P = P_test.copy()
            best_r = r_mid
        else:
            r_hi = r_mid
            
            # Try to fix by moving violating points
            P_fixed = P_test.copy()
            
            # Fix boundary violations
            norms = np.linalg.norm(P_fixed, axis=1)
            viol_b = norms + r_mid > 1.0
            if np.any(viol_b):
                target_norm = 1.0 - r_mid
                P_fixed[viol_b] *= target_norm / norms[viol_b, np.newaxis]
            
            # Fix pairwise violations iteratively
            for _ in range(15):
                diff = P_fixed[:, np.newaxis, :] - P_fixed[np.newaxis, :]
                dist = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dist, np.inf)
                
                violating = dist < 2.0 * r_mid
                if not np.any(violating):
                    break
                
                for i in range(N):
                    for j in range(i+1, N):
                        if violating[i, j]:
                            d = P_fixed[i] - P_fixed[j]
                            norm_d = np.linalg.norm(d)
                            if norm_d < 1e-10:
                                d = np.array([1.0, 0.0])
                                norm_d = 1.0
                            unit_d = d / norm_d
                            gap = (2.0 * r_mid - norm_d) / 2.0
                            P_fixed[i] += unit_d * gap
                            P_fixed[j] -= unit_d * gap
            
            # Re-check if fixed configuration works
            norms = np.linalg.norm(P_fixed, axis=1)
            boundary_ok2 = np.all(norms + r_mid <= 1.0 + 1e-12)
            if boundary_ok2:
                diff2 = P_fixed[:, np.newaxis, :] - P_fixed[np.newaxis, :]
                dist2 = np.sqrt(np.sum(diff2**2, axis=2))
                np.fill_diagonal(dist2, np.inf)
                if np.all(dist2 >= 2.0 * r_mid - 1e-12):
                    r_lo = r_mid
                    best_P = P_fixed.copy()
                    best_r = r_mid
                    P = P_fixed
    
    return best_P, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    np.random.seed(42)
    
    # Try multiple ring structures
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 5, 10, 15, 19],
        [1, 7, 14, 21, 7],
        [1, 6, 11, 17, 15],
        [1, 8, 16, 24, 1],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 18, 13],  # duplicate for different seed
        [1, 7, 14, 28, 0],
    ]
    
    best_P = None
    best_score = -np.inf
    
    # Phase 1: Anneal ring parameters for each structure
    for idx, ring_counts in enumerate(ring_structures):
        if time.time() > deadline - 40.0:
            break
        
        P_ring, score_ring = _anneal_ring_params(ring_counts, deadline, n_restarts=2)
        
        if P_ring is not None and score_ring > best_score:
            # Local refine from ring structure
            P_ref, score_ref = _local_refine(P_ring, score_ring, deadline)
            if score_ref > best_score:
                best_score = score_ref
                best_P = P_ref
        elif P_ring is None:
            # Fallback to hexagonal lattice
            P_hex = _get_hex_initial_points()
            score_hex = _compute_score(P_hex)
            P_hex_ref, score_hex_ref = _local_refine(P_hex, score_hex, deadline)
            if score_hex_ref > best_score:
                best_score = score_hex_ref
                best_P = P_hex_ref
    
    # Also try hexagonal lattice as a baseline
    P_hex = _get_hex_initial_points()
    score_hex = _compute_score(P_hex)
    P_hex_ref, score_hex_ref = _local_refine(P_hex, score_hex, deadline)
    if score_hex_ref > best_score:
        best_score = score_hex_ref
        best_P = P_hex_ref
    
    # Phase 2: High-precision polish
    if time.time() < deadline - 15.0:
        P_polish, r_polish = _high_precision_polish(best_P, best_score, deadline)
        if r_polish > best_score:
            best_P = P_polish
            best_score = r_polish
    
    # Phase 3: Binary search on radius for final precision
    if time.time() < deadline - 8.0:
        r_lo = best_score * 0.995
        r_hi = best_score * 1.005
        P_final, r_final = _binary_search_radius(best_P, r_lo, r_hi, deadline, max_iter=80)
        if r_final > best_score:
            best_P = P_final
            best_score = r_final
    
    # Ensure all points are inside the unit circle
    norms = np.linalg.norm(best_P, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_P[mask] *= (1.0 - 1e-9) / norms[mask, np.newaxis]
    
    # Final safety check
    final_score = _compute_score(best_P)
    if not np.isfinite(final_score) or final_score <= 0:
        best_P = P_hex
        final_score = score_hex
    
    return best_P
# EVOLVE-BLOCK-END
