"""Improved program for packing 50 equal circles in a unit circle."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution, brentq

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

def _energy(P, r, softness=10.0):
    """Energy: smooth approximation using soft-min for better gradient flow."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Use soft-min approximation: -softness * log(sum(exp(-softness * x)))
    # For circle-circle: want to penalize when dist < 2r
    # Smooth penalty: -softness * log(sum(exp(-softness * (dist - 2r)))) for dist < 2r
    # Simpler: use smooth max(0, 2r - dist) approximation
    
    # Circle-circle: smooth penalty for violations
    # Use tanh-based smooth max: max(0, x) ≈ 0.5*(x + sqrt(x^2 + eps))
    eps = 1e-6
    cc_violation = 2.0 * r - np.sqrt(dist_sq + eps)
    cc_smooth = 0.5 * (cc_violation + np.sqrt(cc_violation**2 + eps))
    cc_smooth = np.where(cc_violation > -0.01, cc_smooth, 0.0)
    
    # Circle-boundary
    dist_origin = np.linalg.norm(P, axis=1)
    cb_violation = dist_origin + r - 1.0
    cb_smooth = 0.5 * (cb_violation + np.sqrt(cb_violation**2 + eps))
    cb_smooth = np.where(cb_violation > -0.01, cb_smooth, 0.0)
    
    return np.sum(cc_smooth**2) + np.sum(cb_smooth**2)

def _get_ring_initialization():
    """Generate a ring-structured initialization."""
    points = []
    points.append([0.0, 0.0])
    
    r1 = 0.3
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    r2 = 0.55
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi/12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    r3 = 0.78
    for i in range(18):
        angle = 2 * np.pi * i / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    r4 = 0.92
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi/13
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points)

def _ring_params_to_points(params):
    """
    Convert parametric ring structure to point positions.
    params: [r1, r2, r3, r4, offset1, offset2, offset3, offset4]
    Ring structure: 1 center + 6 + 12 + 18 + 13 = 50
    """
    r1, r2, r3, r4 = params[0], params[1], params[2], params[3]
    off1, off2, off3, off4 = params[4], params[5], params[6], params[7]
    
    points = [[0.0, 0.0]]
    
    for i in range(6):
        angle = 2 * np.pi * i / 6 + off1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    for i in range(12):
        angle = 2 * np.pi * i / 12 + off2
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    for i in range(18):
        angle = 2 * np.pi * i / 18 + off3
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    for i in range(13):
        angle = 2 * np.pi * i / 13 + off4
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points)

def _optimize_ring_params(deadline):
    """
    Use differential evolution to optimize the parametric ring structure.
    Returns the best point configuration found.
    """
    # Parameter bounds: radii in [0, 1], offsets in [0, 2*pi]
    bounds = [
        (0.05, 0.5),   # r1
        (0.3, 0.7),    # r2
        (0.5, 0.9),    # r3
        (0.7, 0.99),   # r4
        (0, 2*np.pi),  # offset1
        (0, 2*np.pi),  # offset2
        (0, 2*np.pi),  # offset3
        (0, 2*np.pi),  # offset4
    ]
    
    def objective(params):
        if time.time() > deadline - 5.0:
            return 0.0
        P = _ring_params_to_points(params)
        return _compute_score(P)
    
    try:
        result = differential_evolution(
            objective, bounds,
            seed=42, maxiter=100, popsize=15,
            tol=1e-8, mutation=(0.3, 1.0), recombination=0.7,
            polish=True, updating='immediate'
        )
        best_params = result.x
        best_P = _ring_params_to_points(best_params)
        return best_P, _compute_score(best_P)
    except Exception:
        # Fallback to default ring
        P = _get_ring_initialization()
        return P, _compute_score(P)

def _binary_search_radius(P, r_lo, r_hi, deadline, max_iter=60):
    """
    High-precision polish: binary search on radius with constraint projection.
    For a given r, check if all constraints are satisfied; if not, project points inward.
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
        diff = P_test[:, np.newaxis, :] - P_test[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pairwise_ok = np.all(dist >= 2.0 * r_mid - 1e-12)
        
        if boundary_ok and pairwise_ok:
            # Feasible: increase lower bound
            r_lo = r_mid
            best_P = P_test.copy()
            best_r = r_mid
        else:
            # Infeasible: decrease upper bound and try to project
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
            for _ in range(10):
                diff = P_fixed[:, np.newaxis, :] - P_fixed[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dist, np.inf)
                
                violating = dist < 2.0 * r_mid
                if not np.any(violating):
                    break
                
                # Move violating pairs apart
                for i in range(N):
                    for j in range(i+1, N):
                        if violating[i, j]:
                            d = P_fixed[i] - P_fixed[j]
                            norm_d = np.linalg.norm(d)
                            if norm_d < 1e-10:
                                d = np.array([1.0, 0.0])
                                norm_d = 1.0
                            unit_d = d / norm_d
                            # Each point moves half the gap
                            gap = (2.0 * r_mid - norm_d) / 2.0
                            P_fixed[i] += unit_d * gap
                            P_fixed[j] -= unit_d * gap
            
            # Re-check if fixed configuration works
            norms = np.linalg.norm(P_fixed, axis=1)
            boundary_ok2 = np.all(norms + r_mid <= 1.0 + 1e-12)
            if boundary_ok2:
                diff2 = P_fixed[:, np.newaxis, :] - P_fixed[np.newaxis, :, :]
                dist2 = np.sqrt(np.sum(diff2**2, axis=2))
                np.fill_diagonal(dist2, np.inf)
                if np.all(dist2 >= 2.0 * r_mid - 1e-12):
                    r_lo = r_mid
                    best_P = P_fixed.copy()
                    best_r = r_mid
                    P = P_fixed  # Use fixed config as new base
    
    return best_P, best_r

def _get_hex_initial_points():
    """Generate a distorted hexagonal lattice and select the best N points."""
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

def _local_refine(P, r, max_iter=300, tol=1e-12):
    """Local refinement using gradient-based optimization with adaptive steps."""
    P_curr = P.copy()
    r_curr = r
    
    # Use scipy's L-BFGS-B for more effective local optimization
    def objective(P_flat):
        P_test = P_flat.reshape(N, 2)
        # Maximize the minimum radius, so minimize negative radius
        score = _compute_score(P_test)
        # Add penalty for points outside unit circle
        norms = np.linalg.norm(P_test, axis=1)
        outside_penalty = np.sum(np.maximum(norms - 1.0, 0.0)**2)
        return -score + outside_penalty * 100
    
    P_flat = P_curr.flatten()
    
    # Try L-BFGS-B optimization
    try:
        result = minimize(objective, P_flat, method='L-BFGS-B', 
                         options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        if result.fun < -r_curr + 1e-10:
            P_opt = result.x.reshape(N, 2)
            score_opt = _compute_score(P_opt)
            if score_opt > r_curr:
                P_curr = P_opt
                r_curr = score_opt
    except:
        pass
    
    # Additional manual refinement pass
    for it in range(max_iter // 2):
        diff = P_curr[:, np.newaxis, :] - P_curr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        forces = np.zeros_like(P_curr)
        
        # Circle-circle repulsion
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
        
        # Boundary attraction
        dist_origin = np.linalg.norm(P_curr, axis=1)
        overlap_cb = np.maximum(dist_origin + r_curr - 1.0, 0.0)
        mask_cb = overlap_cb > 0
        if np.any(mask_cb):
            # Ensure dist_origin is reshaped for broadcasting with (k, 2) unit_vec
            # dist_origin[mask_cb] is (k,), we need (k, 1) to multiply with (k, 2)
            unit_vec = P_curr[mask_cb] / dist_origin[mask_cb, np.newaxis]
            # overlap_cb[mask_cb] is (k,), reshape to (k, 1) for broadcasting
            forces[mask_cb] -= 2.0 * unit_vec * overlap_cb[mask_cb, np.newaxis]
        
        # Adaptive step size with line search
        norm_forces = np.linalg.norm(forces)
        if norm_forces < 1e-15:
            break
        
        # Try multiple step sizes
        best_step = 0
        best_score = r_curr
        for factor in [0.1, 0.05, 0.02, 0.01, 0.005]:
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
        
        if r_curr - best_score < tol:
            break
    
    return P_curr, r_curr

def _simulated_annealing(P_init, r_init, start_time, deadline):
    """Simulated annealing with adaptive perturbation and local refinement."""
    P_curr = P_init.copy()
    r_curr = r_init
    score_best = r_curr
    P_best = P_init.copy()
    
    T_start = 0.05
    T_end = 0.001
    max_iter = 5000
    no_improve_count = 0
    max_no_improve = 500
    
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        # Adaptive perturbation: larger when stuck
        perturb_scale = 0.02 + 0.03 * (1 + no_improve_count / max_no_improve)
        n_perturb = np.random.randint(1, 8)
        idx = np.random.choice(N, n_perturb, replace=False)
        P_trial = P_curr.copy()
        P_trial[idx] += np.random.normal(0, perturb_scale, (n_perturb, 2))
        
        # Keep points inside container
        norms = np.linalg.norm(P_trial, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            P_trial[mask] *= 0.98 / norms[mask, np.newaxis]
        
        E_curr = _energy(P_curr, r_curr)
        E_trial = _energy(P_trial, r_curr)
        dE = E_trial - E_curr
        
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            P_curr = P_trial
            new_score = _compute_score(P_curr)
            if new_score > score_best:
                r_curr = new_score
                score_best = new_score
                P_best = P_curr.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Periodic local refinement
        if it % 200 == 0 and it > 0:
            P_ref, score_ref = _local_refine(P_curr, r_curr, max_iter=50)
            if score_ref > score_best:
                P_curr = P_ref
                r_curr = score_ref
                score_best = score_ref
                P_best = P_ref.copy()
                no_improve_count = 0
    
    return P_best, score_best

def _optuna_like_search(P_init, r_init, start_time, deadline):
    """A more systematic search combining multiple strategies."""
    P_best = P_init.copy()
    score_best = r_init
    
    # Try multiple random restarts with local refinement
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb the current best
        P_trial = P_best + np.random.normal(0, 0.05, (N, 2))
        # Keep inside container
        norms = np.linalg.norm(P_trial, axis=1)
        mask = norms > 0.95
        P_trial[mask] *= 0.95 / norms[mask, np.newaxis]
        
        score_trial = _compute_score(P_trial)
        if score_trial > score_best:
            P_best = P_trial
            score_best = score_trial
        
        # Local refine
        P_ref, score_ref = _local_refine(P_trial, score_trial, max_iter=100)
        if score_ref > score_best:
            P_best = P_ref
            score_best = score_ref
    
    return P_best, score_best

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    # Phase 0: Optimize parametric ring structure with differential evolution
    # This is the key improvement - DE on ring params escapes local minima
    P_ring_opt, score_ring_opt = _optimize_ring_params(deadline)
    
    # Also generate other initializations
    P_hex = _get_hex_initial_points()
    score_hex = _compute_score(P_hex)
    P_ring = _get_ring_initialization()
    score_ring = _compute_score(P_ring)
    
    # Pick the best starting point
    candidates = [(P_ring_opt, score_ring_opt), (P_hex, score_hex), (P_ring, score_ring)]
    P_best, score_best = max(candidates, key=lambda x: x[1])
    
    # Phase 1: Local refinement from best start
    if time.time() < deadline - 50.0:
        P_ref, score_ref = _local_refine(P_best, score_best, max_iter=200)
        if score_ref > score_best:
            P_best = P_ref
            score_best = score_ref
    
    # Phase 2: Simulated annealing with periodic local refinement
    if time.time() < deadline - 40.0:
        P_sa, score_sa = _simulated_annealing(P_best, score_best, start_time, deadline)
        if score_sa > score_best:
            P_best = P_sa
            score_best = score_sa
    
    # Phase 3: Multiple local refinement passes from different perturbations
    if time.time() < deadline - 25.0:
        for restart in range(5):
            if time.time() > deadline - 20.0:
                break
            P_trial = P_best + np.random.normal(0, 0.02, (N, 2))
            norms = np.linalg.norm(P_trial, axis=1)
            mask = norms > 0.95
            if np.any(mask):
                P_trial[mask] *= 0.95 / norms[mask, np.newaxis]
            
            P_ref, score_ref = _local_refine(P_trial, _compute_score(P_trial), max_iter=150)
            if score_ref > score_best:
                P_best = P_ref
                score_best = score_ref
    
    # Phase 4: Final local polish
    if time.time() < deadline - 10.0:
        P_polish, score_polish = _local_refine(P_best, score_best, max_iter=500, tol=1e-14)
        if score_polish > score_best:
            P_best = P_polish
            score_best = score_polish
    
    # Phase 5: Binary search on radius for high-precision final answer
    if time.time() < deadline - 5.0:
        # Estimate tight bounds
        r_lo = score_best * 0.99
        r_hi = score_best * 1.01
        P_final, r_final = _binary_search_radius(P_best, r_lo, r_hi, deadline, max_iter=60)
        if r_final > score_best:
            P_best = P_final
            score_best = r_final
    
    # Ensure all points are inside the unit circle
    norms = np.linalg.norm(P_best, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        P_best[mask] *= (1.0 - 1e-9) / norms[mask, np.newaxis]
    
    # Final safety check
    final_score = _compute_score(P_best)
    if not np.isfinite(final_score) or final_score <= 0:
        # Fallback to hex lattice
        P_best = P_hex
        final_score = score_hex
    
    return P_best
# EVOLVE-BLOCK-END
