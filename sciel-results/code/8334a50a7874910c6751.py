"""Rewritten program for packing 50 equal circles in a unit circle using feasibility search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit


def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)


def _violation_objective_and_grad(params, r):
    """
    Objective: Sum of squared violations for a fixed radius r.
    Violations:
    1. Pairwise overlap: max(0, 2r - dist_ij)
    2. Boundary violation: max(0, norm_i - (1 - r))
    """
    centers = params.reshape(N, 2)
    
    # Calculate pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Pairwise violations
    # v_ij = max(0, 2r - d_ij)
    overlaps = 2.0 * r - pair_dists
    active_pairs = overlaps > 0
    v_pairs = np.zeros_like(overlaps)
    v_pairs[active_pairs] = overlaps[active_pairs]
    
    # Boundary violations
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_violations = norms - (1.0 - r)
    active_bounds = bound_violations > 0
    v_bounds = np.zeros_like(bound_violations)
    v_bounds[active_bounds] = bound_violations[active_bounds]
    
    # Objective: Sum of squares
    obj = np.sum(v_pairs**2) + np.sum(v_bounds**2)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pair contributions: d/dr (max(0, 2r-d)^2) is not needed, we are minimizing wrt positions
    # d/dx_i ( (2r - ||x_i - x_j||)^2 ) = 2 * (2r - ||x_i - x_j||) * (-1) * (x_i - x_j) / ||x_i - x_j||
    # = -2 * overlap * unit_vector
    if np.any(active_pairs):
        active_dists = pair_dists[active_pairs]
        active_overlaps = overlaps[active_pairs]
        active_i = i_idx[active_pairs]
        active_j = j_idx[active_pairs]
        
        # Avoid division by zero
        safe_dists = np.where(active_dists > 1e-10, active_dists, 1.0)
        unit_vecs = (centers[active_i] - centers[active_j]) / safe_dists[:, np.newaxis]
        
        # Gradient contribution for i
        grad_i = -2.0 * active_overlaps[:, np.newaxis] * unit_vecs
        # Gradient contribution for j
        grad_j = -grad_i
        
        # Accumulate
        np.add.at(grad, active_i, grad_i)
        np.add.at(grad, active_j, grad_j)
    
    # Boundary contributions: d/dx_i (max(0, ||x_i|| - (1-r))^2)
    # = 2 * (||x_i|| - (1-r)) * (x_i / ||x_i||)
    if np.any(active_bounds):
        active_norms = norms[active_bounds]
        active_violations = bound_violations[active_bounds]
        active_indices = np.where(active_bounds)[0]
        
        safe_norms = np.where(active_norms > 1e-10, active_norms, 1.0)
        unit_vecs = centers[active_indices] / safe_norms[:, np.newaxis]
        
        grad_b = 2.0 * active_violations[:, np.newaxis] * unit_vecs
        grad[active_indices] += grad_b
        
    return obj, grad


def _is_feasible(r, init_centers, deadline):
    """
    Check if circles of radius r can fit, starting from init_centers.
    Returns (is_feasible, best_centers).
    """
    if time.time() > deadline - 1:
        return False, init_centers
    
    try:
        res = minimize(
            _violation_objective_and_grad,
            init_centers.flatten(),
            args=(r,),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 100, 'gtol': 1e-5}
        )
        
        new_centers = res.x.reshape(N, 2)
        
        if np.any(~np.isfinite(new_centers)):
            return False, init_centers
            
        # Check final violation
        final_obj, _ = _violation_objective_and_grad(new_centers.flatten(), r)
        
        # Threshold for feasibility: very small violation
        if final_obj < 1e-6:
            return True, new_centers
        else:
            return False, new_centers
    except Exception:
        return False, init_centers


def _get_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.125
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_ring_initialization(n, seed=0):
    """Generate initial points on concentric rings."""
    rng = np.random.default_rng(seed)
    points = []
    
    # Center point
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.25
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 2: 12 points
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.5
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 3: 18 points
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.75
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 4: remaining points
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1) + np.pi / max(remaining, 1)
        r = 0.9
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    return np.array(points[:n])


def _get_random_initialization(n, seed=0):
    """Generate random initial points."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)


def construct():
    start_time = time.time()
    
    # Initializations to try
    initializations = [
        _get_hex_lattice(N),
        _get_ring_initialization(N, seed=0),
        _get_random_initialization(N, seed=42),
        _get_ring_initialization(N, seed=1),
        _get_random_initialization(N, seed=123),
    ]
    
    # Binary search for radius
    # Lower bound: definitely fits
    # Upper bound: definitely doesn't fit
    low_r = 0.10
    high_r = 0.14
    
    # Ensure low_r is feasible to establish a baseline
    best_centers = initializations[0]
    best_score = _compute_score(best_centers)
    
    # Try to find a feasible radius in the range
    # We will do a few binary search steps
    # To save time, we limit the number of iterations and restarts
    
    # Pre-compute a good starting point for the highest known radius
    # We'll iterate through a few candidate radii
    
    candidates = []
    # Known optimal is ~0.1258
    # Let's search around that
    search_range = np.linspace(0.120, 0.130, 20)
    
    # Sort candidates by radius descending? No, binary search is better.
    # But binary search requires many feasibility checks.
    # Each feasibility check is an optimization.
    # Let's try a coarse grid search first, then refine.
    
    # Coarse search
    grid = np.linspace(0.120, 0.130, 10)
    
    best_feasible_r = 0.0
    best_feasible_centers = None
    
    for r in grid:
        if time.time() > DEADLINE - 20:
            break
            
        # Try a few initializations for this r
        found = False
        for init in initializations[:3]: # Limit restarts per radius
            if time.time() > DEADLINE - 10:
                break
                
            feasible, centers = _is_feasible(r, init, DEADLINE)
            if feasible:
                found = True
                if r > best_feasible_r:
                    best_feasible_r = r
                    best_feasible_centers = centers
                break # If feasible, move to next r? 
                # Actually, if r is feasible, we want to try higher r.
                # So we should break the init loop and continue to next r.
        
        # If not found, continue to next r (lower r)
        
    # If we found a feasible radius, refine it
    if best_feasible_centers is not None:
        # Binary search between best_feasible_r and high_r (or next grid point)
        # Let's do a few iterations of binary search
        low = best_feasible_r
        high = 0.135 # Upper bound for refinement
        
        for _ in range(5):
            if time.time() > DEADLINE - 5:
                break
                
            mid = (low + high) / 2.0
            
            feasible, centers = _is_feasible(mid, best_feasible_centers, DEADLINE)
            if feasible:
                low = mid
                best_feasible_r = mid
                best_feasible_centers = centers
            else:
                high = mid
        
        # Final polish on the best centers found
        # Use the original soft-min optimizer to squeeze out more precision
        if time.time() < DEADLINE - 5:
            # Re-use the old optimization logic for final polish?
            # Or just return the feasible centers.
            # The feasible centers should be very close to optimal.
            # Let's try a quick polish with L-BFGS on the soft-min objective
            # to ensure we maximize the score, not just feasibility.
            
            def _soft_min_obj(params, beta=100.0):
                centers = params.reshape(N, 2)
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(dists, np.inf)
                min_pair = np.min(dists) / 2.0
                norms = np.sqrt(np.sum(centers * centers, axis=1))
                min_bound = np.min(1.0 - norms)
                # Soft min of the two constraints?
                # Actually, just minimize -min(min_pair, min_bound)
                # This is non-smooth.
                # Let's stick to the feasible centers if they are good.
                pass
            
            # Since we already have a feasible configuration for r, 
            # and the score is at least r, we can just return it.
            # But maybe we can improve it slightly by running the old optimizer
            # starting from these centers.
            
            # Let's try one step of the old optimizer
            try:
                def _obj_grad(params, beta):
                    centers = params.reshape(N, 2)
                    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                    dists = np.sqrt(np.sum(diff * diff, axis=2))
                    i_idx, j_idx = np.triu_indices(N, k=1)
                    pair_dists = dists[i_idx, j_idx]
                    norms = np.sqrt(np.sum(centers * centers, axis=1))
                    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
                    m = np.min(vals)
                    exp_vals = np.exp(-beta * (vals - m))
                    S = np.sum(exp_vals)
                    soft_min_val = m - (1.0 / beta) * np.log(S)
                    obj = -soft_min_val
                    
                    weights = exp_vals / S
                    grad = np.zeros_like(centers)
                    
                    pair_weights = weights[:N*(N-1)//2]
                    for k in range(len(pair_weights)):
                        w = pair_weights[k]
                        i, j = i_idx[k], j_idx[k]
                        d = dists[i, j]
                        if d > 1e-10:
                            vec = (centers[i] - centers[j]) / d
                            grad[i] += w * vec / 2.0
                            grad[j] -= w * vec / 2.0
                    
                    bound_weights = weights[N*(N-1)//2:]
                    for i in range(N):
                        w = bound_weights[i]
                        if norms[i] > 1e-10:
                            grad[i] -= w * centers[i] / norms[i]
                    
                    return obj, -grad

                res = minimize(
                    _obj_grad,
                    best_feasible_centers.flatten(),
                    args=(100.0,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 50, 'gtol': 1e-6}
                )
                new_centers = res.x.reshape(N, 2)
                new_score = _compute_score(new_centers)
                old_score = _compute_score(best_feasible_centers)
                
                if new_score > old_score:
                    best_feasible_centers = new_centers
                    best_score = new_score
            except Exception:
                pass
    
    # If no feasible radius found in the grid, fallback to initial best
    if best_feasible_centers is None:
        # Try to optimize the initial hex lattice with the old method
        # to get a valid answer
        best_centers = initializations[0]
        best_score = _compute_score(best_centers)
        
        # Quick optimization
        try:
            def _obj_grad(params, beta):
                centers = params.reshape(N, 2)
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff * diff, axis=2))
                i_idx, j_idx = np.triu_indices(N, k=1)
                pair_dists = dists[i_idx, j_idx]
                norms = np.sqrt(np.sum(centers * centers, axis=1))
                vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
                m = np.min(vals)
                exp_vals = np.exp(-beta * (vals - m))
                S = np.sum(exp_vals)
                soft_min_val = m - (1.0 / beta) * np.log(S)
                obj = -soft_min_val
                
                weights = exp_vals / S
                grad = np.zeros_like(centers)
                
                pair_weights = weights[:N*(N-1)//2]
                for k in range(len(pair_weights)):
                    w = pair_weights[k]
                    i, j = i_idx[k], j_idx[k]
                    d = dists[i, j]
                    if d > 1e-10:
                        vec = (centers[i] - centers[j]) / d
                        grad[i] += w * vec / 2.0
                        grad[j] -= w * vec / 2.0
                
                bound_weights = weights[N*(N-1)//2:]
                for i in range(N):
                    w = bound_weights[i]
                    if norms[i] > 1e-10:
                        grad[i] -= w * centers[i] / norms[i]
                
                return obj, -grad

            for init in initializations:
                if time.time() > DEADLINE - 5:
                    break
                res = minimize(
                    _obj_grad,
                    init.flatten(),
                    args=(50.0,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 100, 'gtol': 1e-6}
                )
                new_centers = res.x.reshape(N, 2)
                new_score = _compute_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers
        except Exception:
            pass
        
        return best_centers
    
    return best_feasible_centers
