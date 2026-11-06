"""
Combined solution: Physics-based relaxation via Soft-Min Optimization and Newton-Raphson Contact Solving.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _score_raw(centers):
    """
    Compute the actual score (minimum radius) for a set of centers.
    Returns 0.0 if invalid.
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance from origin
    norms = np.sqrt(np.sum(centers**2, axis=1))
    
    # Check all inside unit circle (with small tolerance for numerical noise)
    if np.any(norms > 1.0 + 1e-9):
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    min_edge_dist = np.min(1.0 - norms)
    
    r = min(min_pair_dist / 2.0, min_edge_dist)
    
    # If any circle is effectively outside, score is 0
    if min_edge_dist < -1e-9:
        return 0.0
        
    return max(r, 0.0)

def _soft_min_objective(x, beta=200.0):
    """
    Smooth approximation of the negative score using log-sum-exp.
    We want to maximize r = min(constraints).
    Minimize -soft_min(constraints).
    
    Constraints:
    1. Pairwise distances / 2
    2. Distance to boundary (1 - norm)
    """
    centers = x.reshape(N, 2)
    
    # Boundary constraints
    norms = np.sqrt(np.sum(centers**2, axis=1))
    edge_constraints = 1.0 - norms
    
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Take upper triangle to avoid duplicates
    iu = np.triu_indices(N, 1)
    pair_constraints = dists[iu] / 2.0
    
    # Combine all constraints
    all_constraints = np.concatenate([edge_constraints, pair_constraints])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x_k)))
    # Numerically stable version:
    # Let y_k = -beta * x_k
    # soft_min = -1/beta * (max(y) + log(sum(exp(y - max(y)))))
    
    y = -beta * all_constraints
    max_y = np.max(y)
    
    # Handle case where max_y is very large
    exp_sum = np.sum(np.exp(y - max_y))
    
    if exp_sum == 0:
        return 0.0
        
    log_sum_exp = max_y + np.log(exp_sum)
    soft_min_val = -log_sum_exp / beta
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _generate_hexagonal_initial(r_init):
    """
    Generate a hexagonal lattice packing with given circle radius.
    Returns N points closest to origin.
    """
    points = []
    spacing = 2 * r_init
    # Generate hexagonal lattice points
    limit = int(1.0 / spacing) + 2
    
    for j in range(-limit, limit + 1):
        for i in range(-limit, limit + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.sqrt(x*x + y*y)
            # Only keep points that can fit a circle of radius r_init
            if r <= 1.0 - r_init - 1e-9:
                points.append([x, y])
    
    if len(points) < N:
        return None
    
    points = np.array(points)
    # Sort by distance from origin and take N
    norms = np.sqrt(np.sum(points**2, axis=1))
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def _generate_ring_initial():
    """
    Generate a ring-based initial configuration.
    """
    centers = []
    # Center
    centers.append([0.0, 0.0])
    
    # Rings: 1 + 6 + 12 + 18 + 13 = 50
    ring_configs = [
        (6, 0.25),
        (12, 0.50),
        (18, 0.75),
        (13, 0.92)
    ]
    
    for count, r in ring_configs:
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for a in angles:
            centers.append([r * np.cos(a), r * np.sin(a)])
    
    centers = np.array(centers[:N])
    return centers

def _newton_refine(centers, max_iter=50, tol=1e-12):
    """
    Refine the configuration by solving the active contact constraints
    using Newton-Raphson. This helps achieve high precision in the final radius.
    """
    centers = centers.copy()
    
    for _ in range(max_iter):
        # Calculate distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        norms = np.sqrt(np.sum(centers**2, axis=1))
        
        # Current radius estimate
        r_min = min(np.min(dists) / 2.0, np.min(1.0 - norms))
        
        # Identify active constraints (those close to the minimum)
        # We consider a constraint active if it's within a small epsilon of the min
        eps = 1e-6
        active_pairs = np.argwhere(dists < 2.0 * r_min + eps)
        active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
        
        active_bounds = np.argwhere(1.0 - norms < r_min + eps)
        active_bounds = active_bounds.flatten()
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
            
        # Build linear system J * delta = b
        # Variables: 2N (dx, dy for each center)
        # Equations: For each active pair (i, j), dist(i, j) = 2*r_min
        #            For each active bound (i), norm(i) = 1 - r_min
        
        n_eq = len(active_pairs) + len(active_bounds)
        if n_eq == 0:
            break
            
        J = np.zeros((n_eq, 2 * N))
        b = np.zeros(n_eq)
        
        eq_idx = 0
        # Pair constraints: |c_i - c_j| - 2*r = 0
        # Gradient w.r.t c_i: (c_i - c_j) / |c_i - c_j|
        # Gradient w.r.t c_j: (c_j - c_i) / |c_i - c_j| = - grad_i
        for i, j in active_pairs:
            d_vec = centers[i] - centers[j]
            d_mag = np.linalg.norm(d_vec)
            if d_mag < 1e-12:
                continue # Skip if coincident
            grad = d_vec / d_mag
            J[eq_idx, 2*i:2*i+2] = grad
            J[eq_idx, 2*j:2*j+2] = -grad
            b[eq_idx] = 2.0 * r_min - d_mag
            eq_idx += 1
            
        # Boundary constraints: |c_i| - (1-r) = 0
        # Gradient w.r.t c_i: c_i / |c_i|
        for i in active_bounds:
            c_norm = np.linalg.norm(centers[i])
            if c_norm < 1e-12:
                continue
            grad = centers[i] / c_norm
            J[eq_idx, 2*i:2*i+2] = grad
            b[eq_idx] = (1.0 - r_min) - c_norm
            eq_idx += 1
            
        # Solve J * delta = b
        # Use least squares if overdetermined
        try:
            delta, residuals, rank, sv = np.linalg.lstsq(J, b, rcond=None)
        except np.linalg.LinAlgError:
            break
            
        if rank < n_eq:
            # Rank deficient, reduce step size
            pass
            
        # Apply step with damping to ensure we don't overshoot
        # We want to move in the direction that satisfies constraints
        # Note: The gradient points in direction of increasing distance.
        # We want to increase distance to 2r.
        # So we add delta.
        
        # Simple damping: if step is too large, scale it down
        max_step = 0.1
        step_norm = np.linalg.norm(delta)
        if step_norm > max_step:
            delta = delta * (max_step / step_norm)
            
        new_centers = centers + delta
        
        # Ensure new centers are inside the unit circle (hard constraint for validity)
        new_norms = np.linalg.norm(new_centers, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            new_centers[mask] *= (1.0 / new_norms[mask, np.newaxis])
            
        # Check if score improved or is stable
        new_score = _score_raw(new_centers)
        if new_score >= r_min - 1e-9:
            centers = new_centers
        else:
            # If score dropped, try a smaller step
            delta = delta * 0.5
            new_centers = centers + delta
            new_norms = np.linalg.norm(new_centers, axis=1)
            mask = new_norms > 1.0
            if np.any(mask):
                new_centers[mask] *= (1.0 / new_norms[mask, np.newaxis])
            new_score = _score_raw(new_centers)
            if new_score >= r_min - 1e-9:
                centers = new_centers
            else:
                break # Failed to improve, stop
                
        # Convergence check
        if np.linalg.norm(delta) < tol:
            break
            
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s kill
    
    best_score = 0.0
    best_centers = None
    
    # List of initial configurations
    initial_configs = []
    
    # 1. Hexagonal lattices with different assumed radii
    for r_init in [0.10, 0.11, 0.12, 0.13, 0.14]:
        cfg = _generate_hexagonal_initial(r_init)
        if cfg is not None:
            initial_configs.append(cfg)
            
    # 2. Ring layout
    ring_cfg = _generate_ring_initial()
    initial_configs.append(ring_cfg)
    
    # 3. Random valid configurations
    rng = np.random.default_rng(42)
    for _ in range(5):
        cfg = rng.uniform(-0.9, 0.9, size=(N, 2))
        # Project inside circle to ensure validity start
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.9
        if np.any(mask):
            cfg[mask] *= 0.9 / norms[mask, np.newaxis]
        initial_configs.append(cfg)
    
    # Optimization parameters
    betas = [50.0, 100.0, 200.0]
    
    # Optimize each initial configuration
    for idx, x0 in enumerate(initial_configs):
        if time.time() > deadline - 10.0:
            break
            
        # Try with different beta values
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                # L-BFGS-B is good for this
                result = minimize(
                    _soft_min_objective,
                    x0.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                centers = result.x.reshape(N, 2)
                
                # Check validity and score
                score = _score_raw(centers)
                
                if score > best_score:
                    best_score = score
                    best_centers = centers
                    
            except Exception:
                continue
    
    # Refine the best solution found so far using Newton-Raphson
    if best_centers is not None and time.time() < deadline - 5.0:
        refined_centers = _newton_refine(best_centers, max_iter=100, tol=1e-14)
        refined_score = _score_raw(refined_centers)
        if refined_score > best_score:
            best_score = refined_score
            best_centers = refined_centers
            
    # If we have time, do a few more perturb-and-refine cycles
    if best_centers is not None and time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline - 2.0:
                break
            
            # Perturb best solution
            noise = rng.normal(0, 0.02, size=best_centers.shape)
            perturbed = best_centers + noise
            # Project inside
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.99
            if np.any(mask):
                perturbed[mask] *= 0.99 / norms[mask, np.newaxis]
            
            # Quick L-BFGS polish
            try:
                result = minimize(
                    _soft_min_objective,
                    perturbed.flatten(),
                    args=(200.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                centers = result.x.reshape(N, 2)
                
                # Newton refine
                refined = _newton_refine(centers, max_iter=50, tol=1e-14)
                score = _score_raw(refined)
                
                if score > best_score:
                    best_score = score
                    best_centers = refined
                    
            except Exception:
                continue

    if best_centers is None:
        # Fallback: return a valid, simple configuration
        # This shouldn't happen, but ensures validity
        best_centers = _generate_hexagonal_initial(0.1)
        if best_centers is None:
             # Extremely fallback: grid
             best_centers = np.zeros((N, 2))
             for i in range(N):
                 best_centers[i] = [0.5 * np.cos(i), 0.5 * np.sin(i)]
    
    return best_centers

# EVOLVE-BLOCK-END
