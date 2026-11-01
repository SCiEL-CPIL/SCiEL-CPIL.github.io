import numpy as np
import time
from scipy.optimize import minimize

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 145.0  # safety margin before the 150 s wall-clock limit

def _smooth_hinge(x, eps=1e-3):
    """Smooth approximation of max(0, x)."""
    return 0.5 * (x + np.sqrt(x**2 + eps**2))

def _smooth_hinge_grad(x, eps=1e-3):
    """Gradient of smooth_hinge."""
    return 0.5 * (1 + x / np.sqrt(x**2 + eps**2))

def _pack_obj_and_grad(x, lam=10.0, eps=1e-3):
    """
    Objective and gradient for the packing problem using penalty method.
    x = [c_0, c_1, ..., c_{N-1}, r]
    Minimize -r + lambda * (sum of squared constraint violations)
    """
    c = x[:2*N].reshape(N, 2)
    r = x[2*N]
    
    # Pairwise distances
    # d_ij = ||c_i - c_j||
    # Constraint: d_ij >= 2r  =>  2r - d_ij <= 0
    # Violation: max(0, 2r - d_ij)
    
    # Boundary distances
    # b_i = 1 - ||c_i||
    # Constraint: b_i >= r => ||c_i|| <= 1-r => ||c_i|| - (1-r) <= 0
    # Violation: max(0, ||c_i|| - (1-r))
    
    # Compute pairwise distances
    # Efficient computation
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Only consider i < j
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    # Constraint violations
    # v_ij = 2r - d_ij
    v_pairs = 2.0 * r - pair_dists
    # Smooth hinge
    s_v_pairs = _smooth_hinge(v_pairs, eps)
    # Penalty term
    penalty_pairs = np.sum(s_v_pairs**2)
    
    # Boundary violations
    norms = np.linalg.norm(c, axis=1)
    v_bound = norms - (1.0 - r)
    s_v_bound = _smooth_hinge(v_bound, eps)
    penalty_bound = np.sum(s_v_bound**2)
    
    total_penalty = penalty_pairs + penalty_bound
    obj = -r + lam * total_penalty
    
    # Gradient
    grad = np.zeros_like(x)
    
    # Gradient w.r.t r
    # d/dr (-r) = -1
    # d/dr (lam * sum s_v_pairs^2) = lam * sum 2 * s_v_pairs * ds_v_pairs/dr
    # ds_v_pairs/dr = d/dr smooth_hinge(2r - d_ij) = 2 * smooth_hinge_grad(2r - d_ij)
    grad_r = -1.0
    if len(v_pairs) > 0:
        ds_v_pairs_dr = 2.0 * _smooth_hinge_grad(v_pairs, eps)
        grad_r += lam * np.sum(2.0 * s_v_pairs * ds_v_pairs_dr)
        
    # d/dr (lam * sum s_v_bound^2) = lam * sum 2 * s_v_bound * ds_v_bound/dr
    # ds_v_bound/dr = d/dr smooth_hinge(||c_i|| - (1-r)) = d/dr smooth_hinge(||c_i|| - 1 + r)
    # = 1 * smooth_hinge_grad(||c_i|| - 1 + r)
    if len(v_bound) > 0:
        ds_v_bound_dr = _smooth_hinge_grad(v_bound, eps)
        grad_r += lam * np.sum(2.0 * s_v_bound * ds_v_bound_dr)
        
    grad[2*N] = grad_r
    
    # Gradient w.r.t c_i
    # d/dc_i (sum_{j != i} s(2r - ||c_i - c_j||)^2)
    # Let u_ij = 2r - ||c_i - c_j||
    # d/dc_i s(u_ij)^2 = 2 s(u_ij) * s'(u_ij) * du_ij/dc_i
    # du_ij/dc_i = - (c_i - c_j) / ||c_i - c_j||
    
    # We need to accumulate gradients for each c_i
    # It's easier to compute the full gradient matrix for pairs and then sum.
    
    # grad_c[i] = sum_{j != i} 2 * lam * s_v_pairs[idx] * ds_v_pairs[idx] * (- (c_i - c_j) / d_ij)
    # Note: ds_v_pairs is d s(v) / dv. v = 2r - d. dv/dc_i = -(c_i - c_j)/d.
    # So d s(v)^2 / dc_i = 2 s(v) s'(v) * (-(c_i - c_j)/d)
    
    # Let's compute the force vector for each pair and add to both c_i and c_j.
    # Force on i due to j: F_i = - grad_{c_i} (penalty term for pair i,j)
    # Wait, we are minimizing the penalty. So we want the gradient of the penalty.
    # grad_{c_i} penalty_pair = 2 * lam * s(v) * s'(v) * (-(c_i - c_j)/d)
    
    # Let's compute this efficiently.
    # We have pair_dists and v_pairs.
    # We need (c_i - c_j) / d_ij for each pair.
    
    # Reconstruct diff for pairs
    # diff[i, j] = c_i - c_j
    # We have iu indices.
    c_i = c[iu[0]]
    c_j = c[iu[1]]
    diff_pairs = c_i - c_j
    # d_ij = pair_dists
    # unit vector u_ij = diff_pairs / pair_dists
    # Handle zero distance (should not happen with inf diagonal, but just in case)
    mask = pair_dists > 1e-8
    u_pairs = np.zeros_like(diff_pairs)
    u_pairs[mask] = diff_pairs[mask] / pair_dists[mask, np.newaxis]
    
    # Coefficient for each pair
    coeff = 2.0 * lam * s_v_pairs * _smooth_hinge_grad(v_pairs, eps)
    # coeff is shape (N_pairs,)
    
    # Gradient contribution from pair (i, j) to c_i is:
    # - coeff * u_ij  (since grad is w.r.t c_i, and term is -(c_i-c_j)/d)
    # Wait, v = 2r - d. d/dc_i d = (c_i - c_j)/d.
    # d/dc_i v = -(c_i - c_j)/d.
    # d/dc_i s(v)^2 = 2 s(v) s'(v) * (-(c_i - c_j)/d) = - coeff * u_ij.
    # So grad_{c_i} += - coeff * u_ij.
    # And grad_{c_j} += + coeff * u_ij. (by symmetry, d/dc_j v = (c_i - c_j)/d = -u_ij? No.
    # d/dc_j d = (c_j - c_i)/d = -u_ij.
    # d/dc_j v = -(-u_ij) = u_ij.
    # d/dc_j s(v)^2 = 2 s(v) s'(v) * u_ij = coeff * u_ij.
    
    # So we add -coeff*u to c_i and +coeff*u to c_j.
    
    # Use np.add.at to accumulate
    # grad_c is shape (N, 2)
    grad_c = np.zeros((N, 2))
    np.add.at(grad_c, iu[0], -coeff[:, np.newaxis] * u_pairs)
    np.add.at(grad_c, iu[1], coeff[:, np.newaxis] * u_pairs)
    
    # Boundary gradient
    # Penalty: sum_i s(||c_i|| - (1-r))^2
    # d/dc_i = 2 * lam * s(v_i) * s'(v_i) * d/dc_i (||c_i|| - 1 + r)
    # d/dc_i ||c_i|| = c_i / ||c_i||
    # So grad_{c_i} += 2 * lam * s(v_i) * s'(v_i) * (c_i / ||c_i||)
    
    if N > 0:
        # norms is shape (N,)
        # v_bound is shape (N,)
        # s_v_bound is shape (N,)
        # s_grad_bound = smooth_hinge_grad(v_bound)
        s_grad_bound = _smooth_hinge_grad(v_bound, eps)
        coeff_bound = 2.0 * lam * s_v_bound * s_grad_bound
        # coeff_bound is shape (N,)
        # unit vectors
        # c_i / ||c_i||
        mask_norm = norms > 1e-8
        u_bound = np.zeros_like(c)
        u_bound[mask_norm] = c[mask_norm] / norms[mask_norm, np.newaxis]
        
        grad_c += coeff_bound[:, np.newaxis] * u_bound
        
    # Flatten grad_c into grad
    grad[:2*N] = grad_c.flatten()
    
    return obj, grad

def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = 0.5 * min_pair_dist
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0

def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97

def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    
    # Initial guess for radius
    # Area of unit circle is pi. Area of N circles is N * pi * r^2.
    # N * r^2 <= 1 => r <= 1/sqrt(N)
    # 1/sqrt(300) ~ 0.0577
    # Best known is 0.0531
    init_r = 0.05
    
    # Initial points: Hex lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    pts = _lattice_seed(spacing)
    
    # Construct initial x vector
    # x = [c_0, ..., c_{N-1}, r]
    x0 = np.concatenate([pts.flatten(), [init_r]])
    
    best_r = _score(pts)
    best_pts = pts.copy()
    
    # Optimization loop with multiple restarts
    lam = 10.0
    
    while time.time() - t0 < DEADLINE:
        # Define objective and gradient for this lambda
        def obj_grad(x):
            return _pack_obj_and_grad(x, lam=lam)
        
        # Run L-BFGS-B
        # Bounds:
        # c_i coordinates: [-1, 1]
        # r: [0, 0.1]
        bounds = [(-1.0, 1.0)] * (2 * N) + [(0.0, 0.1)]
        
        try:
            res = minimize(
                lambda x: obj_grad(x)[0],
                x0,
                jac=lambda x: obj_grad(x)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False}
            )
        except Exception as e:
            # If optimization fails, break
            break
            
        c_opt = res.x[:2*N].reshape(N, 2)
        r_opt = res.x[2*N]
        
        # Compute true score
        r_true = _score(c_opt)
        
        if r_true > best_r:
            best_r = r_true
            best_pts = c_opt.copy()
            # Update x0 for next iteration, maybe with some perturbation
            # Keep the radius from the solution
            x0 = np.concatenate([c_opt.flatten(), [r_true]])
            
            # Increase lambda to enforce constraints more strictly?
            # Or keep it constant.
            # Let's try increasing lambda slightly to push towards feasibility
            # lam *= 1.5
            
        else:
            # If no improvement, perturb the starting point
            # Add noise
            noise = 0.01 * rng.standard_normal((N, 2))
            x0 = np.concatenate([(c_opt + noise).flatten(), [r_true]])
            # Also try a fresh start occasionally
            if rng.random() < 0.1:
                pts = _lattice_seed(spacing)
                x0 = np.concatenate([pts.flatten(), [init_r]])
        
        # Check time
        if time.time() - t0 > DEADLINE - 5.0:
            break

    # Final safety check
    final_r = _score(best_pts)
    if final_r <= 0.0:
        best_pts = _lattice_seed(2.0 / (np.sqrt(N) + 1.0))
        final_r = _score(best_pts)

    return best_pts
