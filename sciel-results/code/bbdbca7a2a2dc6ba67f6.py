"""Pack 300 equal circles in the unit disk, maximizing the common radius."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
DEADLINE = 145.0  # safety margin before 150s kill
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return max(0.0, DEADLINE - _elapsed())


def _compute_radius(pts):
    """Compute the packing radius for a set of centers."""
    n = len(pts)
    if n < 2:
        return min(1.0 - np.linalg.norm(pts, axis=1)).item()
    # Boundary term
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    # Pairwise distances (use chunking for memory if needed, but 300 is fine)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(boundary.min(), min_pair)


def _soft_min(x, k):
    """Soft minimum via log-sum-exp for smooth optimization."""
    # For large k, this approximates the true min
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    # Subtract max for numerical stability (for log-sum-exp of exp(-k*x))
    # We want soft_min(x) ≈ min(x)
    # soft_min = -1/k * log(sum(exp(-k*x)))
    # = -1/k * ( -k*max(x) + log(sum(exp(-k*(x - max(x))))) )
    # = max(x) - 1/k * log(sum(exp(-k*(x - max(x)))))
    x_max = np.max(x)
    shifted = x - x_max
    # exp(-k * shifted) where shifted <= 0
    log_sum = np.log(np.sum(np.exp(-k * shifted)))
    return x_max - log_sum / k


def _objective(params, k):
    """
    Objective function for L-BFGS-B.
    We maximize r, which is min(boundary, min_pair/2).
    Equivalent to minimizing -r.
    Using soft-min for smoothness.
    """
    pts = params.reshape(-1, 2)
    # Boundary distances
    dist_to_center = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - dist_to_center
    
    # Pairwise distances
    # For N=300, full matrix is 300x300 = 90k entries, fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    half_dists = dists / 2.0
    
    # Combine all constraints
    # We need soft_min of (boundary, half_dists)
    # To avoid building a huge vector every time, we can compute soft_min separately
    # but they need to be combined. 
    # Let's build the vector.
    # boundary has N elements, half_dists has N*(N-1) elements.
    # Total ~ 300 + 300*299 = 90000 elements.
    
    # For efficiency, let's compute soft_min of boundary and soft_min of pairs separately?
    # No, min(A, B) = min(min(A), min(B)).
    # So soft_min(A, B) ≈ soft_min(soft_min(A), soft_min(B))? No, that's not quite right.
    # Actually, min(all) = min( min(boundary), min(pairs) ).
    # So we can compute s1 = soft_min(boundary, k) and s2 = soft_min(half_dists, k)
    # Then result = soft_min([s1, s2], k)? 
    # Or simply min(s1, s2) if k is large enough?
    # Let's just concatenate for correctness, it's fast enough.
    
    # Actually, for L-BFGS-B, we need gradients.
    # Let's use a simpler approach: minimize -min(...) is non-differentiable.
    # We'll use the soft-min formulation.
    
    # To save memory and time, let's compute the soft min of the combined set.
    # But constructing a 90k vector every iteration is okay.
    
    # Let's try a different approach: 
    # Maximize r subject to constraints.
    # Or minimize sum of violations?
    
    # Let's stick to soft-min.
    # Vector of all "distances" (boundary and half-pairs)
    # This is expensive to build.
    
    # Alternative: Use a penalty method or just evaluate the true min for L-BFGS-B?
    # L-BFGS-B requires a smooth objective.
    
    # Let's compute the soft min properly.
    # For speed, we can compute soft_min of boundary and soft_min of pairs separately
    # and then take the min of those two?
    # min(min(A), min(B)) = min(all).
    # soft_min(all) is not exactly min(soft_min(A), soft_min(B)) but close for large k.
    # Let's just use the true min for the objective value, but that's non-smooth.
    
    # Better: Use the soft-min of the full vector.
    # To avoid building the full vector, we can use the formula:
    # soft_min(A ∪ B) = soft_min([soft_min(A), soft_min(B)])? No.
    
    # Let's just build the vector. 90k floats is small.
    all_dists = np.concatenate([boundary, half_dists.ravel()])
    return -_soft_min(all_dists, k)


def _gradient(params, k):
    """Gradient of the objective (negative soft-min)."""
    pts = params.reshape(-1, 2)
    dist_to_center = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - dist_to_center
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    half_dists = dists / 2.0
    
    all_dists = np.concatenate([boundary, half_dists.ravel()])
    
    # Gradient of soft_min(x, k)
    # d/dx_i soft_min = (k/k) * exp(-k*x_i) / sum(exp(-k*x_j))
    # = exp(-k*(x_i - max(x))) / sum(exp(-k*(x_j - max(x))))
    # Let's compute weights w_i = exp(-k*(x_i - x_max)) / sum(exp(-k*(x_j - x_max)))
    # Then grad soft_min = w (element-wise)
    # grad objective = -grad soft_min = -w
    
    x_max = np.max(all_dists)
    shifted = all_dists - x_max
    exp_terms = np.exp(-k * shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp
    
    # Now, we need to map these weights back to the gradient of the parameters.
    # This is complex. Let's use numerical gradient or a simpler objective.
    
    # Given the complexity, let's use a different strategy:
    # 1. Get a good initial guess.
    # 2. Use a force-directed relaxation with smooth forces.
    # 3. Polish with L-BFGS-B on a simpler objective if time permits.
    
    # For now, let's return a dummy gradient to avoid errors, but we won't use it.
    # We'll use method='L-BFGS-B' with numerical gradient if needed, or just use force-directed.
    return np.zeros_like(params)


def _force_directed_relaxation(pts, n_iters=1000, lr=0.001, k_repel=1.0, k_attract=0.1):
    """
    Relax the points using a force-directed method.
    Forces:
    - Repulsion between all pairs (inverse square or similar)
    - Attraction to keep them inside the disk (or repulsion from boundary)
    """
    pts = pts.copy()
    n = len(pts)
    
    for it in range(n_iters):
        if _remaining() < 10.0:
            break
            
        forces = np.zeros_like(pts)
        
        # Repulsion between pairs
        # Use a soft potential to avoid singularities
        # F_ij = k_repel * (1/d_ij^2) * (p_i - p_j) / d_ij  (for d_ij < d_max)
        # Or simpler: F_ij = k_repel * (p_i - p_j) / (d_ij^2 + eps)^1.5
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, 1e-10)
        dists = np.sqrt(dists_sq)
        
        # Force magnitude ~ 1/d^2
        # Vector force = diff / d^3 * k
        # To avoid explosion, cap the force or use a smoother function
        # Let's use F = k * diff / (d^2 + eps)^1.5
        eps = 1e-6
        factor = k_repel / (dists_sq + eps)**1.5
        factor[np.diag_indices_from(factor)] = 0.0
        
        forces += np.sum(diff * factor[:, :, np.newaxis], axis=1)
        
        # Boundary constraint: push points inward if they are too close to the edge
        dist_to_center = np.linalg.norm(pts, axis=1)
        # If dist_to_center > 1 - r_est, push inward
        # Estimate r_est
        r_est = _compute_radius(pts)
        margin = 0.1 * r_est + 1e-4
        boundary_force = np.zeros_like(pts)
        for i in range(n):
            if dist_to_center[i] > 1.0 - margin:
                # Push toward center
                direction = -pts[i] / (dist_to_center[i] + 1e-10)
                strength = k_attract * (dist_to_center[i] - (1.0 - margin))
                boundary_force[i] += strength * direction
        forces += boundary_force
        
        # Update positions
        pts += lr * forces
        
        # Keep points inside the disk (hard constraint)
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] /= dists[mask, np.newaxis] * (1.0 + 1e-8)
        
        # Decay learning rate
        lr *= 0.999
    
    return pts


def _polish(pts, max_time=10.0):
    """
    Polish the configuration using L-BFGS-B on a smooth objective.
    """
    start = time.time()
    params = pts.flatten()
    
    def obj(p):
        if time.time() - start > max_time:
            return 0.0  # Signal to stop
        pts_p = p.reshape(-1, 2)
        r = _compute_radius(pts_p)
        return -r
    
    def grad(p):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(p)
        for i in range(len(p)):
            p_plus = p.copy()
            p_plus[i] += eps
            p_minus = p.copy()
            p_minus[i] -= eps
            g[i] = (obj(p_plus) - obj(p_minus)) / (2 * eps)
        return g
    
    # Use L-BFGS-B with numerical gradient
    # But this is very slow for 600 variables.
    # Let's skip this if it's too slow.
    
    # Alternative: Use a simple local search or just return the current best.
    # For now, let's just do a few iterations of force-directed with smaller step size.
    
    pts = pts.copy()
    for _ in range(100):
        if time.time() - start > max_time:
            break
        pts = _force_directed_relaxation(pts, n_iters=10, lr=0.0001, k_repel=1.0, k_attract=0.1)
    
    return pts


def _generate_initial_config(seed):
    """
    Generate an initial configuration using a hexagonal lattice or random points.
    """
    rng = np.random.default_rng(seed)
    
    # Try a few different strategies
    # 1. Hexagonal lattice
    # 2. Random points with rejection sampling
    # 3. Ring-based
    
    # Let's use a combination: start with a hexagonal lattice, then perturb
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            # Select N points, prefer those closer to the center for better packing
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx].copy()
            # Add small random perturbation
            pts += rng.normal(0, spacing * 0.1, size=pts.shape)
            # Ensure they are inside the disk
            dists = np.linalg.norm(pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                pts[mask] /= dists[mask, np.newaxis] * (1.0 + 1e-8)
            return pts
        spacing *= 0.98


def construct():
    """
    Construct the optimal packing of 300 circles in the unit disk.
    """
    best_pts = None
    best_radius = 0.0
    
    # Try multiple seeds
    seeds = [42, 123, 456, 789, 1011, 2022, 3033, 4044]
    
    for seed in seeds:
        if _remaining() < 15.0:
            break
            
        # Generate initial configuration
        pts = _generate_initial_config(seed)
        
        # Relax using force-directed method
        pts = _force_directed_relaxation(pts, n_iters=500, lr=0.002, k_repel=1.0, k_attract=0.1)
        
        # Polish
        pts = _polish(pts, max_time=5.0)
        
        # Evaluate
        r = _compute_radius(pts)
        
        if r > best_radius:
            best_radius = r
            best_pts = pts.copy()
    
    # If we still have time, do a final polish on the best configuration
    if best_pts is not None and _remaining() > 10.0:
        best_pts = _polish(best_pts, max_time=min(20.0, _remaining() - 5.0))
        best_radius = _compute_radius(best_pts)
    
    # Ensure the result is valid
    if best_pts is None:
        # Fallback: return a valid configuration
        best_pts = _generate_initial_config(42)
    
    # Final check: ensure all points are inside the disk
    dists = np.linalg.norm(best_pts, axis=1)
    mask = dists > 1.0
    if np.any(mask):
        best_pts[mask] /= dists[mask, np.newaxis] * (1.0 + 1e-8)
    
    return best_pts


# Test
if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Radius: {r}")
    print(f"Shape: {pts.shape}")
