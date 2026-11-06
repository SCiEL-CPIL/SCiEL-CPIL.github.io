import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Computes the maximum radius for the given points.
    r = min(min_pairwise_dist / 2, min_dist_to_boundary)
    """
    # Distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def energy_and_grad(x):
    """
    Objective function and gradient for L-BFGS-B.
    Minimizes sum of repulsive forces + boundary penalty.
    """
    points = x.reshape(N, 2)
    
    # Pairwise repulsion
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    dists_sq_safe = np.maximum(dists_sq, 1e-8)
    
    # Energy term: sum_{i<j} 1/d_ij^2
    iu = np.triu_indices(N, k=1)
    e_pair = np.sum(1.0 / dists_sq_safe[iu])
    
    # Gradient of pair term
    # d/dx_i (1/||x_i - x_j||^2) = -2 (x_i - x_j) / ||x_i - x_j||^4
    grad = np.zeros_like(points)
    
    # Vectorized gradient accumulation
    mask = ~np.eye(N, dtype=bool)
    # F[i,j] = -2 * (x_i - x_j) / dists_sq_ij^2
    # We sum over j for each i
    # Note: dists_sq is symmetric
    # We compute the force on i from all j
    # Force matrix F has shape (N, N, 2)
    # F[i,j] = -2 * diff[i,j] / (dists_sq[i,j]**2)
    
    # To avoid memory blowup and inefficiency, we can do this in a loop or vectorized carefully.
    # For N=50, a vectorized approach is fast enough.
    
    # dists_sq_safe shape (N, N)
    # diff shape (N, N, 2)
    
    # We only need i != j
    # Let's compute the full matrix and mask the diagonal
    # factor = -2.0 / (dists_sq_safe ** 2)
    # factor[~mask] = 0
    # grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # This creates a large temporary array, but for N=50 it's 50*50*2 floats ~ 20KB, which is fine.
    factor = np.zeros_like(dists_sq_safe)
    factor[mask] = -2.0 / (dists_sq_safe[mask] ** 2)
    grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    grad += grad_pair
    
    # Boundary penalty
    # We want to penalize points outside the unit circle.
    # Use a smooth approximation of max(0, r-1)^2
    # Or simply a large weight on (r^2 - 1)^2 for r > 1?
    # A common smooth penalty is:
    # P(r) = 0.5 * (r - 1 + sqrt((r-1)^2 + eps))^2 / (2*eps) ? No.
    # Let's use: if r > 1, penalty = (r-1)^2.
    # To make it smooth everywhere, we can use:
    # penalty = 0.5 * (r - 1 + sqrt((r-1)^2 + eps^2))^2 / (2*eps^2) * eps^2 ?
    # Simpler: Just use a large weight on the squared distance outside.
    # Since L-BFGS-B is a line search method, non-smoothness can be an issue.
    # Let's use a smooth barrier:
    # E_bound = sum_i log(1 / (1 - r_i^2)) ? No, that's for r < 1.
    # If we ensure points stay inside, we don't need a penalty.
    # But they might go outside.
    # Let's use a penalty that is 0 for r < 1 and rises for r > 1.
    # P(r) = max(0, r-1)^2 is not smooth at r=1.
    # Smooth approx: P(r) = (r - 1 + sqrt((r-1)^2 + eps^2))^2 / (4*eps^2) * eps^2 ?
    # Let's use a simpler one:
    # P(r) = 0.5 * (r - 1 + sqrt((r-1)^2 + eps^2))^2 / (2*eps^2)
    # This is the softplus squared?
    
    # Actually, a very effective trick is to use the potential:
    # E_bound = sum_i (r_i^2 - 1)^2 * (r_i^2 > 1)
    # But let's just use a large weight on the squared norm if it exceeds 1.
    # To keep it smooth, we can use:
    # E_bound = sum_i (r_i^2 - 1 + sqrt((r_i^2 - 1)^2 + eps^2))^2 / (4*eps^2)
    
    # For simplicity and speed, let's assume the repulsive force is strong enough to keep them in,
    # and add a hard penalty if they go out.
    # But L-BFGS-B might struggle with non-smooth.
    # Let's use a smooth penalty:
    # P(r) = 0.5 * (r - 1 + sqrt((r-1)^2 + eps^2))^2 / (2*eps^2)
    # This is the "soft" version of max(0, r-1)^2.
    eps = 1e-3
    norms = np.linalg.norm(points, axis=1)
    # We want to penalize r > 1
    # Let u = r - 1
    # P(u) = 0.5 * (u + sqrt(u^2 + eps^2))^2 / (2*eps^2)
    u = norms - 1.0
    soft_u = u + np.sqrt(u**2 + eps**2)
    e_bound = np.sum(soft_u**2) / (4.0 * eps**2)
    
    # Gradient of boundary term
    # d/du P(u) = (u + sqrt(u^2+eps^2)) / (2*eps^2) * (1 + u/sqrt(u^2+eps^2))
    # = soft_u / (2*eps^2) * (1 + u/sqrt(u^2+eps^2))
    # dP/dr_i = dP/du * du/dr_i = dP/du * 1 (since u = r-1)
    # dr_i/dx_i = x_i / r_i
    # So grad_bound_i = dP/dr_i * (x_i / r_i)
    
    grad_bound = np.zeros_like(points)
    sqrt_term = np.sqrt(u**2 + eps**2)
    dP_dr = soft_u / (2.0 * eps**2) * (1.0 + u / sqrt_term)
    
    # Avoid division by zero for r_i = 0
    norms_safe = np.maximum(norms, 1e-8)
    grad_bound = dP_dr[:, np.newaxis] * points / norms_safe[:, np.newaxis]
    
    grad += grad_bound
    
    total_energy = e_pair + e_bound
    return total_energy, grad

def _seed_hexagonal():
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _seed_random(seed_val):
    np.random.seed(seed_val)
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _optimize_from_seed(seed_points, deadline):
    x0 = seed_points.flatten()
    
    # Run L-BFGS-B
    result = minimize(
        energy_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if time.time() > deadline:
        return seed_points, compute_radius(seed_points)
        
    optimized_points = result.x.reshape(N, 2)
    score = compute_radius(optimized_points)
    
    # Ensure points are strictly inside for validity check in evaluator
    # The penalty should have kept them in, but just in case, project if slightly out
    norms = np.linalg.norm(optimized_points, axis=1)
    mask_out = norms > 1.0
    if np.any(mask_out):
        optimized_points[mask_out] = optimized_points[mask_out] / norms[mask_out, np.newaxis] * 0.999
        
    return optimized_points, score

def construct():
    start_time = time.time()
    # Leave 5 seconds for final checks and return
    deadline = start_time + 145.0
    
    best_points = None
    best_score = -1.0
    
    seeds = []
    seeds.append(_seed_hexagonal())
    seeds.append(_seed_random(42))
    seeds.append(_seed_random(123))
    seeds.append(_seed_random(999))
    
    # Add a ring-based seed
    ring_radii = [0, 0.3, 0.6, 0.85, 1.0]
    ring_counts = [1, 6, 12, 18, 13]
    pts = []
    for r, count in zip(ring_radii, ring_counts):
        if count == 0: continue
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    ring_seed = np.array(pts[:N])
    if len(ring_seed) == N:
        seeds.append(ring_seed)
        
    for seed in seeds:
        if time.time() > deadline:
            break
            
        current_points, score = _optimize_from_seed(seed, deadline)
        
        if score > best_score:
            best_score = score
            best_points = current_points.copy()
            
    # If no seed was processed (unlikely), return a valid default
    if best_points is None:
        best_points = _seed_hexagonal()
        best_score = compute_radius(best_points)
        
    return best_points
