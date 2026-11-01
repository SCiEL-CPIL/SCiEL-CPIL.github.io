# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300
# Ring structure: Hexagonal shells + partial outer ring
# 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29 sums to 300
ring_counts = np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29])
num_rings = len(ring_counts)

def _build_points(radii, phases):
    """Build point array from ring radii and phases."""
    points = []
    for i in range(num_rings):
        r = radii[i]
        phi = phases[i]
        n = ring_counts[i]
        if n == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + phi
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        points.append(np.column_stack([xs, ys]))
    
    if not points:
        return np.zeros((0, 2))
    
    return np.vstack(points)

def _compute_radius(points):
    """Compute the largest common radius for the given points."""
    if len(points) == 0:
        return 0.0
    if len(points) == 1:
        return 1.0 - np.hypot(*points[0])
    
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    
    return min(min_boundary, min_pair)

def _smooth_objective(params):
    """
    Smooth objective function for optimization.
    Minimizes sum of inverse square distances between circles and to boundary.
    This encourages maximizing the minimum distance.
    """
    k = num_rings
    radii = params[:k]
    phases = params[k:]
    
    # Ensure radii are non-negative (bounds handle this, but just in case)
    radii = np.maximum(radii, 0.0)
    
    points = _build_points(radii, phases)
    
    if len(points) != N:
        return 1e10 # Penalty for wrong count
    
    # Boundary penalty: sum(1 / (1 - |c_i|)^2)
    dists_boundary = 1.0 - np.hypot(points[:, 0], points[:, 1])
    # Clip to avoid division by zero
    dists_boundary = np.clip(dists_boundary, 1e-6, None)
    boundary_penalty = np.sum(1.0 / (dists_boundary ** 2))
    
    # Pairwise penalty: sum(1 / d_ij^2)
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    # Clip small distances to avoid overflow
    dist_matrix = np.clip(dist_matrix, 1e-6, None)
    pairwise_penalty = np.sum(1.0 / (dist_matrix ** 2))
    
    return pairwise_penalty + boundary_penalty

def _initial_guess(seed):
    """Generate a good initial guess based on hexagonal lattice."""
    rng = np.random.default_rng(seed)
    
    radii = np.zeros(num_rings)
    # Approximate hexagonal spacing
    # r_i ~ i * sqrt(3) * r_est
    # r_est ~ 0.05
    # spacing ~ 0.0866
    for i in range(num_rings):
        if ring_counts[i] > 1:
            radii[i] = i * 0.0866
        else:
            radii[i] = 0.0
            
    # Scale to fit within unit circle roughly
    if radii[-1] > 0.8:
        radii *= 0.8 / radii[-1]
        
    # Add jitter
    radii += rng.uniform(-0.01, 0.01, num_rings)
    radii = np.clip(radii, 0.0, 1.0)
    
    # Phases: staggered
    phases = np.zeros(num_rings)
    for i in range(1, num_rings):
        # Offset by half the angular spacing of the previous ring
        if ring_counts[i-1] > 0:
            phases[i] = np.pi / ring_counts[i-1]
        else:
            phases[i] = 0.0
        phases[i] += rng.uniform(-0.1, 0.1)
        
    return np.concatenate([radii, phases])

def construct():
    """Construct the packing configuration."""
    best_points = None
    best_r = 0.0
    
    bounds = [(0.0, 1.0)] * num_rings + [(0.0, 2 * np.pi)] * num_rings
    
    # Try multiple initial guesses
    for seed in range(10):
        x0 = _initial_guess(seed)
        
        try:
            # Use L-BFGS-B for smooth optimization
            result = minimize(
                _smooth_objective, x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            radii = result.x[:num_rings]
            phases = result.x[num_rings:]
            
            points = _build_points(radii, phases)
            
            if len(points) != N:
                continue
            
            r = _compute_radius(points)
            
            if r > best_r:
                best_r = r
                best_points = points
                
        except Exception:
            continue
            
    # Fallback if optimization failed or didn't find a valid config
    if best_points is None or best_r < 0.01:
        # Simple hexagonal grid fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            pts = np.array(rows)
            radius = np.hypot(pts[:, 0], pts[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            pts = pts[mask]
            if len(pts) >= N:
                return pts[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
            if spacing < 0.01:
                break
                
    return best_points

# EVOLVE-BLOCK-END
