"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 10.0

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    # There are C(13,3) = 286 triangles.
    # We compute all cross products vectorized.
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i = i[mask]
    j = j[mask]
    k = k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _neg_score(pts_flat):
    """Objective function for minimization: -score."""
    pts = pts_flat.reshape(N, 2)
    # Prevent points from collapsing or going to infinity
    if np.any(~np.isfinite(pts)):
        return 1e10
    return -_score(pts)

def construct():
    entry_time = time.time()
    deadline = entry_time + (DEADLINE - SAFETY_MARGIN)
    
    rng = np.random.default_rng(42)
    best_score = 0.0
    best_points = None
    
    # Strategy: Physics-based repulsion + Gradient Descent
    # We model points as particles repelling each other.
    # The "potential energy" is a combination of:
    # 1. Pairwise repulsion (inverse square or similar) to keep points apart.
    # 2. A penalty for small triangles to directly maximize the min area.
    # 3. A constraint/penalty to keep points within a convex region (or just let them move and normalize).
    
    # Since the score is scale-invariant (min_area / hull_area), we can fix the scale
    # by normalizing the points to have unit area hull or fixed max norm during optimization,
    # OR we can optimize in a fixed domain and let the score handle the scaling.
    # However, the physics repulsion naturally expands the set. 
    # It is better to optimize the positions directly and use the normalized score.
    
    # Let's use a multi-start gradient descent approach.
    # Initial configurations:
    # 1. Regular 13-gon
    # 2. Random points in unit circle
    # 3. Symmetric structures (hexagon + center, etc.)
    
    initial_candidates = []
    
    # 1. Regular 13-gon
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    poly13 = np.column_stack([np.cos(theta), np.sin(theta)])
    initial_candidates.append(poly13)
    
    # 2. Center + 12 points on circle
    center = np.array([[0.0, 0.0]])
    theta12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ring12 = np.column_stack([np.cos(theta12), np.sin(theta12)])
    initial_candidates.append(np.vstack([center, ring12]))
    
    # 3. Two rings: 6 + 6 + 1 center
    center = np.array([[0.0, 0.0]])
    theta6a = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring6a = 0.5 * np.column_stack([np.cos(theta6a), np.sin(theta6a)])
    theta6b = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    ring6b = 1.0 * np.column_stack([np.cos(theta6b), np.sin(theta6b)])
    initial_candidates.append(np.vstack([center, ring6a, ring6b]))
    
    # 4. Random points
    for _ in range(10):
        pts = rng.random((N, 2)) * 2 - 1
        initial_candidates.append(pts)
        
    # Normalize initial candidates to have max norm 1
    def normalize(pts):
        pts = pts - pts.mean(axis=0)
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
        return pts
        
    for pts in initial_candidates:
        pts = normalize(pts)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            
    # Optimization Loop
    # We will run gradient descent (L-BFGS-B) from multiple starting points.
    # To handle the non-differentiability of min() and hull area, 
    # L-BFGS-B uses numerical gradients or quasi-Newton approximations which can be noisy.
    # A more robust approach for this specific "max-min" problem is often 
    # a combination of repulsion forces and direct optimization of the min triangle area.
    
    # Alternative: Use a custom gradient descent with subgradients or just rely on 
    # the robustness of L-BFGS-B on the smooth approximation.
    # Actually, the score function is piecewise smooth. 
    # Let's try a "soft" version of the min to make it differentiable, 
    # or just use Nelder-Mead which doesn't require gradients and is robust to non-smoothness.
    # Given the previous code used Nelder-Mead, let's stick to a robust local search 
    # but with a better initialization strategy based on physics.
    
    # Physics-based initialization:
    # Simulate repulsion for a few steps to get a "spread out" configuration.
    def physics_init(rng):
        pts = rng.random((N, 2)) * 2 - 1
        # Repulsion steps
        for step in range(50):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    diff = pts[i] - pts[j]
                    dist_sq = np.dot(diff, diff)
                    if dist_sq < 1e-6:
                        # Add random jitter
                        forces[i] += rng.normal(0, 0.01, 2)
                    else:
                        # Inverse square repulsion
                        force_mag = 1.0 / (dist_sq * np.sqrt(dist_sq)) # 1/r^3 for direction, 1/r^2 for mag?
                        # Standard Coulomb: F = k * q1 * q2 / r^2. Direction is diff/r.
                        # So F_vec = k * diff / r^3
                        forces[i] += diff / (dist_sq * np.sqrt(dist_sq))
            pts += 0.1 * forces
            # Keep in bounds
            pts = np.clip(pts, -1.5, 1.5)
        return normalize(pts)
        
    # Generate a few physics-initialized candidates
    for _ in range(5):
        if time.time() > deadline: break
        pts = physics_init(rng)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            
    # Now, refine the best candidate using Nelder-Mead.
    # We will do this from the best point and a few perturbed versions.
    
    def neg_score_wrapper(pts_flat):
        pts = pts_flat.reshape(N, 2)
        # Add a small penalty if points are too close to prevent numerical issues
        # or if they go out of bounds (though we normalize, NM might wander)
        return -_score(pts)
        
    x0 = best_points.flatten()
    
    # Run Nelder-Mead
    # We allow it to run for a while.
    # To ensure we beat the seed, we need to be careful with the optimization landscape.
    # The seed score is ~39.3.
    
    # Let's try a few restarts of NM with different initial step sizes.
    for restart in range(5):
        if time.time() > deadline - 20: break
        
        # Perturb the best point
        if restart == 0:
            x_start = x0
        else:
            x_start = x0 + rng.normal(0, 0.05, (2*N,))
            
        try:
            res = minimize(
                neg_score_wrapper, 
                x_start, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.success or res.fun < 0: # If it found a valid improvement
                pts_opt = res.x.reshape(N, 2)
                s_opt = -res.fun
                if s_opt > best_score:
                    best_score = s_opt
                    best_points = pts_opt.copy()
        except Exception:
            pass
            
    # Final normalization to ensure it's well-behaved
    if best_points is not None:
        best_points = normalize(best_points)
        
    return best_points

# EVOLVE-BLOCK-END
