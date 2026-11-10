"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _cover_radius_fast(pts, n_samples=4000):
    """
    Fast covering radius estimate via dense sampling on the sphere.
    Returns the maximum angular distance from any sample point to its nearest config point.
    """
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # dots[i, j] = dot(samples[i], pts[j])
    dots = samples @ pts.T  # (n_samples, N)
    min_dots = np.min(dots, axis=1)  # (n_samples,)
    min_dots = np.clip(min_dots, -1.0, 1.0)
    angles = np.arccos(min_dots)
    return float(np.max(angles))


def _cover_radius_hull(pts):
    """
    Compute the exact covering radius using the convex hull facets.
    For each triangular facet, compute the circumcenter and the angular distance
    to the nearest vertex. The covering radius is the maximum over all facets.
    """
    if len(pts) < 3:
        return np.pi
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _cover_radius_fast(pts, n_samples=8000)
    
    max_angle = 0.0
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute circumcenter using weighted average
        cross_bc = np.cross(b, c)
        cross_ca = np.cross(c, a)
        cross_ab = np.cross(a, b)
        
        norm_bc = np.linalg.norm(cross_bc)
        norm_ca = np.linalg.norm(cross_ca)
        norm_ab = np.linalg.norm(cross_ab)
        
        total = norm_bc + norm_ca + norm_ab
        if total < 1e-15:
            continue
        
        circum_dir = (a * norm_bc + b * norm_ca + c * norm_ab) / total
        circum_norm = np.linalg.norm(circum_dir)
        if circum_norm < 1e-15:
            continue
        circum_dir = circum_dir / circum_norm
        
        # Compute the minimum dot product from circumcenter to the three vertices
        dot_a = np.dot(circum_dir, a)
        dot_b = np.dot(circum_dir, b)
        dot_c = np.dot(circum_dir, c)
        
        min_dot = min(dot_a, dot_b, dot_c)
        min_dot = np.clip(min_dot, -1.0, 1.0)
        angle = np.arccos(min_dot)
        
        if angle > max_angle:
            max_angle = angle
    
    return max_angle


def _force_based_optimize(pts, max_iters=10000, initial_step=0.01, time_budget=30.0):
    """
    Force-based optimization: for each point, compute the net force from
    nearby points. Points that are too close together repel; points that
    leave large gaps attract.
    """
    pts = pts.copy()
    step = initial_step
    best_radius = _cover_radius_fast(pts, n_samples=3000)
    no_improve_count = 0
    start_time = time.time()
    
    for it in range(max_iters):
        # Time check
        if time.time() - start_time > time_budget:
            break
        
        # Compute forces based on local density
        forces = np.zeros_like(pts)
        
        # Compute all pairwise dot products
        dots = pts @ pts.T  # (N, N)
        np.fill_diagonal(dots, 1.0)
        
        # Target spacing: for 50 points, ideal angular spacing is about
        # sqrt(4*pi/50) ≈ 0.502 radians ≈ 28.7 degrees
        target_dot = np.cos(np.sqrt(4 * np.pi / N))
        
        # Vectorized force computation
        # Repulsion: if d > target_dot, push i away from j
        diff = target_dot - dots  # positive when too far, negative when too close
        # We want repulsion when d > target_dot, i.e., diff < 0
        repel_mask = diff < 0
        repel_mag = np.where(repel_mask, (-diff) ** 2, 0.0)
        
        # For each point i, force = sum_j repel_mag[i,j] * (pts[i] - pts[j])
        # This can be computed as: pts[i] * sum_j repel_mag[i,j] - sum_j repel_mag[i,j] * pts[j]
        row_sums = np.sum(repel_mag, axis=1)  # (N,)
        forces = pts * row_sums[:, np.newaxis] - repel_mag @ pts
        
        # Add gentle force toward antipode of farthest point to prevent clustering
        far_idx = np.argmin(dots, axis=1)
        antipodes = -pts[far_idx]
        forces += 0.005 * antipodes
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-15, 1.0, force_norms)
        forces = forces / force_norms
        
        # Apply forces
        new_pts = pts + step * forces
        # Normalize back to unit sphere
        new_pts_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts_norms = np.where(new_pts_norms < 1e-15, 1.0, new_pts_norms)
        new_pts = new_pts / new_pts_norms
        
        # Evaluate
        new_radius = _cover_radius_fast(new_pts, n_samples=3000)
        
        if new_radius < best_radius - 1e-12:
            pts = new_pts
            best_radius = new_radius
            no_improve_count = 0
            # Slightly increase step if we're making good progress
            if it % 50 == 0 and step < initial_step:
                step = min(step * 1.02, initial_step)
        else:
            no_improve_count += 1
            step *= 0.97
            if step < 1e-8:
                break
        
        if no_improve_count > 80:
            break
    
    return pts, best_radius


def _fibonacci_spiral(N):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _hill_climbing(pts, max_iters=3000, initial_step=0.005, time_budget=20.0):
    """
    Hill climbing: try small random perturbations and keep the best.
    """
    pts = pts.copy()
    best_radius = _cover_radius_fast(pts, n_samples=3000)
    step = initial_step
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random subset of points to perturb
        n_perturb = rng.randint(1, 4)
        indices = rng.choice(N, n_perturb, replace=False)
        
        old_pts = pts[indices].copy()
        
        # Apply random perturbations
        perturbations = rng.randn(n_perturb, 3) * step
        new_pts = pts.copy()
        for i, idx in enumerate(indices):
            new_pts[idx] = old_pts[i] + perturbations[i]
            norm = np.linalg.norm(new_pts[idx])
            if norm > 1e-15:
                new_pts[idx] /= norm
        
        new_radius = _cover_radius_fast(new_pts, n_samples=3000)
        
        if new_radius < best_radius - 1e-12:
            pts = new_pts
            best_radius = new_radius
            # Occasionally increase step
            if it % 20 == 0 and step < initial_step:
                step = min(step * 1.05, initial_step)
        else:
            step *= 0.98
            if step < 1e-9:
                break
    
    return pts, best_radius


def _gradient_descent(pts, max_iters=5000, lr=0.002, time_budget=20.0):
    """
    Gradient descent on the covering radius using sampling.
    For each point, estimate the gradient of the max-min distance and move opposite.
    """
    pts = pts.copy()
    start_time = time.time()
    best_radius = _cover_radius_fast(pts, n_samples=3000)
    n_samples = 2000
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # dots[i, j] = dot(samples[i], pts[j])
        dots = samples @ pts.T  # (n_samples, N)
        min_dots, min_idx = np.min(dots, axis=1)  # (n_samples,), (n_samples,)
        min_dots = np.clip(min_dots, -1.0, 1.0)
        angles = np.arccos(min_dots)
        current_radius = float(np.max(angles))
        
        if current_radius < best_radius:
            best_radius = current_radius
        
        # Find the worst sample (largest angle)
        worst_idx = np.argmax(angles)
        worst_sample = samples[worst_idx]
        nearest_pt_idx = min_idx[worst_idx]
        
        # Gradient: move the nearest point toward the worst sample
        # The gradient of -min_dot w.r.t. pts[nearest_pt_idx] is -worst_sample
        # So we move pts[nearest_pt_idx] in the direction of worst_sample
        grad = worst_sample.copy()
        
        # Also consider second-worst to avoid oscillation
        angles_sorted = np.argsort(angles)[::-1]
        for k in range(min(5, n_samples)):
            idx = angles_sorted[k]
            if angles[idx] < angles[worst_idx] - 0.01:
                break
            ni = min_idx[idx]
            grad += samples[idx] * (angles[idx] / angles[worst_idx]) if angles[worst_idx] > 1e-10 else 0
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        grad /= grad_norm
        
        # Move the nearest point
        new_pts = pts.copy()
        new_pts[nearest_pt_idx] = pts[nearest_pt_idx] + lr * grad
        norm = np.linalg.norm(new_pts[nearest_pt_idx])
        if norm > 1e-15:
            new_pts[nearest_pt_idx] /= norm
        
        # Evaluate
        new_radius = _cover_radius_fast(new_pts, n_samples=3000)
        
        if new_radius < best_radius - 1e-12:
            pts = new_pts
            best_radius = new_radius
        else:
            lr *= 0.95
            if lr < 1e-6:
                break
    
    return pts, best_radius


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_radius = _cover_radius_fast(best_pts, n_samples=5000)
    
    # Phase 1: Force-based optimization from Fibonacci start
    remaining = deadline - 10 - time.time()
    if remaining > 15:
        budget = min(45.0, remaining * 0.4)
        optimized, opt_radius = _force_based_optimize(pts, max_iters=15000, initial_step=0.01, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 2: Gradient descent refinement
    remaining = deadline - 10 - time.time()
    if remaining > 10:
        budget = min(25.0, remaining * 0.4)
        optimized, opt_radius = _gradient_descent(best_pts, max_iters=5000, lr=0.003, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 3: Hill climbing refinement
    remaining = deadline - 10 - time.time()
    if remaining > 8:
        budget = min(20.0, remaining * 0.4)
        optimized, opt_radius = _hill_climbing(best_pts, max_iters=3000, initial_step=0.003, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Phase 4: Multiple random restarts with perturbed Fibonacci
    rng = np.random.RandomState(42)
    n_restarts = 8
    for restart in range(n_restarts):
        remaining = deadline - 8 - time.time()
        if remaining < 12:
            break
        
        # Generate perturbed Fibonacci
        base = _fibonacci_spiral(N)
        noise_scale = 0.05 + 0.15 * restart / n_restarts
        noise = rng.randn(N, 3) * noise_scale
        random_pts = base + noise
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        # Quick force-based optimization
        budget = min(15.0, remaining * 0.35)
        optimized, opt_radius = _force_based_optimize(random_pts, max_iters=5000, initial_step=0.008, time_budget=budget)
        
        # Quick hill climbing
        remaining = deadline - 8 - time.time()
        if remaining > 5 and opt_radius < best_radius:
            budget2 = min(8.0, remaining * 0.4)
            optimized, opt_radius = _hill_climbing(optimized, max_iters=1500, initial_step=0.002, time_budget=budget2)
        
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Final refinement with hill climbing
    remaining = deadline - 5 - time.time()
    if remaining > 5:
        budget = min(15.0, remaining * 0.8)
        optimized, opt_radius = _hill_climbing(best_pts, max_iters=2000, initial_step=0.002, time_budget=budget)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Final check: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_spiral(N)
    
    # Normalize to ensure unit sphere
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Validate with convex hull to ensure no degenerate configurations
    try:
        hull = ConvexHull(best_pts)
        # Check that the configuration is not in a closed hemisphere
        # by verifying the convex hull has positive volume
        if hull.volume < 1e-6:
            best_pts = _fibonacci_spiral(N)
            best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    except Exception:
        best_pts = _fibonacci_spiral(N)
        best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
