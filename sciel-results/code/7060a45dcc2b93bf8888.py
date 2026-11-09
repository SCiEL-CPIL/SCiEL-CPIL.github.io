"""Concentric ring-based packing with SLSQP refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _compute_score(points):
    """Compute the packing score: min(min pairwise dist/2, min boundary margin)."""
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Boundary constraint
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary = 1.0 - radii
    if np.any(boundary < 0):
        return 0.0
    # Pairwise distances
    # Use a chunked approach to avoid memory issues
    min_dist = np.inf
    for i in range(N):
        diff = points[i+1:] - points[i]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        if dists.size > 0:
            min_dist = min(min_dist, np.min(dists))
    return min(min_dist / 2.0, np.min(boundary))


def _make_ring_initialization():
    """
    Create an initial configuration using concentric rings.
    Ring 0: 1 point at center
    Ring k: points evenly spaced on circle of radius r_k
    We need to choose the number of points per ring and the radii.
    
    For 300 points, a good distribution might be:
    - Center: 1
    - Then rings with increasing counts
    
    Let's use a scheme where ring k has approximately 6k points (hexagonal-like).
    But we need to tune the radii so that points don't overlap.
    
    Actually, let's think more carefully. If we have rings with n_k points at radius r_k,
    the angular spacing is 2*pi/n_k, and the chord distance between adjacent points is
    2*r_k*sin(pi/n_k). We need this to be > 2*r (the diameter of small circles).
    
    For an initial guess, let's use a simple scheme:
    - Start with 1 at center
    - Add rings with counts: 6, 12, 18, ... (multiples of 6)
    - Radii chosen so that points are roughly equally spaced
    """
    points = []
    # Center point
    points.append([0.0, 0.0])
    
    # Determine ring structure
    # We want roughly 300 points total
    # Let's use rings with counts that sum to 299 more points
    ring_counts = []
    remaining = N - 1
    k = 1
    while remaining > 0:
        # Use 6*k points for ring k, but cap to remaining
        count = min(6 * k, remaining)
        ring_counts.append(count)
        remaining -= count
        k += 1
    
    # Now assign radii. We want the points to be roughly uniformly distributed.
    # The area of a disk of radius R is pi*R^2. If we want N points uniformly,
    # the density is N/(pi*R^2). The radius of the k-th ring (with cumulative count C)
    # should be approximately sqrt(C/N) * R.
    
    # Let's compute cumulative counts
    cumulative = [1]
    for count in ring_counts:
        cumulative.append(cumulative[-1] + count)
    
    # Assign radii based on area fraction
    for i, count in enumerate(ring_counts):
        # Fraction of area up to this ring
        frac = cumulative[i+1] / N
        r = np.sqrt(frac) * 0.95  # Leave some margin
        # Place points evenly on this ring
        angles = np.linspace(0, 2*np.pi, count, endpoint=False) + np.pi/count  # Offset to avoid alignment
        for angle in angles:
            points.append([r * np.cos(angle), r * np.sin(angle)])
    
    points = np.array(points[:N])
    return points


def _refine(points, time_limit=100.0):
    """
    Refine the configuration using SLSQP optimization.
    We maximize the minimum of (pairwise distances / 2) and (boundary margins).
    
    To make this tractable, we use a surrogate: minimize the maximum of
    (1/d_ij - 1/(2r)) for pairs that are too close, and (|c_i| + r - 1) for boundary.
    
    Actually, a cleaner approach: we want to maximize r such that:
    - |c_i - c_j| >= 2r for all i < j
    - |c_i| + r <= 1 for all i
    
    We can use a sequential approach: start with the current configuration,
    compute the current r, then try to improve by moving points.
    
    For SLSQP, we'll maximize a smoothed version of the minimum constraint.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _compute_score(points)
    
    # Objective: minimize negative of the minimum constraint
    # We'll use a soft version: minimize sum of violations
    def objective(x):
        pts = x.reshape(-1, 2)
        # Boundary
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_violation = np.maximum(0, radii - 1.0)
        # Pairwise - use only nearby pairs for speed
        # For now, use all pairs but this is O(N^2)
        # Let's use a chunked approach
        min_dist = np.inf
        for i in range(N):
            diff = pts[i+1:] - pts[i]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            if dists.size > 0:
                min_dist = min(min_dist, np.min(dists))
        # Soft penalty for small distances
        # Target distance: we want min_dist to be as large as possible
        # Use: -min_dist + penalty for boundary
        return -min_dist + 10.0 * np.sum(boundary_violation**2)
    
    # Constraints for SLSQP
    # We'll use inequality constraints: |c_i| <= 1 - epsilon
    # And pairwise: |c_i - c_j| >= 2*epsilon (but this is too many constraints)
    
    # Let's use a simpler approach: just minimize the objective with bounds
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Try a few iterations with different initial perturbations
    for attempt in range(3):
        if time.time() - start_time > time_limit:
            break
        
        # Add small random perturbation for diversity
        if attempt == 0:
            x0 = points.flatten()
        else:
            x0 = points.flatten() + 0.01 * np.random.randn(2 * N)
            # Project back to feasible region approximately
            for i in range(N):
                r = np.hypot(x0[2*i], x0[2*i+1])
                if r > 0.95:
                    scale = 0.95 / r
                    x0[2*i] *= scale
                    x0[2*i+1] *= scale
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < objective(x0):
                new_points = result.x.reshape(-1, 2)
                # Check validity
                if np.all(np.isfinite(new_points)):
                    score = _compute_score(new_points)
                    if score > best_score:
                        best_score = score
                        best_points = new_points.copy()
        except Exception:
            pass
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Generate initial configuration
    points = _make_ring_initialization()
    
    # Initial score
    score = _compute_score(points)
    
    # Refine with time budget
    remaining_time = deadline - time.time() - 5.0  # Leave 5s for final processing
    if remaining_time > 10.0:
        points, score = _refine(points, time_limit=remaining_time)
    
    # Final validation
    if not np.all(np.isfinite(points)) or points.shape != (N, 2):
        # Fallback to simple grid
        points = _make_ring_initialization()
    
    return points
