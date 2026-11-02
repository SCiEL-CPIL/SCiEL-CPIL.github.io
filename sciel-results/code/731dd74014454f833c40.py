"""Pack 300 equal circles in the unit disk using lattice initialization + local refinement."""
import numpy as np
import time

N = 300
DEADLINE_FRACTION = 0.85  # safety margin before 120s (we are killed at 150s, but evaluator says 120s finish)
# Actually the contract says "Each evaluation must finish within 120 s". Let's use 115s to be safe.
TIME_LIMIT = 115.0


def _score(points):
    """Compute the packing radius r for given centers."""
    # min distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    r_boundary = np.min(dists_boundary)
    
    # min pairwise distance / 2
    # For N=300, 300x300 matrix is fine
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(r_boundary, min_pair)


def _repulsion_force(points, target_r, k=1.0, soft_min_alpha=10.0):
    """
    Compute forces pushing points apart and inward.
    We want to maximize r = min(min_pair/2, min_boundary).
    This is equivalent to maximizing the minimum of a set of values.
    
    A common heuristic is to use a soft-min approximation or simply gradient ascent on the sum of logs,
    or simply move points in the direction of the closest constraint.
    
    Here we use a simple gradient ascent approach based on the 'bottleneck' constraints.
    However, for robustness and speed, a common method is:
    1. Identify points that are violating or near the current optimal r.
    2. Move them away from their nearest neighbors and towards the center if too far out.
    
    Let's use a force-based method:
    Force on i:
    - If d_ij < 2*r_current + epsilon, push i away from j.
    - If |c_i| > 1 - r_current - epsilon, push i towards origin.
    
    But r_current changes. A better approach for this specific 'max-min' problem is often:
    Gradient of the objective f(x) = min_k g_k(x).
    The subgradient is the average of gradients of active constraints.
    
    Let's implement a simple iterative solver:
    1. Calculate current r.
    2. Identify active constraints (those where g_k(x) <= r + delta).
    3. Compute gradient of g_k for active k.
    4. Move x in direction of sum of gradients (scaled).
    """
    r = _score(points)
    # We want to increase r.
    # Let's just do a simple "move away from collisions and walls" step.
    
    forces = np.zeros_like(points)
    
    # Boundary forces
    dists_to_center = np.linalg.norm(points, axis=1)
    # If a point is close to the boundary, push it in.
    # Threshold: 1 - r
    boundary_margin = 1.0 - r
    for i in range(len(points)):
        d = dists_to_center[i]
        if d > boundary_margin - 0.01:
            # Push towards origin
            if d > 1e-6:
                forces[i] -= points[i] / d * (d - boundary_margin + 0.01)
    
    # Pairwise forces
    # Only compute for pairs closer than 2*r + threshold
    threshold = 2.0 * r + 0.05
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            diff = points[i] - points[j]
            dist = np.linalg.norm(diff)
            if dist < threshold:
                if dist < 1e-6:
                    # Random kick
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                else:
                    # Push apart
                    force_mag = (threshold - dist)
                    unit_vec = diff / dist
                    forces[i] += unit_vec * force_mag
                    forces[j] -= unit_vec * force_mag
    
    return forces


def _local_refine(points, max_iter=5000, step_size=0.005, time_limit=100.0):
    """
    Local refinement using gradient-like steps.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Adaptive step size
    current_step = step_size
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        forces = _repulsion_force(best_points, _score(best_points))
        
        # Normalize forces to avoid huge jumps
        norm_f = np.linalg.norm(forces)
        if norm_f > 1e-6:
            forces = forces / norm_f * 0.01 # Scale down significantly
        
        new_points = best_points + forces * current_step
        
        new_score = _score(new_points)
        
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            # Increase step slightly if we made progress
            current_step = min(current_step * 1.05, 0.05)
        else:
            # Decrease step if no progress
            current_step = max(current_step * 0.9, 0.0001)
            
            # If step is too small, try random perturbation to escape local minima
            if current_step < 0.0005 and it % 100 == 0:
                best_points = best_points + np.random.randn(*best_points.shape) * 0.001
                best_score = _score(best_points)
                current_step = 0.005
    
    return best_points, best_score


def _hexagonal_lattice_init(n_points):
    """
    Generate initial points using a hexagonal lattice.
    """
    points = []
    # Estimate spacing
    # Area of unit circle = pi. Area per point ~ pi/n.
    # Hex lattice area per point = s^2 * sqrt(3)/2.
    # s ~ sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    
    # Generate lattice points
    max_r = 1.0
    # Range of indices
    # x = i*s + j*s/2, y = j*s*sqrt(3)/2
    # We need to cover radius 1.
    # Max index approx 1/s
    k = int(1.0 / s) + 2
    
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - s/2:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more points, select the ones closest to center (or random)
    if len(points) > n_points:
        # Sort by distance to center to keep the "core"
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n_points]
        points = points[idx]
    elif len(points) < n_points:
        # This shouldn't happen with the estimation, but if it does, add random points
        while len(points) < n_points:
            angle = np.random.uniform(0, 2*np.pi)
            rad = np.sqrt(np.random.uniform(0, 1))
            points = np.vstack([points, [rad*np.cos(angle), rad*np.sin(angle)]])
            
    return points


def construct():
    start_time = time.time()
    
    # 1. Initialize
    points = _hexagonal_lattice_init(N)
    
    # 2. Local Refinement
    # We have ~110 seconds. Let's use 100s for refinement.
    time_for_refine = TIME_LIMIT - (time.time() - start_time) - 2.0 # 2s buffer
    if time_for_refine < 5:
        return points
        
    refined_points, score = _local_refine(points, time_limit=time_for_refine)
    
    return refined_points

# EVOLVE-BLOCK-END
