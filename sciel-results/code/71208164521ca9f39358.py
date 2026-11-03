import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _get_circumcenters_and_radii(points, hull):
    """
    Compute the circumcenters and circumradii for all facets of the convex hull.
    Returns:
        centers: array of shape (M, 3) with circumcenters on the unit sphere.
        radii: array of shape (M,) with angular circumradii.
        facets: array of shape (M, 3) with indices of vertices for each facet.
    """
    facets = hull.simplices
    M = len(facets)
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    for k in range(M):
        tri = facets[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            radii[k] = 0.0
            centers[k] = p1
            continue
            
        c = cross / norm_cross
        
        # Ensure the circumcenter is on the correct side (outward normal)
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        radii[k] = dist
        centers[k] = c
        
    return centers, radii, facets

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        centers, radii, _ = _get_circumcenters_and_radii(points, hull)
        return np.degrees(np.max(radii))
    except Exception:
        return 180.0

def _compute_facet_forces(points, centers, radii, facets, equalize_weight=1.0):
    """
    Compute forces on each point that drive toward equalizing circumradii.
    
    For each facet k with circumradius r_k and circumcenter c_k, the force on
    vertex p_i of that facet is:
    - A "balancing" force that pushes p_i toward the circumcenter if r_k is
      larger than average (to shrink large facets) and pulls away if r_k is
      smaller (to grow small facets).
    
    This is a structural change: instead of smooth-max gradient, we directly
    equalize facet sizes.
    """
    M = len(facets)
    forces = np.zeros_like(points)
    
    if M == 0:
        return forces
    
    mean_r = np.mean(radii)
    std_r = np.std(radii) + 1e-12
    
    for k in range(M):
        r_k = radii[k]
        c = centers[k]
        tri = facets[k]
        
        # Normalized deviation from mean
        deviation = (r_k - mean_r) / std_r
        
        # Force magnitude: proportional to deviation
        force_mag = equalize_weight * deviation
        
        for idx in tri:
            p = points[idx]
            
            # Direction: from p toward c (to shrink) or away (to grow)
            # We want to move p to reduce r_k if r_k > mean
            # Moving p toward c reduces the angular distance
            direction = c - p
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-12:
                continue
            direction /= dir_norm
            
            # Project to tangent plane at p
            direction_tangent = direction - np.dot(direction, p) * p
            dir_tangent_norm = np.linalg.norm(direction_tangent)
            if dir_tangent_norm < 1e-12:
                continue
            direction_tangent /= dir_tangent_norm
            
            forces[idx] += force_mag * direction_tangent
    
    return forces

def _compute_repulsive_forces(points, strength=0.1):
    """
    Compute pairwise repulsive forces to prevent points from getting too close.
    This helps maintain a well-distributed configuration.
    """
    forces = np.zeros_like(points)
    n = len(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = points[i] - points[j]
            dist = np.linalg.norm(d)
            if dist < 1e-8:
                continue
            # Repulsive force proportional to 1/dist^2
            force = strength / (dist * dist)
            d_unit = d / dist
            forces[i] += force * d_unit
            forces[j] -= force * d_unit
    
    # Project to tangent plane
    for i in range(n):
        p = points[i]
        forces[i] -= np.dot(forces[i], p) * p
    
    return forces

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Optimization parameters
    step_size = 0.01
    min_step_size = 1e-8
    equalize_weight = 0.5
    repulse_strength = 0.05
    stagnant_count = 0
    max_stagnant = 50
    polish_phase = False
    
    while time.time() < deadline:
        
        try:
            hull = ConvexHull(points)
            centers, radii, facets = _get_circumcenters_and_radii(points, hull)
        except Exception:
            # If hull fails, add noise and retry
            points += rng.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue

        current_radius_rad = np.max(radii)
        current_radius_deg = np.degrees(current_radius_rad)
        
        if current_radius_deg < best_radius - 1e-9:
            best_radius = current_radius_deg
            best_points = points.copy()
            stagnant_count = 0
        else:
            stagnant_count += 1
            
            if stagnant_count > max_stagnant:
                # Random kick to escape local minimum
                kick = rng.normal(0, 0.02, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.01
                stagnant_count = 0
                continue
        
        # Check if we should enter polish phase
        if best_radius < 11.35 and not polish_phase:
            polish_phase = True
            step_size = 0.003
            equalize_weight = 0.3
            repulse_strength = 0.02
        
        # Compute combined forces
        facet_forces = _compute_facet_forces(points, centers, radii, facets, equalize_weight)
        repulsive = _compute_repulsive_forces(points, repulse_strength)
        total_forces = facet_forces + repulsive
        
        # Normalize forces
        force_norms = np.linalg.norm(total_forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-12:
            total_forces = total_forces / max_force
        
        # Line search
        best_trial = None
        best_trial_radius = current_radius_deg
        
        for factor in [1.0, 0.5, 0.25, 0.1]:
            trial_step = step_size * factor
            trial_points = points + trial_step * total_forces
            trial_points /= np.linalg.norm(trial_points, axis=1, keepdims=True)
            
            if not np.all(np.isfinite(trial_points)):
                continue
            
            try:
                trial_hull = ConvexHull(trial_points)
                trial_centers, trial_radii, _ = _get_circumcenters_and_radii(trial_points, trial_hull)
                trial_radius = np.degrees(np.max(trial_radii))
            except Exception:
                continue
            
            if trial_radius < best_trial_radius:
                best_trial_radius = trial_radius
                best_trial = trial_points
        
        if best_trial is not None:
            points = best_trial
            stagnant_count = 0
            step_size = max(step_size * 0.995, min_step_size)
        else:
            step_size = max(step_size * 0.9, min_step_size)
            stagnant_count += 1
            
            if stagnant_count > max_stagnant:
                kick = rng.normal(0, 0.02, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                step_size = 0.005
                stagnant_count = 0
    
    # Final polish
    if time.time() < deadline - 5.0:
        for _ in range(100):
            if time.time() > deadline - 2.0:
                break
            try:
                hull = ConvexHull(points)
                centers, radii, facets = _get_circumcenters_and_radii(points, hull)
            except Exception:
                break
                
            facet_forces = _compute_facet_forces(points, centers, radii, facets, 0.1)
            repulsive = _compute_repulsive_forces(points, 0.01)
            total_forces = facet_forces + repulsive
            
            force_norms = np.linalg.norm(total_forces, axis=1)
            max_force = np.max(force_norms)
            if max_force < 1e-12:
                break
            total_forces = total_forces / max_force
            
            trial_points = points + 0.001 * total_forces
            trial_points /= np.linalg.norm(trial_points, axis=1, keepdims=True)
            
            if not np.all(np.isfinite(trial_points)):
                break
            
            try:
                trial_hull = ConvexHull(trial_points)
                trial_centers, trial_radii, _ = _get_circumcenters_and_radii(trial_points, trial_hull)
                trial_radius = np.degrees(np.max(trial_radii))
            except Exception:
                break
            
            if trial_radius < _compute_covering_radius(points) - 1e-10:
                points = trial_points
    
    return best_points
