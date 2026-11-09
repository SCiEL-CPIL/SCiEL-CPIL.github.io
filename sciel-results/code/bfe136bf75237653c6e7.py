# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _min_triangle_area_and_indices(pts):
    """
    Calculate the minimum triangle area and the indices of the vertices forming it.
    """
    n = len(pts)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            xj, yj = pts[j]
            for k in range(j + 1, n):
                xk, yk = pts[k]
                area = 0.5 * abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if area < min_area:
                    min_area = area
                    min_idx = (i, j, k)
    return min_area, min_idx

def _hull_area(pts):
    """
    Calculate the area of the convex hull.
    """
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """
    Calculate the score: A_min / A_hull.
    """
    a_min, _ = _min_triangle_area_and_indices(pts)
    a_hull = _hull_area(pts)
    if a_hull == 0:
        return 0.0
    return a_min / a_hull

def _triangular_lattice_subset(n=13):
    """
    Generate a subset of the triangular lattice that approximates a near-optimal
    configuration for the Heilbronn problem.
    
    The triangular lattice has points at (i + j/2, j*sqrt(3)/2) for integers i,j.
    We select n points from this lattice that form a reasonably balanced set.
    """
    # Generate a grid of triangular lattice points
    points = []
    for j in range(-3, 4):
        for i in range(-3, 4):
            x = i + j / 2.0
            y = j * np.sqrt(3) / 2.0
            points.append((x, y))
    
    points = np.array(points)
    
    # Select n points that are well-distributed
    # Use a greedy approach: start with the centroid, then add farthest points
    centroid = points.mean(axis=0)
    selected = [np.argmin(np.sum((points - centroid)**2, axis=1))]
    
    while len(selected) < n:
        # For each unselected point, compute minimum distance to selected points
        min_dists = []
        for idx in range(len(points)):
            if idx in selected:
                continue
            min_dist = min(np.linalg.norm(points[idx] - points[s]) for s in selected)
            min_dists.append((min_dist, idx))
        
        # Select the point with the largest minimum distance
        min_dists.sort(reverse=True)
        selected.append(min_dists[0][1])
    
    return points[selected]

def _radial_angular_params(pts, center=None):
    """
    Convert points to radial-angular parameters relative to a center.
    Returns (r, theta) arrays where r is distance from center and theta is angle.
    """
    if center is None:
        center = pts.mean(axis=0)
    
    dx = pts[:, 0] - center[0]
    dy = pts[:, 1] - center[1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    
    return r, theta, center

def _from_radial_angular(r, theta, center):
    """
    Convert radial-angular parameters back to Cartesian coordinates.
    """
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y])

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest
    triangle area to the convex hull area.
    
    Uses a low-dimensional parameterization based on radial/angular deviations
    from a triangular lattice subset, combined with simulated annealing.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    # Start with triangular lattice subset
    base_pts = _triangular_lattice_subset(n)
    
    # Normalize to unit scale
    center = base_pts.mean(axis=0)
    base_pts = base_pts - center
    max_r = np.max(np.linalg.norm(base_pts, axis=1))
    if max_r > 0:
        base_pts /= max_r
    
    # Extract initial radial-angular parameters
    r0, theta0, _ = _radial_angular_params(base_pts)
    
    best_pts = base_pts.copy()
    best_score = _score(base_pts)
    
    # Current parameters
    r = r0.copy()
    theta = theta0.copy()
    current_center = np.array([0.0, 0.0])
    current_scale = 1.0
    
    current_pts = base_pts.copy()
    current_score = best_score
    
    # Simulated annealing parameters
    temp = 0.05
    cooling = 0.9995
    min_temp = 1e-5
    no_improvement_count = 0
    max_no_improvement = 500
    
    # Step sizes for different parameter types
    r_step = 0.02
    theta_step = 0.05
    center_step = 0.01
    scale_step = 0.01
    
    while time.time() < deadline:
        # Generate a random perturbation of the parameters
        new_r = r.copy()
        new_theta = theta.copy()
        new_center = current_center.copy()
        new_scale = current_scale
        
        # Randomly perturb a subset of points' radial and angular parameters
        num_perturbed = np.random.randint(1, 4)
        indices = np.random.choice(n, size=num_perturbed, replace=False)
        
        for idx in indices:
            new_r[idx] += np.random.normal(0, r_step)
            new_r[idx] = max(0.1, new_r[idx])  # Keep positive
            new_theta[idx] += np.random.normal(0, theta_step)
        
        # Occasionally perturb center and scale
        if np.random.random() < 0.1:
            new_center += np.random.normal(0, center_step, size=2)
        if np.random.random() < 0.05:
            new_scale += np.random.normal(0, scale_step)
            new_scale = max(0.5, min(2.0, new_scale))
        
        # Convert to Cartesian coordinates
        new_pts = _from_radial_angular(new_r, new_theta, new_center)
        new_pts *= new_scale
        
        # Evaluate
        new_score = _score(new_pts)
        
        # Acceptance logic
        accepted = False
        if new_score > current_score:
            r = new_r
            theta = new_theta
            current_center = new_center
            current_scale = new_scale
            current_pts = new_pts
            current_score = new_score
            accepted = True
        else:
            delta = new_score - current_score
            if np.random.random() < np.exp(delta / temp):
                r = new_r
                theta = new_theta
                current_center = new_center
                current_scale = new_scale
                current_pts = new_pts
                current_score = new_score
                accepted = True
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Occasionally inject larger random noise to escape local minima
        if no_improvement_count > max_no_improvement or np.random.random() < 0.005:
            # Reset to a new random lattice-based configuration
            new_base = _triangular_lattice_subset(n)
            center_new = new_base.mean(axis=0)
            new_base = new_base - center_new
            max_r_new = np.max(np.linalg.norm(new_base, axis=1))
            if max_r_new > 0:
                new_base /= max_r_new
            
            new_r0, new_theta0, _ = _radial_angular_params(new_base)
            r = new_r0 + np.random.normal(0, 0.05, n)
            r = np.maximum(0.1, r)
            theta = new_theta0 + np.random.normal(0, 0.1, n)
            current_center = np.random.normal(0, 0.05, 2)
            current_scale = np.random.uniform(0.8, 1.2)
            current_pts = _from_radial_angular(r, theta, current_center) * current_scale
            current_score = _score(current_pts)
            no_improvement_count = 0
            
            # If score dropped significantly, revert to best
            if current_score < best_score * 0.5:
                r = r0.copy()
                theta = theta0.copy()
                current_center = np.array([0.0, 0.0])
                current_scale = 1.0
                current_pts = best_pts.copy()
                current_score = best_score
        
        # Cooling
        temp *= cooling
        if temp < min_temp:
            temp = 0.02  # Reheat slightly
            # Adjust step sizes
            r_step = np.random.uniform(0.01, 0.03)
            theta_step = np.random.uniform(0.03, 0.07)
            center_step = np.random.uniform(0.005, 0.02)
            scale_step = np.random.uniform(0.005, 0.015)

    return best_pts
# EVOLVE-BLOCK-END
