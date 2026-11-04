"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1.0


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area


def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _compute_forces(points, alpha):
    """
    Compute the gradient of the potential energy E = sum(A_ijk^(-alpha)).
    Force on point k is -grad_k E.
    grad_k E = sum_{i<j, i,j != k} -alpha * A_ijk^(-alpha-1) * 0.5 * (p_j - p_i)^perp
    Force = sum_{i<j, i,j != k} alpha * A_ijk^(-alpha-1) * 0.5 * (p_j - p_i)^perp
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    # Precompute areas and vectors for efficiency? 
    # N=14 is small, O(N^3) is fine.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p_i = points[i]
                p_j = points[j]
                p_k = points[k]
                
                # Area
                cross = (p_j[0] - p_i[0]) * (p_k[1] - p_i[1]) - (p_j[1] - p_i[1]) * (p_k[0] - p_i[0])
                area = 0.5 * abs(cross)
                
                if area < 1e-12:
                    # Avoid division by zero
                    area = 1e-12
                
                # Term: alpha * area^(-alpha-1) * 0.5
                coeff = alpha * (area ** (-alpha - 1)) * 0.5
                
                # Vector (p_j - p_i)
                v_ji = p_j - p_i
                # Perpendicular vector (dx, dy) -> (-dy, dx)
                # Note: The sign depends on orientation, but since we use abs(area) in the potential,
                # we need to be careful. 
                # d/dx (0.5 * |cross|) is tricky if cross changes sign.
                # However, for a convex configuration, all points are in general position.
                # The gradient of Area w.r.t p_k is 0.5 * (p_j - p_i)^perp.
                # If we use abs(Area), the gradient direction flips if Area becomes negative?
                # Area is always positive in the formula 0.5*|cross|.
                # The geometric area is positive. The vector derivative of the *oriented* area is 0.5*(p_j-p_i)^perp.
                # If we maximize the minimum *absolute* area, we should push in the direction that increases the absolute area.
                # If the points are in CCW order, cross > 0, grad is 0.5*(p_j-p_i)^perp.
                # If CW, cross < 0, abs(cross) = -cross. d/dx(-cross) = -0.5*(p_j-p_i)^perp.
                # So the force direction should be sign(cross) * 0.5 * (p_j-p_i)^perp.
                
                sign = 1.0 if cross > 0 else -1.0
                perp_ji = np.array([-v_ji[1], v_ji[0]])
                
                force_vec = coeff * sign * perp_ji
                
                # Force on k is in direction of perp_ji
                forces[k] += force_vec
                # Force on i and j?
                # d/dp_i Area = 0.5 * (p_k - p_j)^perp
                # d/dp_j Area = 0.5 * (p_i - p_k)^perp
                # Note: (p_k - p_j) = -(p_j - p_k). 
                # Let's just add forces to i and j as well for completeness, though 
                # summing over all triples and adding to k, i, j is equivalent.
                
                v_kj = p_k - p_j
                perp_kj = np.array([-v_kj[1], v_kj[0]])
                forces[i] += coeff * sign * perp_kj
                
                v_ik = p_i - p_k
                perp_ik = np.array([-v_ik[1], v_ik[0]])
                forces[j] += coeff * sign * perp_ik

    return forces


def _normalize_points(points):
    """Center and scale points to unit max radius."""
    center = np.mean(points, axis=0)
    points = points - center
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 1e-10:
        points = points / max_dist
    return points


def construct():
    """
    Place 14 points in a planar convex region to maximize the minimum triangle area
    divided by the convex hull area.
    
    Strategy: Gradient descent on a potential energy function that penalizes small triangle areas.
    E = sum(A_ijk^(-alpha)). Minimizing E increases the minimum area.
    """
    rng = np.random.RandomState(42)
    best_points = None
    best_score = -1.0
    
    # Try different alpha values and restarts
    alphas = [4, 6, 8, 12]
    
    for alpha in alphas:
        for restart in range(3):
            # Initialize
            if restart == 0:
                angles = 2 * np.pi * np.arange(N) / N
                points = np.column_stack([np.cos(angles), np.sin(angles)])
            else:
                # Random perturbation of regular polygon
                angles = 2 * np.pi * np.arange(N) / N
                points = np.column_stack([np.cos(angles), np.sin(angles)])
                points += 0.1 * rng.randn(N, 2)
            
            points = _normalize_points(points)
            
            # Gradient Descent with Simulated Annealing style step decay
            # We want to move in the direction of the force (which increases area)
            # So we do points = points + step * forces
            
            step = 0.01
            for iteration in range(1000):
                forces = _compute_forces(points, alpha)
                
                # Normalize forces to prevent huge steps if areas are tiny
                force_norm = np.linalg.norm(forces)
                if force_norm > 1e-10:
                    forces = forces / force_norm * 0.1 # Scale down for stability
                
                # Update
                new_points = points + step * forces
                
                # Normalize to keep on unit scale
                new_points = _normalize_points(new_points)
                
                # Optional: Acceptance probability (Simulated Annealing)
                # For simplicity, we always accept if it improves the "potential" or just take the step.
                # Since forces are derived from gradient of E, moving +force decreases E (increases Area).
                # So we should always accept.
                
                points = new_points
                
                # Decay step size
                step *= 0.995
                
                # If step becomes too small, break
                if step < 1e-6:
                    break
            
            current_score = _score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                
    return best_points
# EVOLVE-BLOCK-END
