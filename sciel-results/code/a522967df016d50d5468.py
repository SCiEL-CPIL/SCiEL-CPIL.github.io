"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
# Vertices of the equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3 / 2.0]])
# Area of this triangle
TRI_AREA = S3 / 4.0


def _project_to_triangle(p):
    """Project a point onto the interior of the equilateral triangle."""
    # Triangle vertices
    A, B, C = V[0], V[1], V[2]
    
    # Check if point is inside the triangle
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(A, B, p)
    d2 = cross(B, C, p)
    d3 = cross(C, A, p)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    if not (has_neg and has_pos):
        # Point is inside (or on boundary)
        return p
    
    # Point is outside; project to nearest edge
    def dist_to_segment(p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest), closest
    
    d1_val, c1 = dist_to_segment(p, A, B)
    d2_val, c2 = dist_to_segment(p, B, C)
    d3_val, c3 = dist_to_segment(p, C, A)
    
    dists = [d1_val, d2_val, d3_val]
    closests = [c1, c2, c3]
    idx = np.argmin(dists)
    
    # Move slightly inward to avoid boundary issues
    centroid = (A + B + C) / 3.0
    projected = closests[idx]
    # Nudge toward interior
    nudge = 1e-8
    projected = (1 - nudge) * projected + nudge * centroid
    
    return projected


def _all_triangle_areas(points):
    """Compute areas of all triangles formed by triples of points. Returns array of shape (C(N,3),)."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                areas.append(area)
    return np.array(areas)


def _min_triangle_area(points):
    """Minimum area among all triangles formed by triples of points."""
    areas = _all_triangle_areas(points)
    return np.min(areas)


def _objective(x):
    """Objective function: negative of the minimum triangle area (to minimize)."""
    points = x.reshape(-1, 2)
    # Project points to ensure they're inside the triangle
    points = np.array([_project_to_triangle(p) for p in points])
    return -_min_triangle_area(points)


def _objective_lse(x, temperature=0.05):
    """Smooth surrogate: -log-sum-exp of (areas / temperature) scaled, to get gradient info."""
    points = x.reshape(-1, 2)
    points = np.array([_project_to_triangle(p) for p in points])
    areas = _all_triangle_areas(points)
    # Use a soft-min approximation
    scaled = areas / temperature
    # log-sum-exp for numerical stability
    max_val = np.max(scaled)
    lse = max_val + np.log(np.sum(np.exp(scaled - max_val)))
    return -temperature * lse


def _objective_gradient(x, temperature=0.05):
    """Numerical gradient of the LSE surrogate."""
    eps = 1e-6
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (_objective_lse(x_plus, temperature) - _objective_lse(x_minus, temperature)) / (2 * eps)
    return grad


def _refine_nelder_mead(x0, max_iter=2000, timeout=None, start_time=None):
    """Refine using Nelder-Mead on the exact objective."""
    if start_time is None:
        start_time = time.time()
    
    def check_time():
        if timeout is not None and time.time() - start_time > timeout:
            return True
        return False
    
    result = minimize(
        _objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
    )
    
    if check_time():
        return result.x
    
    return result.x


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    best_x = None
    best_val = -np.inf
    
    # Try multiple random initializations
    np.random.seed(42)
    
    # Generate initial points
    def random_points_in_triangle():
        """Generate N random points inside the triangle."""
        points = []
        for _ in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1.0:
                u = 1.0 - u
                v = 1.0 - v
            x = u + 0.5 * v
            y = (S3 / 2.0) * v
            points.append([x, y])
        return np.array(points)
    
    # Try several random starts
    n_starts = 15
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Random initialization
        points = random_points_in_triangle()
        x0 = points.flatten()
        
        # First, use trust-region with LSE surrogate for a few iterations
        # Then switch to Nelder-Mead on the exact objective
        
        # Phase 1: Gradient-based with LSE surrogate (a few iterations to get in the basin)
        try:
            # Use a moderate temperature for smoothing
            temp = 0.02
            
            # Do a few iterations of gradient descent with adaptive step
            x_curr = x0.copy()
            for it in range(50):
                if time.time() > deadline:
                    break
                grad = _objective_gradient(x_curr, temp)
                # Adaptive step size
                step_size = 0.01
                x_new = x_curr - step_size * grad
                f_curr = _objective(x_curr)
                f_new = _objective(x_new)
                if f_new < f_curr:
                    x_curr = x_new
                else:
                    step_size *= 0.5
                    if step_size < 1e-8:
                        break
        except Exception:
            pass
        
        # Phase 2: Nelder-Mead refinement on exact objective
        remaining_time = deadline - time.time()
        if remaining_time < 5.0:
            # Not enough time, just evaluate current
            val = -_objective(x_curr)
            if val > best_val:
                best_val = val
                best_x = x_curr.copy()
            continue
        
        try:
            x_refined = _refine_nelder_mead(
                x_curr,
                max_iter=1000,
                timeout=min(remaining_time - 2.0, 20.0),
                start_time=start_time
            )
        except Exception:
            x_refined = x_curr
        
        # Evaluate
        val = -_objective(x_refined)
        if val > best_val:
            best_val = val
            best_x = x_refined.copy()
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Final check: if we have time, do one more refinement
    if best_x is not None and time.time() < deadline - 5.0:
        try:
            remaining = deadline - time.time() - 2.0
            if remaining > 2.0:
                x_final = _refine_nelder_mead(
                    best_x,
                    max_iter=500,
                    timeout=remaining,
                    start_time=start_time
                )
                val = -_objective(x_final)
                if val > best_val:
                    best_val = val
                    best_x = x_final.copy()
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: random points
        points = random_points_in_triangle()
        best_x = points.flatten()
    
    # Convert to (N, 2) array
    result = best_x.reshape(-1, 2)
    
    # Ensure all points are inside the triangle
    result = np.array([_project_to_triangle(p) for p in result])
    
    return result
