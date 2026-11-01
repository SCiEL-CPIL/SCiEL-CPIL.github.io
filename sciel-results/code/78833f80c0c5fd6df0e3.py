"""Rewritten program: sphere covering with 50 points using physics-based repulsion + Lloyd relaxation."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _fibonacci_sphere(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _hull_covering_radius(pts):
    """Compute covering radius (in radians) from convex hull facet circumcenters."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            e1 = v2 - v1
            e2 = v3 - v1
            n = np.cross(e1, e2)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            if np.dot(n, v1) < 0:
                n = -n
            # circumcenter of triangle on unit sphere: solve for point c on sphere
            # equidistant from v1, v2, v3. Use the formula: c = n * cot(theta/2) style
            # Actually: the circumcenter of a spherical triangle with vertices on unit sphere
            # is in direction of (cross(v1,v2)+cross(v2,v3)+cross(v3,v1)) normalized,
            # but more robustly: solve linear system.
            # For a triangle on the unit sphere, the circumcenter is:
            # c = (|e1|^2 * (v2 x v3) + |e2|^2 * (v3 x v1) + |e3|^2 * (v1 x v2)) / denom
            # where e3 = v1 - v2, and denom is chosen so |c| = 1
            e3 = v1 - v2
            d1 = np.dot(e1, e1)
            d2 = np.dot(e2, e2)
            d3 = np.dot(e3, e3)
            c_raw = d1 * np.cross(v2, v3) + d2 * np.cross(v3, v1) + d3 * np.cross(v1, v2)
            cn = np.linalg.norm(c_raw)
            if cn < 1e-14:
                # degenerate; use centroid as approximation
                c = (v1 + v2 + v3) / 3.0
                c = c / np.linalg.norm(c)
            else:
                c = c_raw / cn
            # Ensure c is on the correct side
            if np.dot(c, v1) < 0:
                c = -c
            # Distance from c to v1 (chord length)
            d_chord = np.linalg.norm(c - v1)
            angle = 2.0 * np.arcsin(np.clip(d_chord / 2.0, 0.0, 1.0))
            if angle > max_r:
                max_r = angle
        return max_r
    except Exception:
        return np.pi

def _repulsion_forces(pts, power=4.0):
    """Compute electrostatic-like repulsion forces on each point."""
    n = len(pts)
    forces = np.zeros_like(pts)
    for i in range(n):
        diff = pts[i] - pts  # (n, 3)
        dist_sq = np.sum(diff * diff, axis=1)
        # Avoid self
        dist_sq[i] = 1.0
        dist = np.sqrt(dist_sq)
        # Force magnitude: 1 / dist^power, direction: away from others
        magnitude = np.ones(n) / np.maximum(dist_sq, 1e-12) ** (power / 2.0)
        magnitude[i] = 0.0
        forces[i] = np.sum(magnitude[:, np.newaxis] * diff, axis=0)
    return forces

def _lloyd_step(pts):
    """One Lloyd iteration: move each point toward the circumcenter of its Voronoi cell
    (approximated by the centroid of the hull facet circumcenters that are 'closest').
    For a proper Lloyd step on the sphere, we move each point toward the centroid of
    the points in its spherical Voronoi cell. We approximate this by computing the
    convex hull and for each point, averaging the circumcenters of adjacent facets."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts
    
    n = len(pts)
    new_pts = pts.copy()
    
    # For each point, find adjacent facets and their circumcenters
    point_facets = [[] for _ in range(n)]
    for idx, simplex in enumerate(hull.simplices):
        for p in simplex:
            point_facets[p].append(idx)
    
    for i in range(n):
        if not point_facets[i]:
            continue
        centers = []
        for fidx in point_facets[i]:
            simplex = hull.simplices[fidx]
            v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            e1 = v2 - v1
            e2 = v3 - v1
            n_vec = np.cross(e1, e2)
            nn = np.linalg.norm(n_vec)
            if nn < 1e-12:
                continue
            n_vec /= nn
            if np.dot(n_vec, v1) < 0:
                n_vec = -n_vec
            e3 = v1 - v2
            d1 = np.dot(e1, e1)
            d2 = np.dot(e2, e2)
            d3 = np.dot(e3, e3)
            c_raw = d1 * np.cross(v2, v3) + d2 * np.cross(v3, v1) + d3 * np.cross(v1, v2)
            cn = np.linalg.norm(c_raw)
            if cn < 1e-14:
                c = (v1 + v2 + v3) / 3.0
                c = c / np.linalg.norm(c)
            else:
                c = c_raw / cn
            if np.dot(c, v1) < 0:
                c = -c
            centers.append(c)
        
        if centers:
            centroid = np.mean(centers, axis=0)
            nc = np.linalg.norm(centroid)
            if nc > 1e-14:
                centroid /= nc
                # Move point toward centroid with a step
                new_pts[i] = 0.5 * pts[i] + 0.5 * centroid
    
    return _normalize(new_pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Phase 1: Physics-based repulsion relaxation
    def _repulsion_optimize(init_pts, n_iters=2000, power=6.0):
        pts = init_pts.copy()
        lr = 0.02
        for it in range(n_iters):
            if time.time() > deadline - 15:
                break
            forces = _repulsion_forces(pts, power=power)
            # Project forces to tangent space
            dots = np.sum(forces * pts, axis=1, keepdims=True)
            forces_tangent = forces - dots * pts
            pts = pts + lr * forces_tangent
            pts = _normalize(pts)
            # Reduce learning rate
            lr = max(0.001, 0.02 * 0.995 ** it)
        return pts
    
    # Phase 2: Lloyd relaxation
    def _lloyd_relax(pts, n_iters=100):
        for it in range(n_iters):
            if time.time() > deadline - 10:
                break
            pts = _lloyd_step(pts)
        return pts
    
    # Phase 3: Fine-grained local optimization using gradient descent on covering radius surrogate
    def _local_polish(pts, n_iters=500):
        """Gradient descent on a soft-max covering radius surrogate."""
        # Use a moderate grid for the surrogate
        n_grid = 20000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        grid = _normalize(grid)
        
        beta = 30.0
        lr = 0.005
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            
            # Dots: (M, N)
            dots = grid @ pts.T
            dist_sq = 2.0 - 2.0 * dots
            dist_sq = np.clip(dist_sq, 0.0, None)
            
            # Soft min over sites for each grid point
            A = 2.0 * beta
            exp_dots = np.exp(A * dots)
            S = np.sum(exp_dots, axis=1)
            inv_S = 1.0 / S
            sum_inv_S = np.sum(inv_S)
            
            # Objective (for tracking, not needed for gradient)
            # Obj = 2 + (1/beta) * log(sum_inv_S)
            
            # Gradient
            # w_i = (1/S_i) / sum_inv_S
            w = inv_S / sum_inv_S
            # P(j|i) = exp(2*beta*dots[i,j]) / S_i
            P = exp_dots / S[:, np.newaxis]
            
            # grad_pts[j] = -2 * sum_i (w_i * P[i,j] * grid[i])
            grad_pts = -2.0 * (grid.T * w) @ P  # (3, N)
            
            # Project to tangent space
            dots_grad = np.sum(grad_pts * pts, axis=0)
            grad_tangent = grad_pts - dots_grad[np.newaxis, :] * pts
            
            pts = pts + lr * grad_tangent
            pts = _normalize(pts)
            
            # Adaptive learning rate
            lr = max(0.0001, 0.005 * 0.999 ** it)
        
        return pts
    
    # Run multiple restarts with different strategies
    best_points = None
    best_radius = np.inf
    
    n_restarts = 4
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            init_pts = _fibonacci_sphere(N, offset=0.5)
        elif restart == 1:
            init_pts = _fibonacci_sphere(N, offset=0.0)
        elif restart == 2:
            # Random perturbation of Fibonacci
            rng = np.random.RandomState(restart)
            init_pts = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.15
            init_pts = _normalize(init_pts)
        else:
            # Antipodal pairs plus jitter
            rng = np.random.RandomState(42 + restart)
            init_pts = rng.randn(N, 3)
            init_pts = _normalize(init_pts)
        
        # Phase 1: Repulsion
        if time.time() < deadline - 30:
            pts = _repulsion_optimize(init_pts, n_iters=1500, power=6.0)
        
        # Phase 2: Lloyd
        if time.time() < deadline - 20:
            pts = _lloyd_relax(pts, n_iters=80)
        
        # Phase 3: Local polish
        if time.time() < deadline - 10:
            pts = _local_polish(pts, n_iters=400)
        
        # Evaluate
        radius = _hull_covering_radius(pts)
        if radius < best_radius:
            best_radius = radius
            best_points = pts.copy()
    
    # Final refinement on the best points
    if best_points is not None and time.time() < deadline - 10:
        # Do a final Lloyd pass
        best_points = _lloyd_relax(best_points, n_iters=40)
        # Final polish
        if time.time() < deadline - 5:
            best_points = _local_polish(best_points, n_iters=300)
        
        final_radius = _hull_covering_radius(best_points)
        if final_radius < best_radius:
            best_radius = final_radius
            best_points = best_points.copy()
    
    # Fallback
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
