"""Heilbronn 11 triangles: Simulated Annealing with Barycentric Coordinates and Local Polish."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

DEADLINE = 145  # Safety margin from 150s limit

# Precompute all triplets for N=11 points
# There are C(11, 3) = 165 triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Barycentric to Cartesian conversion
# Vertices: A=(0,0), B=(1,0), C=(0.5, S3/2)
# P = (1-u-v)A + uB + vC = u*(1,0) + v*(0.5, S3/2)
# P = (u + 0.5v, v*S3/2)

def _bary_to_cart(bary):
    """Convert barycentric coordinates (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + 0.5 * v
    y = v * S3H
    return np.column_stack((x, y))

def _min_triangle_area_cartesian(pts):
    """
    Compute the minimum area of any triangle formed by three points.
    Input: pts is an (N, 2) array of Cartesian coordinates.
    Returns the raw area (not normalized).
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _min_triangle_area_bary(bary):
    """
    Compute the minimum area of any triangle formed by three points given in barycentric coordinates.
    Area in barycentric coordinates is proportional to the area in Cartesian.
    Specifically, Area_cart = TRI_AREA * Area_bary_relative_to_unit_triangle_area?
    Actually, it's easier to just convert to Cartesian or use the determinant formula directly.
    The area of a triangle with barycentric coordinates (u1,v1), (u2,v2), (u3,v3) is:
    Area = TRI_AREA * |det([[1, u1, v1], [1, u2, v2], [1, u3, v3]])|
    Wait, the determinant gives the ratio of areas relative to the reference triangle.
    So Area = TRI_AREA * |det|
    """
    # Construct matrix M where rows are [1, u, v]
    # We can compute the determinant for each triplet
    # det = 1*(u2*v3 - u3*v2) - u1*(1*v3 - 1*v2) + v1*(1*u3 - 1*u2)
    # det = u2*v3 - u3*v2 - u1*v3 + u1*v2 + v1*u3 - v1*u2
    
    u = bary[:, 0]
    v = bary[:, 1]
    
    u1 = u[_T_I]
    v1 = v[_T_I]
    u2 = u[_T_J]
    v2 = v[_T_J]
    u3 = u[_T_K]
    v3 = v[_T_K]
    
    dets = (u2*v3 - u3*v2) - u1*(v3 - v2) + v1*(u3 - u2)
    areas = TRI_AREA * np.abs(dets)
    return np.min(areas)

def _project_to_bary(bary):
    """Ensure barycentric coordinates are within the triangle (u>=0, v>=0, u+v<=1)."""
    u = bary[:, 0]
    v = bary[:, 1]
    # Simple clipping is not ideal for barycentric as it distorts the metric,
    # but for SA steps, we can just reject moves that go out of bounds or project.
    # Projection in barycentric space to the triangle:
    # If u < 0, set u=0. If v < 0, set v=0.
    # If u+v > 1, scale (u,v) by 1/(u+v) to bring to boundary.
    
    u_clipped = np.maximum(0, u)
    v_clipped = np.maximum(0, v)
    sum_uv = u_clipped + v_clipped
    mask = sum_uv > 1
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u_clipped[mask] *= scale
        v_clipped[mask] *= scale
        
    return np.column_stack((u_clipped, v_clipped))

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts_bary = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_pts_bary, best_area
        # Ensure points are inside
        bary = _project_to_bary(bary)
        area = _min_triangle_area_bary(bary)
        if area > best_area:
            best_area = area
            best_pts_bary = bary.copy()
            return True
        return False

    def _generate_initial_configs():
        """Generate diverse starting points in barycentric coordinates."""
        configs = []
        
        # 1. Symmetric base: Corners, midpoints, centroid, etc.
        # Vertices in bary: (0,0), (1,0), (0,1)
        # Midpoints: (0.5, 0), (0.5, 0.5), (0, 0.5)
        # Centroid: (1/3, 1/3)
        # 9 points: 3 corners, 3 midpoints, 1 centroid = 7. Need 4 more.
        # Let's use a 3x3 grid inside the triangle.
        # Grid points (i, j) such that i/j <= 1/3? No, just standard grid.
        # i in 0..2, j in 0..2, i+j <= 2?
        # (0,0), (1,0), (2,0) -> (0,0), (0.5,0), (1,0)
        # (0,1), (1,1), (2,1) -> (0,0.5), (0.5,0.5), (1,0.5) -- (1,0.5) is outside? 1+0.5=1.5>1.
        # Valid barycentric grid: i, j >= 0, i+j <= K.
        # Let's pick 11 points from a fine grid.
        
        # Specific symmetric configuration known to be good for small N
        base_sym = np.array([
            [0.0, 0.0],       # Vertex A
            [1.0, 0.0],       # Vertex B
            [0.0, 1.0],       # Vertex C
            [0.5, 0.0],       # Mid AB
            [0.0, 0.5],       # Mid AC
            [0.5, 0.5],       # Mid BC
            [1/3, 1/3],       # Centroid
            [1/6, 1/6],       # 1/6, 1/6
            [5/6, 1/6],       # 5/6, 1/6
            [1/6, 5/6],       # 1/6, 5/6
            [0.5, 0.0],       # Duplicate? No, let's pick a different one.
        ])
        # The last point [0.5, 0.0] is a duplicate of index 3.
        # Let's replace with [0.5, 0.5] duplicate? No.
        # Let's just use random perturbation of this base for the "symmetric" start.
        
        configs.append(base_sym)

        # 2. Random uniform points in barycentric space
        # To sample uniformly in the triangle, generate u, v ~ U(0,1).
        # If u+v > 1, reflect: u = 1-u, v = 1-v.
        for _ in range(10):
            bary = rng.random((N, 2))
            mask = bary[:, 0] + bary[:, 1] > 1
            bary[mask, 0] = 1 - bary[mask, 0]
            bary[mask, 1] = 1 - bary[mask, 1]
            configs.append(bary)
            
        # 3. Perturbed symmetric base
        for _ in range(5):
            pts = base_sym + rng.normal(0, 0.1, size=(N, 2))
            configs.append(_project_to_bary(pts))

        return configs

    def _simulated_annealing(bary0, steps, t_start, t_end, alpha):
        """
        Run Simulated Annealing.
        bary0: initial barycentric coordinates (N, 2)
        steps: number of iterations
        t_start: initial temperature
        t_end: final temperature
        alpha: cooling rate
        """
        current_bary = bary0.copy()
        current_area = _min_triangle_area_bary(current_bary)
        best_bary = current_bary.copy()
        best_area = current_area
        
        for i in range(steps):
            if not _elapsed_ok(10):
                break
            
            # Generate a random move
            # Perturb a random subset of points
            n_perturb = rng.integers(1, 4) # 1 to 3 points
            idx = rng.choice(N, size=n_perturb, replace=False)
            
            # Step size proportional to temperature
            step_size = t_start * (t_end / t_start) ** (i / steps)
            
            new_bary = current_bary.copy()
            new_bary[idx] += rng.normal(0, step_size, size=(n_perturb, 2))
            new_bary = _project_to_bary(new_bary)
            
            new_area = _min_triangle_area_bary(new_bary)
            delta = new_area - current_area
            
            # Accept if better, or with probability exp(delta / T)
            if delta > 0 or rng.random() < np.exp(delta / (t_start * (t_end / t_start) ** (i / steps) + 1e-12)):
                current_bary = new_bary
                current_area = new_area
                
                if current_area > best_area:
                    best_area = current_area
                    best_bary = current_bary.copy()
                    _update_best(best_bary)
                    
        return best_bary, best_area

    # Phase 1: Multi-start Simulated Annealing
    configs = _generate_initial_configs()
    
    # Parameters for SA
    # Area is small (~0.01). Delta area is small.
    # T needs to be comparable to delta area.
    # Let's try T_start = 0.05, T_end = 0.001
    # Steps: ~2000 per start? 10 starts * 2000 = 20000 evals.
    # Each eval is vectorized over 165 triplets. Should be fast.
    
    for config in configs:
        if not _elapsed_ok(20):
            break
        
        # Run SA for a fixed number of steps or until time limit
        # We can run multiple SA instances from the same start with different random seeds?
        # Or just one long run.
        # Let's do 2 runs of 1000 steps each to allow for some diversity.
        for _ in range(2):
            if not _elapsed_ok(5):
                break
            sa_best_bary, sa_best_area = _simulated_annealing(config, steps=1000, t_start=0.05, t_end=0.005, alpha=0.99)
            # _update_best is called inside SA, so best_pts_bary is updated globally.
            
    # Phase 2: Local Polish on the best candidate
    if best_pts_bary is not None and _elapsed_ok(10):
        # Convert best barycentric to Cartesian for Nelder-Mead?
        # Or run NM in barycentric space?
        # NM in barycentric space is tricky because of the constraint u+v<=1.
        # It's safer to polish in Cartesian space using projection.
        
        best_cart = _bary_to_cart(best_pts_bary)
        
        def nm_obj(x):
            p = x.reshape(-1, 2)
            # Project to triangle in Cartesian space
            # This projection is not differentiable at boundaries, but NM doesn't need gradients.
            # We can use the same projection logic as before.
            p = _project_to_cartesian(p)
            return -_min_triangle_area_cartesian(p)
            
        try:
            res = minimize(
                nm_obj,
                best_cart.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_cartesian(p)
            # Convert back to barycentric to update global best?
            # Or just evaluate Cartesian area.
            area = _min_triangle_area_cartesian(p)
            if area > best_area:
                best_area = area
                # We need to return Cartesian points, so we can just keep this.
                # But our internal state is barycentric.
                # Let's convert back to barycentric for consistency, or just return p.
                # The function returns Cartesian points.
                best_pts_bary = _cart_to_bary(p) # Need this function
                _update_best(best_pts_bary)
        except Exception:
            pass

    # Phase 3: Final fine polish
    if best_pts_bary is not None and _elapsed_ok(5):
        best_cart = _bary_to_cart(best_pts_bary)
        
        def nm_obj_final(x):
            p = x.reshape(-1, 2)
            p = _project_to_cartesian(p)
            return -_min_triangle_area_cartesian(p)
            
        try:
            res = minimize(
                nm_obj_final,
                best_cart.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            p = res.x.reshape(-1, 2)
            p = _project_to_cartesian(p)
            area = _min_triangle_area_cartesian(p)
            if area > best_area:
                best_area = area
                best_pts_bary = _cart_to_bary(p)
                _update_best(best_pts_bary)
        except Exception:
            pass

    if best_pts_bary is None:
        # Fallback
        bary = rng.random((N, 2))
        mask = bary[:, 0] + bary[:, 1] > 1
        bary[mask, 0] = 1 - bary[mask, 0]
        bary[mask, 1] = 1 - bary[mask, 1]
        best_pts_bary = _project_to_bary(bary)

    return _bary_to_cart(best_pts_bary)

def _project_to_cartesian(pts):
    """
    Project points into the equilateral triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        # Clamp y to [0, S3H]
        y = max(0.0, min(S3H, y))
        # For a given y, x must be between the left and right boundaries.
        # Left boundary: x = y/sqrt(3)
        # Right boundary: x = 1 - y/sqrt(3)
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def _cart_to_bary(pts):
    """
    Convert Cartesian points to Barycentric coordinates.
    P = u*B + v*C, where B=(1,0), C=(0.5, S3/2).
    x = u + 0.5v
    y = v*S3/2
    => v = y / (S3/2) = 2y/S3
    => u = x - 0.5v
    """
    u = np.zeros(N)
    v = np.zeros(N)
    for i in range(N):
        x, y = pts[i]
        v[i] = 2 * y / S3
        u[i] = x - 0.5 * v[i]
    return np.column_stack((u, v))

# EVOLVE-BLOCK-END
