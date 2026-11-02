"""Combined program for Heilbronn-11: SLSQP on 165 quadratic constraints with multi-restart."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Safety margin from 150s limit.
DEADLINE = 140.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _project_cartesian(pts):
    """Project points into the triangle if they are slightly outside."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_valid_cart(pts, tol=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True


def _min_area_and_triplets(pts):
    """
    Compute the minimum triangle area and the indices of the triplets that achieve it.
    Returns: (min_area, list_of_triplet_indices)
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
    
    min_area = np.min(areas)
    tol = 1e-9
    mask = areas <= (min_area + tol)
    triplet_indices = np.where(mask)[0]
    
    return min_area, triplet_indices


def _bary_to_cart(uv_flat):
    """
    Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2).
    w = 1 - u - v.
    P = u*V0 + v*V1 + w*V2.
    Since V0=(0,0), P = v*(1,0) + w*(0.5, S3H) = (v + 0.5w, w*S3H).
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))


def _cart_to_bary(pts):
    """
    Convert Cartesian points to barycentric coords (u, v) where w = 1-u-v.
    Points must be inside the triangle.
    """
    uv = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        x, y = pts[i]
        w = y / S3H
        v = x - 0.5 * w
        u = 1.0 - v - w
        uv[i] = [u, v]
    return uv


def _project_to_simplex(uv_flat):
    """Project barycentric coords to the simplex u>=0, v>=0, u+v<=1."""
    uv = uv_flat.reshape(-1, 2).copy()
    for i in range(len(uv)):
        u, v = uv[i]
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        if u + v > 1.0:
            scale = 1.0 / (u + v)
            u *= scale
            v *= scale
        uv[i] = [u, v]
    return uv.flatten()


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area, _ = _min_area_and_triplets(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Initial configuration: High symmetry guess
    # Corners, mid-edges, centroid, and points on medians
    init_cart = np.array([
        [0.0, 0.0],          # Corner 0
        [1.0, 0.0],          # Corner 1
        [0.5, S3H],          # Corner 2
        [0.5, 0.0],          # Mid bottom
        [0.75, S3H / 2],     # Mid right
        [0.25, S3H / 2],     # Mid left
        [1/3, S3H/3],        # Centroid
        [0.333, 0.0],        # Bottom edge
        [0.667, 0.0],        # Bottom edge
        [0.5, S3H/3],        # Median
        [0.2, 0.1],          # Interior
    ], dtype=float)
    
    # Ensure initial points are valid
    init_cart = _project_cartesian(init_cart)
    _update_best(init_cart)
    
    # SLSQP optimization with explicit constraints
    # We optimize in barycentric coordinates (u, v) for each point.
    # Variables: x = [u0, v0, u1, v1, ..., u10, v10] (22 variables)
    
    def _slsqp_objective(uv_flat):
        """Negative of the minimum triangle area."""
        pts = _bary_to_cart(uv_flat)
        # Project to ensure validity during optimization
        pts = _project_cartesian(pts)
        area, _ = _min_area_and_triplets(pts)
        return -area
    
    def _slsqp_constraints(uv_flat):
        """
        Generate constraint functions.
        We have 165 triangle area constraints: area_i >= 0 (always true) but we want to maximize min.
        Actually, SLSQP maximizes objective subject to constraints.
        We can set up: maximize -min_area.
        But min_area is non-smooth.
        
        Alternative approach:
        Introduce a variable 't' representing the minimum area.
        Maximize t subject to:
        1. Area(i,j,k) >= t for all triplets (165 constraints)
        2. u_i >= 0, v_i >= 0, u_i + v_i <= 1 for all points (33 constraints)
        
        This is a linear programming problem if we fix the points, but points are variables.
        The area constraints are quadratic (bilinear in point coords).
        
        Let's use the standard form:
        Objective: maximize t
        Variables: [t, u0, v0, ..., u10, v10] (23 variables)
        Constraints:
        - Area(i,j,k) - t >= 0 for all triplets
        - u_i >= 0
        - v_i >= 0
        - 1 - u_i - v_i >= 0
        """
        t = uv_flat[0]
        uv_pts = uv_flat[1:]
        pts = _bary_to_cart(uv_pts)
        pts = _project_cartesian(pts)
        
        pi = pts[_T_I]
        pj = pts[_T_J]
        pk = pts[_T_K]
        dx1 = pj[:, 0] - pi[:, 0]
        dy1 = pj[:, 1] - pi[:, 1]
        dx2 = pk[:, 0] - pi[:, 0]
        dy2 = pk[:, 1] - pi[:, 1]
        cross = dx1 * dy2 - dy1 * dx2
        areas = np.abs(cross) * 0.5
        
        # Constraints: area_i - t >= 0
        # SLSQP expects constraint functions that return >= 0 for equality/inequality
        # Inequality constraints are of the form c(x) >= 0
        area_constraints = areas - t
        
        # Simplex constraints for each point
        # u_i >= 0, v_i >= 0, 1 - u_i - v_i >= 0
        uv_pts = uv_pts.reshape(-1, 2)
        u = uv_pts[:, 0]
        v = uv_pts[:, 1]
        w = 1.0 - u - v
        
        simplex_constraints = np.concatenate([u, v, w])
        
        return np.concatenate([area_constraints, simplex_constraints])
    
    # Define the full optimization problem
    # Variables: [t, u0, v0, ..., u10, v10]
    # t is unbounded below (but will be pushed up), bounded above by max possible area.
    # u, v are in [0, 1].
    
    # We need to provide gradients for SLSQP to be efficient, or let it approximate.
    # Approximate gradients are slow for 23 vars and 198 constraints.
    # Let's try to provide analytical gradients? That's complex.
    # Maybe use a smaller number of constraints? No, all 165 are needed for exactness.
    
    # Let's stick to the "targeted local search" from the current program but improve it,
    # OR use the soft-min annealing from the inspiration but with SLSQP?
    # The prompt says "SLSQP handles the non-smooth min-area constraint directly...".
    # This implies using the t-variable formulation.
    
    # Let's implement the t-variable SLSQP formulation.
    
    def _slsqp_full(uv_flat):
        """Objective: -t (minimize negative t to maximize t)"""
        return -uv_flat[0]
    
    def _slsqp_grad(uv_flat):
        """Gradient of objective"""
        grad = np.zeros_like(uv_flat)
        grad[0] = -1.0
        return grad
    
    def _slsqp_constraints_full(uv_flat):
        """
        Returns array of constraint values.
        Constraint i is satisfied if value >= 0.
        """
        t = uv_flat[0]
        uv_pts = uv_flat[1:]
        pts = _bary_to_cart(uv_pts)
        pts = _project_cartesian(pts)
        
        pi = pts[_T_I]
        pj = pts[_T_J]
        pk = pts[_T_K]
        dx1 = pj[:, 0] - pi[:, 0]
        dy1 = pj[:, 1] - pi[:, 1]
        dx2 = pk[:, 0] - pi[:, 0]
        dy2 = pk[:, 1] - pi[:, 1]
        cross = dx1 * dy2 - dy1 * dx2
        areas = np.abs(cross) * 0.5
        
        area_constraints = areas - t
        
        uv_pts = uv_pts.reshape(-1, 2)
        u = uv_pts[:, 0]
        v = uv_pts[:, 1]
        w = 1.0 - u - v
        
        simplex_constraints = np.concatenate([u, v, w])
        
        return np.concatenate([area_constraints, simplex_constraints])
    
    # We will use multi-start with SLSQP.
    # Start from the symmetric guess.
    # Also try random starts.
    
    starts = []
    # 1. Symmetric start
    init_uv = _cart_to_bary(init_cart).flatten()
    init_t = _min_area_and_triplets(init_cart)[0]
    starts.append(np.concatenate([[init_t], init_uv]))
    
    # 2. Random starts
    for _ in range(5):
        if not _elapsed_ok(10):
            break
        # Random points in triangle
        rand_cart = np.zeros((N, 2))
        for i in range(N):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            # Map to triangle
            # Barycentric: u=0, v=r1, w=r2? No.
            # Let's just generate random barycentric coords
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u, v = v, 1 - u # Reflect? Or just resample
                u, v = rng.random(), rng.random()
                while u + v > 1:
                    u, v = rng.random(), rng.random()
            rand_uv = np.array([u, v])
            # Actually, just generate random points in Cartesian and convert
            # Or generate random barycentric
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            rand_uv = np.array([u, v])
            rand_cart[i] = _bary_to_cart(rand_uv.flatten())[0]
        
        rand_cart = _project_cartesian(rand_cart)
        rand_area, _ = _min_area_and_triplets(rand_cart)
        rand_uv = _cart_to_bary(rand_cart).flatten()
        starts.append(np.concatenate([[rand_area], rand_uv]))
    
    # Run SLSQP for each start
    # Bounds: t in [-1, 1], u in [0, 1], v in [0, 1]
    bounds = [(-1.0, 1.0)] + [(0.0, 1.0), (0.0, 1.0)] * N
    
    for i, start in enumerate(starts):
        if not _elapsed_ok(10):
            break
        
        try:
            result = minimize(
                _slsqp_full,
                start,
                jac=_slsqp_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=_slsqp_constraints_full,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < -best_area:
                # Extract points
                uv_res = result.x[1:]
                pts = _bary_to_cart(uv_res)
                pts = _project_cartesian(pts)
                _update_best(pts)
                
                # Refine with targeted local search?
                # The SLSQP result should be good.
                # Let's do a quick local polish using the current program's method.
                # But SLSQP might be slow. Let's see.
                
        except Exception as e:
            pass
    
    # If SLSQP was too slow or didn't improve, fall back to the targeted local search
    if best_area < 0.01: # Threshold for "bad" result
        # Run the old targeted local search
        current_pts = init_cart.copy()
        current_area, current_triplets = _min_area_and_triplets(current_pts)
        
        def _optimize_point_local(pts, idx):
            x0 = pts[idx, 0]
            y0 = pts[idx, 1]
            
            def objective(coords):
                x, y = coords
                min_area_involving = np.inf
                for j in range(N):
                    if j == idx: continue
                    for k in range(j+1, N):
                        if k == idx: continue
                        dx1 = pts[j, 0] - x
                        dy1 = pts[j, 1] - y
                        dx2 = pts[k, 0] - x
                        dy2 = pts[k, 1] - y
                        cross = dx1 * dy2 - dy1 * dx2
                        area = 0.5 * np.abs(cross)
                        if area < min_area_involving:
                            min_area_involving = area
                return -min_area_involving
            
            def objective_penalized(coords):
                x, y = coords
                if y < 0 or y > S3H:
                    return 1e6
                x_min = y / S3
                x_max = 1 - y / S3
                if x < x_min or x > x_max:
                    return 1e6
                return objective(coords)
            
            try:
                res = minimize(objective_penalized, [x0, y0], method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
                return res.x
            except Exception:
                return [x0, y0]
        
        for outer_iter in range(1000):
            if not _elapsed_ok(5):
                break
            
            current_area, current_triplets = _min_area_and_triplets(current_pts)
            if current_area > best_area:
                best_area = current_area
                best_pts = current_pts.copy()
            
            critical_indices = set()
            for t_idx in current_triplets:
                critical_indices.add(_T_I[t_idx])
                critical_indices.add(_T_J[t_idx])
                critical_indices.add(_T_K[t_idx])
            
            improved = False
            for idx in critical_indices:
                if not _elapsed_ok(5):
                    break
                new_coords = _optimize_point_local(current_pts, idx)
                x_new, y_new = new_coords
                y_new = max(0.0, min(S3H, y_new))
                x_min = y_new / S3
                x_max = 1 - y_new / S3
                x_new = max(x_min, min(x_max, x_new))
                
                old_x, old_y = current_pts[idx, 0], current_pts[idx, 1]
                current_pts[idx, 0] = x_new
                current_pts[idx, 1] = y_new
                
                new_area, _ = _min_area_and_triplets(current_pts)
                if new_area > current_area + 1e-10:
                    current_area = new_area
                    improved = True
                else:
                    current_pts[idx, 0] = old_x
                    current_pts[idx, 1] = old_y
            
            if not improved:
                idx = list(critical_indices)[rng.integers(len(critical_indices))]
                old_x, old_y = current_pts[idx, 0], current_pts[idx, 1]
                new_x = old_x + rng.normal(0, 0.01)
                new_y = old_y + rng.normal(0, 0.01)
                new_y = max(0.0, min(S3H, new_y))
                x_min = new_y / S3
                x_max = 1 - new_y / S3
                new_x = max(x_min, min(x_max, new_x))
                
                current_pts[idx, 0] = new_x
                current_pts[idx, 1] = new_y
                new_area, _ = _min_area_and_triplets(current_pts)
                if new_area > current_area:
                    current_area = new_area
                else:
                    current_pts[idx, 0] = old_x
                    current_pts[idx, 1] = old_y
    
    if best_pts is None:
        return init_cart
        
    return best_pts

# EVOLVE-BLOCK-END
