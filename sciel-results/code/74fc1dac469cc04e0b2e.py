import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

# Fixed vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    return 0.5 * np.abs(det)

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _soft_min_area(points, temperature=0.005):
    """Smooth approximation of min area."""
    areas = _compute_areas(points)
    areas = np.maximum(areas, 1e-15)
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum

def _repulsive_energy(params, power=6.0, boundary_weight=50.0):
    """
    Repulsive force field objective.
    params: 20 values (x,y) for 10 interior points.
    The 3 vertices are fixed. Total 13 points.
    Energy = sum over all pairs of 1/d^power + boundary penalty.
    """
    pts_interior = params.reshape(10, 2)
    # All 13 points: 10 interior + 3 vertices
    all_pts = np.vstack([pts_interior, V0, V1, V2])
    
    # Compute pairwise distances
    n_total = 13
    # Use broadcasting for pairwise distances
    diff = all_pts[:, np.newaxis, :] - all_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Avoid division by zero for self-distances
    np.fill_diagonal(dists, 1.0)
    
    # Repulsive energy: sum of 1/d^power for all pairs (i < j)
    # Use upper triangle only
    iu = np.triu_indices(n_total, k=1)
    pair_dists = dists[iu]
    pair_dists = np.maximum(pair_dists, 1e-10)
    energy = np.sum(pair_dists ** (-power))
    
    # Boundary penalty: push interior points away from edges
    # For each interior point, compute distance to each of the 3 edges
    # Edge 1: from (0,0) to (1,0), line: y = 0, distance = y
    # Edge 2: from (1,0) to (0.5, H), line: sqrt(3)*x + y = sqrt(3), distance = (sqrt(3) - sqrt(3)*x - y)/2
    # Edge 3: from (0.5, H) to (0,0), line: -sqrt(3)*x + y = 0, distance = (-sqrt(3)*x + y)/2
    
    x = pts_interior[:, 0]
    y = pts_interior[:, 1]
    
    # Distance to edge 1 (bottom): y
    d1 = y
    # Distance to edge 2 (right): (sqrt(3) - sqrt(3)*x - y) / 2
    d2 = (SQRT3 - SQRT3 * x - y) / 2.0
    # Distance to edge 3 (left): (-sqrt(3)*x + y) / 2
    d3 = (-SQRT3 * x + y) / 2.0
    
    # Boundary penalty: large if any distance is negative (outside) or very small
    # Use a smooth penalty: sum of max(0, -d)^2 for each edge
    penalty = 0.0
    for d in [d1, d2, d3]:
        penalty += boundary_weight * np.sum(np.maximum(0, -d) ** 2)
        # Also add a small penalty for being too close to the boundary
        penalty += 1.0 * np.sum(np.maximum(0, 0.01 - d) ** 2)
    
    return energy + penalty

def _repulsive_grad(params, power=6.0, boundary_weight=50.0):
    """
    Gradient of repulsive energy.
    params: 20 values (x,y) for 10 interior points.
    """
    pts_interior = params.reshape(10, 2)
    all_pts = np.vstack([pts_interior, V0, V1, V2])
    
    n_total = 13
    diff = all_pts[:, np.newaxis, :] - all_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    dists = np.maximum(dists, 1e-10)
    
    # Gradient of sum(1/d^power) with respect to point positions
    # d/dx_i [1/|p_i - p_j|^power] = -power * (p_i - p_j) / |p_i - p_j|^(power+2)
    # For each interior point m (0..9), gradient contribution from all other points
    
    grad = np.zeros((10, 2))
    for m in range(10):
        # Point m interacts with all other 12 points
        for j in range(n_total):
            if j == m:
                continue
            d_vec = all_pts[m] - all_pts[j]
            d = dists[m, j]
            # d/d(p_m) [1/d^power] = -power * d_vec / d^(power+2)
            grad[m] += -power * d_vec / (d ** (power + 2))
    
    # Boundary penalty gradient
    x = pts_interior[:, 0]
    y = pts_interior[:, 1]
    
    # Edge 1: d1 = y, penalty = boundary_weight * max(0, -y)^2 + max(0, 0.01-y)^2
    # d/dy [max(0, -y)^2] = -2*max(0, -y) if y < 0 else 0
    # d/dy [max(0, 0.01-y)^2] = -2*max(0, 0.01-y) if y < 0.01 else 0
    for m in range(10):
        # Edge 1
        if y[m] < 0:
            grad[m, 1] += boundary_weight * (-2 * (-y[m]))
        if y[m] < 0.01:
            grad[m, 1] += (-2 * (0.01 - y[m]))
        
        # Edge 2: d2 = (sqrt(3) - sqrt(3)*x - y) / 2
        # d/dx [max(0, -d2)^2] = -2*max(0,-d2) * d/dx(-d2) = -2*max(0,-d2) * (sqrt(3)/2)
        # d/dy [max(0, -d2)^2] = -2*max(0,-d2) * d/dy(-d2) = -2*max(0,-d2) * (1/2)
        d2 = (SQRT3 - SQRT3 * x[m] - y[m]) / 2.0
        if d2 < 0:
            grad[m, 0] += boundary_weight * (-2 * (-d2)) * (SQRT3 / 2)
            grad[m, 1] += boundary_weight * (-2 * (-d2)) * (1 / 2)
        if d2 < 0.01:
            grad[m, 0] += (-2 * (0.01 - d2)) * (SQRT3 / 2)
            grad[m, 1] += (-2 * (0.01 - d2)) * (1 / 2)
        
        # Edge 3: d3 = (-sqrt(3)*x + y) / 2
        d3 = (-SQRT3 * x[m] + y[m]) / 2.0
        if d3 < 0:
            grad[m, 0] += boundary_weight * (-2 * (-d3)) * (-SQRT3 / 2)
            grad[m, 1] += boundary_weight * (-2 * (-d3)) * (1 / 2)
        if d3 < 0.01:
            grad[m, 0] += (-2 * (0.01 - d3)) * (-SQRT3 / 2)
            grad[m, 1] += (-2 * (0.01 - d3)) * (1 / 2)
    
    return grad.flatten()

def _bary_to_cart_10(bary):
    """Convert 10 barycentric coords to Cartesian, ensuring validity."""
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    pts = np.empty((10, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    return pts

def _cart_to_bary_10(pts):
    """Convert 10 Cartesian points to barycentric coords."""
    bary = np.empty((10, 2))
    for m in range(10):
        x = pts[m, 0]
        y = pts[m, 1]
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
    return bary

def _soft_min_objective(params, temperature=0.005):
    """Soft-min objective for 10 interior points."""
    pts_interior = params.reshape(10, 2)
    all_pts = np.vstack([pts_interior, V0, V1, V2])
    # Recompute areas for all 11 points (10 interior + ... wait, we have 13 points)
    # Actually, we need 11 points total. Let's reconsider.
    # The task says 11 points. So we have 11 points total, not 10 interior + 3 vertices.
    # Let me fix this: we have 11 points, all free to move.
    # But the strategy says "10-point set of interior points plus the 3 fixed vertices" = 13 points.
    # That doesn't match the task of 11 points.
    # Let me re-read: "Model the configuration as a 10-point set of interior points plus the 3 fixed vertices"
    # That's 13 points, but we only need 11. This is confusing.
    # I think the intent is: use 11 points, but in the repulsive field, also consider the 3 vertices as fixed repulsive sources.
    # So we have 11 free points, and 3 fixed vertices that also exert repulsive force.
    # Total points in the force field: 14 (11 free + 3 fixed vertices).
    
    # Let me rewrite: 11 free points, 3 fixed vertices.
    pts = pts_interior  # This is wrong, we need 11 points
    return None

def _soft_min_objective_11(params, temperature=0.005):
    """Soft-min objective for 11 free points."""
    pts = params.reshape(11, 2)
    # Ensure points are inside the triangle by projecting
    # Use barycentric projection
    x = pts[:, 0]
    y = pts[:, 1]
    # Check if inside: y >= 0, y <= sqrt(3)*x, y <= -sqrt(3)*x + sqrt(3)
    # If outside, project to nearest edge
    # For simplicity, use the barycentric approach
    bary = _cart_to_bary_10(pts[:10])  # This is getting messy
    
    # Let's just compute areas directly and use soft-min
    # But we need to handle the 3 vertices as part of the 11 points? No, the 11 points are separate.
    # The task: place 11 points, maximize min triangle area among all C(11,3) triangles.
    # The repulsive field uses 11 free points + 3 fixed vertices = 14 points for the force.
    
    areas = _compute_areas(pts)
    areas = np.maximum(areas, 1e-15)
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum

def _repulsive_energy_11(params, power=6.0, boundary_weight=50.0):
    """
    Repulsive energy for 11 free points + 3 fixed vertices.
    params: 22 values (x,y) for 11 free points.
    """
    pts_free = params.reshape(11, 2)
    all_pts = np.vstack([pts_free, V0, V1, V2])
    
    n_total = 14
    diff = all_pts[:, np.newaxis, :] - all_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    dists = np.maximum(dists, 1e-10)
    
    iu = np.triu_indices(n_total, k=1)
    pair_dists = dists[iu]
    pair_dists = np.maximum(pair_dists, 1e-10)
    energy = np.sum(pair_dists ** (-power))
    
    # Boundary penalty for free points
    x = pts_free[:, 0]
    y = pts_free[:, 1]
    
    d1 = y
    d2 = (SQRT3 - SQRT3 * x - y) / 2.0
    d3 = (-SQRT3 * x + y) / 2.0
    
    penalty = 0.0
    for d in [d1, d2, d3]:
        penalty += boundary_weight * np.sum(np.maximum(0, -d) ** 2)
        penalty += 1.0 * np.sum(np.maximum(0, 0.01 - d) ** 2)
    
    return energy + penalty

def _repulsive_grad_11(params, power=6.0, boundary_weight=50.0):
    """Gradient of repulsive energy for 11 free points."""
    pts_free = params.reshape(11, 2)
    all_pts = np.vstack([pts_free, V0, V1, V2])
    
    n_total = 14
    diff = all_pts[:, np.newaxis, :] - all_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1.0)
    dists = np.maximum(dists, 1e-10)
    
    grad = np.zeros((11, 2))
    for m in range(11):
        for j in range(n_total):
            if j == m:
                continue
            d_vec = all_pts[m] - all_pts[j]
            d = dists[m, j]
            grad[m] += -power * d_vec / (d ** (power + 2))
    
    x = pts_free[:, 0]
    y = pts_free[:, 1]
    
    for m in range(11):
        if y[m] < 0:
            grad[m, 1] += boundary_weight * (-2 * (-y[m]))
        if y[m] < 0.01:
            grad[m, 1] += (-2 * (0.01 - y[m]))
        
        d2 = (SQRT3 - SQRT3 * x[m] - y[m]) / 2.0
        if d2 < 0:
            grad[m, 0] += boundary_weight * (-2 * (-d2)) * (SQRT3 / 2)
            grad[m, 1] += boundary_weight * (-2 * (-d2)) * (1 / 2)
        if d2 < 0.01:
            grad[m, 0] += (-2 * (0.01 - d2)) * (SQRT3 / 2)
            grad[m, 1] += (-2 * (0.01 - d2)) * (1 / 2)
        
        d3 = (-SQRT3 * x[m] + y[m]) / 2.0
        if d3 < 0:
            grad[m, 0] += boundary_weight * (-2 * (-d3)) * (-SQRT3 / 2)
            grad[m, 1] += boundary_weight * (-2 * (-d3)) * (1 / 2)
        if d3 < 0.01:
            grad[m, 0] += (-2 * (0.01 - d3)) * (-SQRT3 / 2)
            grad[m, 1] += (-2 * (0.01 - d3)) * (1 / 2)
    
    return grad.flatten()

def _project_to_triangle(pts):
    """Project points to be inside the triangle."""
    pts = pts.copy()
    for m in range(len(pts)):
        x, y = pts[m]
        # Check edges
        # Edge 1: y >= 0
        if y < 0:
            y = 0
        # Edge 2: sqrt(3)*x + y <= sqrt(3)
        if SQRT3 * x + y > SQRT3:
            # Project to line sqrt(3)*x + y = sqrt(3)
            # Normal direction: (sqrt(3), 1)/2
            # Point on line closest to (x,y)
            t = (SQRT3 * x + y - SQRT3) / (3 + 1)
            x -= SQRT3 * t
            y -= t
        # Edge 3: -sqrt(3)*x + y >= 0 => y >= sqrt(3)*x
        if y < SQRT3 * x:
            # Project to line y = sqrt(3)*x
            # Normal direction: (-sqrt(3), 1)/2
            t = (SQRT3 * x - y) / (3 + 1)
            x += SQRT3 * t
            y += t
        pts[m] = [x, y]
    return pts

def _random_valid_points(rng, n=11):
    """Generate random valid points inside the triangle."""
    pts = np.zeros((n, 2))
    for m in range(n):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                x = v + 0.5 * (1 - u - v)
                y = H * (1 - u - v)
                pts[m] = [x, y]
                break
    return pts

def _initial_grid():
    """Generate a structured initial guess for 11 points."""
    # Place points in a rough grid pattern
    positions = []
    # 3 points along bottom
    for t in [0.15, 0.5, 0.85]:
        positions.append([t, 0.08])
    # 3 points in middle-lower
    for t in [0.25, 0.5, 0.75]:
        positions.append([t, 0.35])
    # 3 points in middle-upper
    for t in [0.35, 0.5, 0.65]:
        positions.append([t, 0.6])
    # 2 points near top
    positions.append([0.4, 0.8])
    positions.append([0.6, 0.8])
    
    pts = np.array(positions)
    return _project_to_triangle(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    def _update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_points = pts.copy()
            best_score = score
            return True
        return False
    
    def _check_valid(pts):
        """Check if all points are inside the triangle."""
        x = pts[:, 0]
        y = pts[:, 1]
        return (np.all(y >= -1e-8) and 
                np.all(SQRT3 * x + y <= SQRT3 + 1e-8) and 
                np.all(y >= SQRT3 * x - 1e-8) and
                np.all(x >= -1e-8) and np.all(x <= 1 + 1e-8))
    
    # Phase 1: Repulsive force field optimization
    # Start with grid initial guess
    initial_pts = _initial_grid()
    initial_params = initial_pts.flatten()
    
    # Optimize repulsive energy
    try:
        result = minimize(
            _repulsive_energy_11,
            initial_params,
            jac=_repulsive_grad_11,
            args=(6.0, 50.0),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < np.inf:
            pts = result.x.reshape(11, 2)
            pts = _project_to_triangle(pts)
            if _check_valid(pts):
                score = _min_area(pts)
                _update_best(pts, score)
    except Exception:
        pass
    
    # Phase 2: Soft-min refinement
    if best_points is not None:
        if time.time() > deadline - 20:
            return best_points
        
        best_params = best_points.flatten()
        
        # Refine with soft-min at various temperatures
        for temp in [0.01, 0.005, 0.002, 0.001]:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(
                    _soft_min_objective_11,
                    best_params,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.fun < -np.inf + 1:
                    pts = result.x.reshape(11, 2)
                    pts = _project_to_triangle(pts)
                    if _check_valid(pts):
                        score = _min_area(pts)
                        if _update_best(pts, score):
                            best_params = pts.flatten()
            except Exception:
                pass
    
    # Phase 3: Multiple restarts with repulsive field
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break
        
        random_pts = _random_valid_points(rng)
        random_params = random_pts.flatten()
        
        try:
            result = minimize(
                _repulsive_energy_11,
                random_params,
                jac=_repulsive_grad_11,
                args=(6.0, 50.0),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < np.inf:
                pts = result.x.reshape(11, 2)
                pts = _project_to_triangle(pts)
                if _check_valid(pts):
                    score = _min_area(pts)
                    if _update_best(pts, score):
                        # Refine with soft-min
                        if time.time() < deadline - 10:
                            best_params = pts.flatten()
                            for temp in [0.005, 0.001]:
                                if time.time() > deadline - 5:
                                    break
                                try:
                                    result2 = minimize(
                                        _soft_min_objective_11,
                                        best_params,
                                        args=(temp,),
                                        method='L-BFGS-B',
                                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                                    )
                                    if result2.fun < -np.inf + 1:
                                        pts2 = result2.x.reshape(11, 2)
                                        pts2 = _project_to_triangle(pts2)
                                        if _check_valid(pts2):
                                            score2 = _min_area(pts2)
                                            _update_best(pts2, score2)
                                            best_params = pts2.flatten()
                                except Exception:
                                    pass
        except Exception:
            pass
    
    # Phase 4: Differential evolution for global search if time permits
    if best_points is None or time.time() < deadline - 30:
        if time.time() > deadline - 25:
            if best_points is not None:
                return best_points
            bary_pts = _random_valid_points(rng)
            return _project_to_triangle(bary_pts)
        
        def de_objective(params):
            pts = params.reshape(11, 2)
            # Simple penalty for being outside
            x = pts[:, 0]
            y = pts[:, 1]
            penalty = 0.0
            penalty += np.sum(np.maximum(0, -y) ** 2)
            penalty += np.sum(np.maximum(0, SQRT3 * x + y - SQRT3) ** 2)
            penalty += np.sum(np.maximum(0, SQRT3 * x - y) ** 2)
            if penalty > 1.0:
                return 1.0 + penalty
            return _repulsive_energy_11(params, 6.0, 50.0)
        
        bounds = [(0.0, 1.0)] * 22
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                seed=42,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=False,
                init='sobol'
            )
            if de_result.fun < np.inf:
                pts = de_result.x.reshape(11, 2)
                pts = _project_to_triangle(pts)
                if _check_valid(pts):
                    score = _min_area(pts)
                    if _update_best(pts, score):
                        # Refine
                        if time.time() < deadline - 10:
                            best_params = pts.flatten()
                            for temp in [0.005, 0.001]:
                                if time.time() > deadline - 5:
                                    break
                                try:
                                    result2 = minimize(
                                        _soft_min_objective_11,
                                        best_params,
                                        args=(temp,),
                                        method='L-BFGS-B',
                                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                                    )
                                    if result2.fun < -np.inf + 1:
                                        pts2 = result2.x.reshape(11, 2)
                                        pts2 = _project_to_triangle(pts2)
                                        if _check_valid(pts2):
                                            score2 = _min_area(pts2)
                                            _update_best(pts2, score2)
                                            best_params = pts2.flatten()
                                except Exception:
                                    pass
        except Exception:
            pass
    
    # Final fallback
    if best_points is None:
        best_points = _random_valid_points(rng)
        best_points = _project_to_triangle(best_points)
    
    # Final projection to ensure validity
    best_points = _project_to_triangle(best_points)
    
    return best_points
