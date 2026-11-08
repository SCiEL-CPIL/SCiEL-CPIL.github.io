"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from scipy.optimize import minimize

N = 11

# Precompute triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0




def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by AREA."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        vecs = points - pi
        X = vecs[:, 0]
        Y = vecs[:, 1]
        C = np.outer(X, Y) - np.outer(Y, X)
        C_abs = np.abs(C)
        C_abs[i, :] = float('inf')
        C_abs[:, i] = float('inf')
        np.fill_diagonal(C_abs, float('inf'))
        current_min = np.min(C_abs)
        if current_min < min_area:
            min_area = current_min
            if min_area < 1e-12:
                return 0.0
    return (0.5 * min_area) / AREA


def _is_inside_triangle(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -tol:
        return False
    if x < -tol or x > 1.0 + tol:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + tol


def _params_to_points(params):
    """Convert 2*N parameters to N points inside the triangle using barycentric coordinates."""
    points = np.zeros((N, 2))
    for i in range(N):
        u = 1.0 / (1.0 + np.exp(-params[2*i]))
        v = 1.0 / (1.0 + np.exp(-params[2*i+1])) * (1.0 - u)
        points[i] = V0 * (1 - u - v) + V1 * u + V2 * v
    return points


def _soft_min_area(params, beta=50.0):
    """Objective: negative soft-min triangle area (smooth surrogate for min)."""
    points = _params_to_points(params)
    n = len(points)
    areas = []
    for i in range(n):
        pi = points[i]
        vecs = points - pi
        X = vecs[:, 0]
        Y = vecs[:, 1]
        C = np.outer(X, Y) - np.outer(Y, X)
        C_abs = np.abs(C)
        C_abs[i, :] = float('inf')
        C_abs[:, i] = float('inf')
        np.fill_diagonal(C_abs, float('inf'))
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mask[i, :] = False
        mask[:, i] = False
        vals = C_abs[mask]
        areas.extend((0.5 * vals) / AREA)
    areas = np.array(areas)
    areas_safe = np.maximum(areas, 1e-12)
    x = -beta * areas_safe
    x_max = np.max(x)
    sum_exp = np.sum(np.exp(x - x_max))
    log_sum_exp = x_max + np.log(sum_exp)
    soft_min_val = -1.0 / beta * log_sum_exp
    return -soft_min_val


def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    pts = [
        V0 + np.array([0.08, 0.08]),
        V1 + np.array([-0.08, 0.08]),
        V2 + np.array([0.0, -0.08]),
        (V0 + V1) / 2.0 + np.array([0.02, 0.02]),
        (V1 + V2) / 2.0 + np.array([-0.02, 0.02]),
        (V2 + V0) / 2.0 + np.array([0.02, 0.02]),
        (V0 + V1 + V2) / 3.0,
        (V0 + 2*V1 + V2) / 4.0 + np.array([0.01, 0.01]),
        (2*V0 + V1 + V2) / 4.0 + np.array([0.01, 0.01]),
        (V0 + V1 + 2*V2) / 4.0 + np.array([0.01, 0.01]),
        (V0 + V1 + V2) / 3.0 + np.array([0.05, 0.05])
    ]
    return np.array(pts)


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)


def _points_to_params(points):
    """Inverse of _params_to_points."""
    params = []
    for p in points:
        x, y = p
        v = 2 * y / np.sqrt(3)
        u = x - 0.5 * v
        eps = 1e-6
        u = np.clip(u, eps, 1-eps)
        v = np.clip(v, eps, 1-eps)
        p1 = np.log(u / (1 - u))
        s2 = v / (1 - u)
        s2 = np.clip(s2, eps, 1-eps)
        p2 = np.log(s2 / (1 - s2))
        params.append([p1, p2])
    return np.array(params).flatten()


# EVOLVE-BLOCK-START
def _force_directed_step(points, dt=0.001, repulse_strength=0.5, attract_strength=0.3, damping=0.95):
    """
    Force-directed step using symplectic (leapfrog) integrator.
    Points repel each other to maintain separation; attractive springs pull toward
    a target configuration that maximizes minimum triangle area.
    """
    n = len(points)
    forces = np.zeros_like(points)
    
    # Repulsive forces between all pairs
    for i in range(n):
        for j in range(i+1, n):
            diff = points[i] - points[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-12:
                # Random nudge if coincident
                diff = np.array([rng.random() - 0.5, rng.random() - 0.5]) * 0.01
                dist_sq = np.dot(diff, diff)
            dist = np.sqrt(dist_sq)
            # Repulsive force ~ 1/d^2
            f_mag = repulse_strength / (dist_sq + 1e-6)
            f_vec = f_mag * diff / dist
            forces[i] += f_vec
            forces[j] -= f_vec
    
    # Attractive forces: pull points toward positions that increase min area
    # Use gradient of soft-min as attraction
    # For simplicity, use a spring toward the centroid of the triangle
    centroid = (V0 + V1 + V2) / 3.0
    for i in range(n):
        diff = centroid - points[i]
        forces[i] += attract_strength * diff
    
    # Symplectic integrator (leapfrog)
    # v_new = v + 0.5*dt*a
    # x_new = x + v_new*dt
    # v_new = v_new + 0.5*dt*a_new
    # For simplicity, use semi-implicit Euler with damping
    velocities = np.zeros_like(points)  # We don't track velocity across calls, so use simple update
    points_new = points + dt * forces * damping
    
    # Project to triangle
    for i in range(n):
        p = points_new[i]
        x, y = p
        # Clamp to triangle
        if y < 0:
            y = 0.0
        if x < 0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
        if y > max_y:
            y = max_y
        points_new[i] = np.array([x, y])
    
    return points_new

def _simulated_annealing(points, rng, n_iter=500, T0=0.05, T1=0.001):
    """Simulated annealing to escape local minima."""
    current = points.copy()
    current_score = _min_triangle_area(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(n_iter):
        T = T0 * (T1 / T0) ** (i / n_iter)
        
        # Propose a move: pick a random point and perturb it
        idx = rng.randint(0, N-1)
        new_pts = current.copy()
        perturbation = np.array([rng.gauss(0, 0.02), rng.gauss(0, 0.02)])
        new_pts[idx] += perturbation
        
        # Project to triangle
        x, y = new_pts[idx]
        if y < 0: y = 0.0
        if x < 0: x = 0.0
        if x > 1.0: x = 1.0
        max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
        if y > max_y: y = max_y
        new_pts[idx] = np.array([x, y])
        
        new_score = _min_triangle_area(new_pts)
        
        # Accept if better or with probability exp((new_score - current_score)/T)
        if new_score >= current_score:
            current = new_pts
            current_score = new_score
            if new_score > best_score:
                best = new_pts.copy()
                best_score = new_score
        else:
            prob = np.exp((new_score - current_score) / T)
            if rng.random() < prob:
                current = new_pts
                current_score = new_score
    
    return best, best_score

def _slsqp_polish(points, max_iter=50):
    """Polish with SLSQP on the true objective (negative min area)."""
    def objective(params):
        pts = _params_to_points(params)
        return -_min_triangle_area(pts)
    
    def jac(params):
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(params)
        f0 = objective(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            grad[i] = (objective(params_plus) - f0) / eps
        return grad
    
    init_params = _points_to_points(points) if False else _points_to_params(points)
    
    try:
        res = minimize(
            objective,
            init_params,
            jac=jac,
            method='SLSQP',
            options={'maxiter': max_iter, 'ftol': 1e-10}
        )
        polished = _params_to_points(res.x)
        score = _min_triangle_area(polished)
        return polished, score
    except Exception:
        return points, _min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Phase 1: Multi-start L-BFGS-B with soft-min
    initial_configs = [_generate_structured_initial()]
    for _ in range(10):
        initial_configs.append(_generate_random_points(rng))
    
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        init_params = _points_to_params(init_pts)
        max_iter = 150 if idx == 0 else 80
        
        try:
            res = minimize(
                _soft_min_area,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 3000}
            )
            candidate = _params_to_points(res.x)
            score = _min_triangle_area(candidate)
            if score > best_min:
                best_min = score
                best_points = candidate.copy()
        except Exception:
            continue
    
    # Phase 2: Force-directed exploration
    if best_points is not None and time.time() < deadline - 20:
        fd_points = best_points.copy()
        n_fd_steps = 200
        for step in range(n_fd_steps):
            if time.time() > deadline - 15:
                break
            fd_points = _force_directed_step(fd_points, dt=0.0005, repulse_strength=0.3, attract_strength=0.1, damping=0.98)
            score = _min_triangle_area(fd_points)
            if score > best_min:
                best_min = score
                best_points = fd_points.copy()
    
    # Phase 3: Simulated annealing
    if best_points is not None and time.time() < deadline - 15:
        sa_points, sa_score = _simulated_annealing(best_points, rng, n_iter=300, T0=0.03, T1=0.005)
        if sa_score > best_min:
            best_min = sa_score
            best_points = sa_points.copy()
    
    # Phase 4: SLSQP polish
    if best_points is not None and time.time() < deadline - 5:
        polished, polish_score = _slsqp_polish(best_points, max_iter=30)
        if polish_score > best_min:
            best_min = polish_score
            best_points = polished.copy()
    
    if best_points is None:
        best_points = _generate_random_points(rng)
    
    return best_points
# EVOLVE-BLOCK-END
