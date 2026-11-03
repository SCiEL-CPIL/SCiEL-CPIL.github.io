"""Rewritten program: Hex init + Simulated Annealing + L-BFGS-B Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
DEADLINE = 140.0  # seconds, safety margin before 150s limit


def _hex_lattice_init():
    """Generate a hexagonal lattice of points inside the unit disk, return N closest to center."""
    best_points = None
    for spacing in [0.10, 0.102, 0.104, 0.106, 0.108, 0.11, 0.112, 0.114]:
        k = int(1.0 / spacing) + 4
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 0.98:
                    points.append((x, y))
        points = np.array(points)
        if len(points) >= N:
            radii = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(radii, kind='stable')[:N]
            pts = points[idx]
            if best_points is None:
                best_points = pts
    
    if best_points is None:
        best_points = np.random.randn(N, 2) * 0.5
    
    return best_points[:N]


def _compute_radius(points):
    """Compute the minimum radius: min of half min distance and min distance to boundary."""
    n = len(points)
    # Efficient min distance calculation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - radii)
    return min(min_dist / 2.0, min_boundary)


def _sa_optimize(points, max_time=60.0):
    """
    Simulated Annealing to maximize the minimum radius.
    State: points (N, 2)
    Score: r = min(min_pair_dist/2, min_boundary_dist)
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _compute_radius(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Initial temperature
    T = 0.01
    cooling_rate = 0.9995
    step_size = 0.05
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < max_time:
        # Generate a new state
        new_points = current_points.copy()
        idx = rng.integers(0, N)
        move = rng.normal(0, step_size, size=2)
        new_points[idx] += move
        
        # Ensure new point is inside the disk (hard constraint for SA moves to avoid huge penalties)
        # If it goes outside, we can either reject or reflect. Rejecting is simpler but might get stuck.
        # Let's reflect or clamp. Clamping to boundary is better.
        r_new = np.hypot(new_points[idx, 0], new_points[idx, 1])
        if r_new > 0.99:
            new_points[idx] *= 0.99 / r_new
            
        new_score = _compute_radius(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Accept with probability exp((new - current) / T)
            delta = new_score - current_score
            # If delta is very negative, prob is 0
            if delta > -1e-6: # Only calculate exp if it's not too bad
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
            
        # Cool down
        T *= cooling_rate
        # Adjust step size if we are stuck? 
        # If no improvement for a long time, we might want to increase T (reheat) or step size.
        # For now, simple cooling.
        
        if T < 1e-5:
            break

    return best_points, best_score


def _polish(points, max_time=30.0):
    """
    Local polish using L-BFGS-B on the epigraph formulation.
    This refines the solution found by SA.
    """
    n = N
    start_time = time.time()
    
    # Initial t estimate
    r_est = _compute_radius(points)
    t0 = max(r_est * 0.99, 0.01)
    
    x0 = np.concatenate([points.flatten(), [t0]])
    
    best_points = points.copy()
    best_r = r_est
    
    def objective_and_grad(x):
        if time.time() - start_time > max_time:
            raise TimeoutError()
        
        pts = x[:2*n].reshape(n, 2)
        t = x[2*n]
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Constraint violations
        pair_viol = 2*t - dists
        radii = np.hypot(pts[:, 0], pts[:, 1])
        bound_viol = radii + t - 1.0
        
        # Soft penalty
        eps = 1e-6
        soft_pair = (pair_viol + np.sqrt(pair_viol**2 + eps)) / 2.0
        soft_bound = (bound_viol + np.sqrt(bound_viol**2 + eps)) / 2.0
        
        penalty = np.sum(soft_pair**2) + np.sum(soft_bound**2)
        val = -t + penalty
        
        # Gradient
        sqrt_pair = np.sqrt(pair_viol**2 + eps)
        sig_pair = (sqrt_pair + pair_viol) / (2.0 * sqrt_pair)
        
        sqrt_bound = np.sqrt(bound_viol**2 + eps)
        sig_bound = (sqrt_bound + bound_viol) / (2.0 * sqrt_bound)
        
        grad = np.zeros_like(x)
        
        # d/dt
        d_obj_dt = -1.0 + 4.0 * np.sum(soft_pair * sig_pair) + 2.0 * np.sum(soft_bound * sig_bound)
        grad[2*n] = d_obj_dt
        
        # d/dpoints
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_ij = dists[i, j]
                if d_ij < 1e-10:
                    continue
                dir_ij = (pts[i] - pts[j]) / d_ij
                grad[2*i] += -2.0 * soft_pair[i, j] * sig_pair[i, j] * dir_ij[0]
                grad[2*i+1] += -2.0 * soft_pair[i, j] * sig_pair[i, j] * dir_ij[1]
            
            r_i = radii[i]
            if r_i > 1e-10:
                dir_bound = pts[i] / r_i
                grad[2*i] += 2.0 * soft_bound[i] * sig_bound[i] * dir_bound[0]
                grad[2*i+1] += 2.0 * soft_bound[i] * sig_bound[i] * dir_bound[1]
        
        return val, grad

    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': 500,
                'maxfun': 1000,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 50
            }
        )
        
        x_opt = result.x
        pts_opt = x_opt[:2*n].reshape(n, 2)
        r_opt = _compute_radius(pts_opt)
        if r_opt > best_r:
            best_r = r_opt
            best_points = pts_opt.copy()
            
    except (TimeoutError, Exception) as e:
        pass
    
    return best_points, best_r


def construct():
    """Construct 300 circle centers in the unit disk to maximize minimum radius."""
    start_time = time.time()
    
    # Step 1: Initialize with hexagonal lattice
    points = _hex_lattice_init()
    
    # Step 2: Simulated Annealing for global exploration
    sa_time = 60.0
    if time.time() - start_time + sa_time < DEADLINE:
        points, best_r = _sa_optimize(points, max_time=sa_time)
    else:
        best_r = _compute_radius(points)
        
    # Step 3: Local Polish
    remaining = DEADLINE - (time.time() - start_time) - 5.0
    if remaining > 10.0:
        points, best_r = _polish(points, max_time=remaining)
    
    # Step 4: Final verification and cleanup
    radii = np.hypot(points[:, 0], points[:, 1])
    mask = radii > 0.9999
    if np.any(mask):
        points[mask] *= 0.9999 / radii[mask]
    
    # Ensure exactly N points
    if len(points) != N:
        if len(points) < N:
            while len(points) < N:
                p = np.random.randn(2) * 0.5
                if np.hypot(p[0], p[1]) < 0.9:
                    points = np.vstack([points, p])
        else:
            points = points[:N]
    
    return points[:N]
