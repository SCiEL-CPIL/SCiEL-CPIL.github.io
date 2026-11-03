"""Pack 50 equal circles in the unit disk via ring-based parametric optimization + local refinement."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 140.0  # safety margin before 150s limit


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    n = len(pts)
    # Container constraint
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_container = np.min(container_margin)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_container, min_pair)


def _ring_positions(ring_params):
    """
    Convert ring parameters to point positions.
    ring_params: array of shape (n_rings, 4) where each row is [radius, n_points, angle_offset, angular_jitter]
    Actually, let's use a simpler parameterization:
    We'll define rings with (radius, n_points, phase)
    """
    points = []
    for i in range(len(ring_params)):
        radius = ring_params[i, 0]
        n_pts = int(round(ring_params[i, 1]))
        phase = ring_params[i, 2]
        if n_pts <= 0:
            continue
        if n_pts == 1:
            angles = np.array([phase])
        else:
            angles = phase + 2 * np.pi * np.arange(n_pts) / n_pts
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        points.append(np.column_stack([xs, ys]))
    if not points:
        return np.zeros((1, 2))
    return np.vstack(points)


def _parametric_energy(params, r_target):
    """
    Energy function for parametric ring optimization.
    params: flat array of ring parameters
    We want to maximize the minimum radius.
    """
    n_rings = len(params) // 3
    ring_params = params.reshape(n_rings, 3)
    pts = _ring_positions(ring_params)
    
    # If we don't have exactly N points, penalize heavily
    if len(pts) != N:
        return 1e6 + abs(len(pts) - N)
    
    score = _compute_score(pts)
    # We want to maximize score, so minimize (r_target - score) if score < r_target
    if score >= r_target:
        return 0.0
    else:
        return (r_target - score) ** 2 + 0.01 * (r_target - score)


def _local_refine(pts, max_time=5.0):
    """Local refinement using L-BFGS-B on the smooth energy."""
    n = len(pts)
    t_start = time.time()
    
    # Estimate current radius
    r_current = _compute_score(pts)
    
    # Try to improve by a small amount
    r_target = r_current * 1.002
    
    def objective(flat):
        p = flat.reshape(n, 2)
        # Container violation
        container_margin = 1.0 - np.sqrt(np.sum(p ** 2, axis=1))
        cont_viol = np.maximum(0, r_target - container_margin)
        cont_energy = np.sum(cont_viol ** 2)
        
        # Pairwise violation
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_viol = np.maximum(0, 2 * r_target - dist)
        pair_energy = np.sum(pair_viol ** 2)
        
        return pair_energy + cont_energy
    
    def gradient(flat):
        p = flat.reshape(n, 2)
        grad = np.zeros_like(flat)
        
        # Container gradient
        norms = np.sqrt(np.sum(p ** 2, axis=1))
        norms_safe = np.where(norms > 1e-10, norms, 1e-10)
        cont_margin = 1.0 - norms
        cont_viol = np.maximum(0, r_target - cont_margin)
        # d/dp_i of (r_target - (1 - |p_i|))^2 when violated
        # = 2*(r_target - 1 + |p_i|) * d|p_i|/dp_i
        # = 2*(r_target - 1 + |p_i|) * p_i / |p_i|
        for i in range(n):
            if cont_viol[i] > 0:
                factor = 2 * (r_target - 1 + norms[i]) / norms_safe[i]
                grad[2*i] += factor * p[i, 0]
                grad[2*i+1] += factor * p[i, 1]
        
        # Pairwise gradient
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_viol = np.maximum(0, 2 * r_target - dist)
        
        # For each pair (i,j) with violation, gradient on i:
        # d/dp_i of (2*r_target - |p_i - p_j|)^2
        # = -2*(2*r_target - |p_i - p_j|) * (p_i - p_j) / |p_i - p_j|
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if pair_viol[i, j] > 0:
                    d = diff[i, j]
                    dist_ij = dist[i, j]
                    if dist_ij > 1e-10:
                        factor = -2 * pair_viol[i, j] / dist_ij
                        grad[2*i] += factor * d[0]
                        grad[2*i+1] += factor * d[1]
        
        return grad
    
    best_pts = pts.copy()
    best_score = _compute_score(best_pts)
    
    # Try a few different target radii
    for r_mult in [1.001, 1.002, 1.005]:
        if time.time() - t_start > max_time:
            break
        r_target = best_score * r_mult
        try:
            result = minimize(
                objective,
                best_pts.flatten(),
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            new_pts = result.x.reshape(n, 2)
            new_score = _compute_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
        except Exception:
            pass
    
    return best_pts, best_score


def construct():
    t0 = time.time()
    
    # Strategy 1: Ring-based parametric optimization with differential evolution
    # Try different ring configurations
    
    best_pts = None
    best_score = 0.0
    
    # Define a set of ring configurations to try
    # Each config: list of (radius, n_points, phase)
    # We'll use differential evolution to optimize these parameters
    
    def _de_ring_search(n_rings, max_time=30.0):
        """Use differential evolution to find optimal ring parameters."""
        t_start = time.time()
        
        # Parameters: for each ring, [radius, n_points (continuous, will round), phase]
        # Radius: [0, 0.95]
        # n_points: [1, 20] (continuous, will round to int)
        # phase: [0, 2*pi]
        
        bounds = []
        for i in range(n_rings):
            bounds.append((0.01, 0.95))  # radius
            bounds.append((1.0, 20.0))   # n_points
            bounds.append((0.0, 2 * np.pi))  # phase
        
        # Objective function for DE
        def de_objective(params):
            if time.time() - t_start > max_time:
                return 1e6
            ring_params = params.reshape(n_rings, 3)
            pts = _ring_positions(ring_params)
            if len(pts) != N:
                return 1e6 + abs(len(pts) - N)
            score = _compute_score(pts)
            return -score  # Minimize negative score
        
        try:
            result = differential_evolution(
                de_objective,
                bounds,
                maxiter=100,
                tol=1e-6,
                popsize=15,
                seed=42,
                polish=False,
                init='latinhypercube'
            )
            
            if result.fun < 1e5:  # Valid result
                ring_params = result.x.reshape(n_rings, 3)
                pts = _ring_positions(ring_params)
                if len(pts) == N:
                    score = _compute_score(pts)
                    return pts, score
        except Exception:
            pass
        
        return None, 0.0
    
    # Try different numbers of rings
    for n_rings in [3, 4, 5, 6, 7]:
        if time.time() - t0 > DEADLINE - 40:
            break
        
        pts, score = _de_ring_search(n_rings, max_time=25.0)
        if pts is not None and score > best_score:
            best_pts = pts
            best_score = score
    
    # If we didn't find a good solution, use a hexagonal fallback
    if best_pts is None:
        # Simple hexagonal initialization
        rng = np.random.default_rng(42)
        spacing = 0.25
        k = int(1.0 / spacing) + 3
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                    rows.append((x, y))
        pts_hex = np.array(rows)
        if len(pts_hex) >= N:
            idx = rng.choice(len(pts_hex), size=N, replace=False)
            best_pts = pts_hex[idx].copy()
            best_score = _compute_score(best_pts)
        else:
            # Fallback to random
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0, 0.9, N)
            best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            best_score = _compute_score(best_pts)
    
    # Local refinement
    if time.time() - t0 < DEADLINE - 20:
        best_pts, best_score = _local_refine(best_pts, max_time=15.0)
    
    # Another round of local refinement with a slightly higher target
    if time.time() - t0 < DEADLINE - 10:
        best_pts, best_score = _local_refine(best_pts, max_time=10.0)
    
    # Validate
    radii = np.sqrt(np.sum(best_pts ** 2, axis=1))
    if np.any(~np.isfinite(best_pts)) or np.any(radii >= 1.0):
        # Fallback
        rng = np.random.default_rng(42)
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0, 0.9, N)
        best_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    return best_pts
