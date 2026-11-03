"""Pack 50 equal circles in the unit disk via simulated annealing + L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 110.0  # safety margin before 120s limit


def _hex_init(n, seed=0):
    """Generate n points on a hexagonal grid inside the unit disk."""
    rng = np.random.default_rng(seed)
    spacing = 0.15
    best_points = None
    for _ in range(50):
        # Generate hex lattice
        k = int(1.0 / spacing) + 3
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= n:
            # Random subset
            idx = rng.choice(len(pts), size=n, replace=False)
            best_points = pts[idx].copy()
            break
        spacing *= 0.97
    if best_points is None:
        # Fallback: random points
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = rng.uniform(0, 0.9, n)
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return best_points


def _compute_score(pts):
    """Compute the packing radius: min of half-min-pair-dist and min-container-margin."""
    n = len(pts)
    # Container constraint
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    min_container = np.min(container_margin)
    
    # Pairwise distances
    # Use broadcasting for all pairs
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_container, min_pair)


def _smooth_energy(pts, r_target, beta=50.0):
    """
    Smooth surrogate for the constraint violation.
    We want to maximize r = min(min_pair_dist/2, min_container_margin).
    Equivalently, minimize the maximum violation when trying to achieve radius r_target.
    
    Energy = sum of soft violations:
      - For pairs: if dist_ij < 2*r_target, penalty
      - For container: if |c_i| + r_target > 1, penalty
    """
    n = len(pts)
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Soft penalty for pairs closer than 2*r_target
    pair_viol = np.maximum(0, 2 * r_target - dist)
    # Use log-sum-exp style soft sum
    pair_energy = np.sum(np.logaddexp(0, -beta * pair_viol)) / beta if beta > 0 else np.sum(pair_viol)
    
    # Container
    container_margin = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    cont_viol = np.maximum(0, r_target - container_margin)
    cont_energy = np.sum(np.logaddexp(0, -beta * cont_viol)) / beta if beta > 0 else np.sum(cont_viol)
    
    return pair_energy + cont_energy


def _lbfgs_polish(pts, r_target, maxiter=200):
    """Polish using L-BFGS-B to minimize smooth energy for a target radius."""
    n = len(pts)
    
    def objective(flat):
        p = flat.reshape(n, 2)
        return _smooth_energy(p, r_target, beta=100.0)
    
    def gradient(flat):
        p = flat.reshape(n, 2)
        # Numerical gradient (central difference)
        eps = 1e-6
        grad = np.zeros_like(flat)
        for i in range(2 * n):
            fp = flat.copy()
            fm = flat.copy()
            fp[i] += eps
            fm[i] -= eps
            grad[i] = (objective(fp) - objective(fm)) / (2 * eps)
        return grad
    
    result = minimize(
        objective,
        pts.flatten(),
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.x.reshape(n, 2)


def construct():
    t0 = time.time()
    best_pts = _hex_init(N, seed=42)
    best_score = _compute_score(best_pts)
    
    # Simulated annealing
    rng = np.random.default_rng(12345)
    
    # Start with a reasonable radius estimate
    r_current = best_score
    r_target = r_current * 1.001
    
    temperature = 0.01
    cooling = 0.9995
    step_size = 0.02
    
    n_sa_iters = 50000
    
    for it in range(n_sa_iters):
        if time.time() - t0 > DEADLINE - 30:
            break
        
        # Propose a move: pick a random point and perturb it
        idx = rng.integers(0, N)
        old_pos = best_pts[idx].copy()
        new_pos = old_pos + step_size * rng.normal(size=2)
        
        # Keep inside container (soft constraint)
        new_r = np.sqrt(new_pos[0] ** 2 + new_pos[1] ** 2)
        if new_r > 0.98:
            new_pos *= 0.98 / new_r
        
        # Compute current and new scores (approximate for speed)
        # Use a fast local score: only check pairs involving idx
        old_local = _compute_score(best_pts)
        
        test_pts = best_pts.copy()
        test_pts[idx] = new_pos
        new_local = _compute_score(test_pts)
        
        delta = new_local - old_local
        
        if delta > 0 or rng.random() < np.exp(delta / temperature):
            best_pts[idx] = new_pos
            if new_local > best_score:
                best_score = new_local
                r_target = best_score * 1.001
        
        temperature *= cooling
        step_size *= 0.9999
        if step_size < 0.001:
            step_size = 0.001
    
    # Final polish with L-BFGS-B
    if time.time() - t0 < DEADLINE - 10:
        r_target = best_score
        polished = _lbfgs_polish(best_pts, r_target, maxiter=100)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # One more polish round
    if time.time() - t0 < DEADLINE - 5:
        r_target = best_score * 1.0005
        polished = _lbfgs_polish(best_pts, r_target, maxiter=50)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_pts = polished
            best_score = polished_score
    
    # Validate: ensure all points are inside and finite
    radii = np.sqrt(np.sum(best_pts ** 2, axis=1))
    if np.any(~np.isfinite(best_pts)) or np.any(radii >= 1.0):
        # Fallback to hex init
        best_pts = _hex_init(N, seed=42)
    
    return best_pts
