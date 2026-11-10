"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Uses dot-product trick for speed: |x_i - x_j|^2 = 2 - 2*x_i·x_j for unit vectors.
    """
    pts = pts.reshape(-1, 3)
    n = pts.shape[0]
    
    # Compute pairwise dot products
    dot = pts @ pts.T  # (n, n)
    
    # Squared distances: |x_i - x_j|^2 = 2 - 2*dot_ij for unit vectors
    diff_sq = 2.0 - 2.0 * dot
    # Clamp to avoid negative values from floating point
    diff_sq = np.maximum(diff_sq, 1e-15)
    
    # Mask diagonal
    np.fill_diagonal(diff_sq, np.inf)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d) = -0.5 * sum_{i!=j} log(d)
    # Using upper triangle for i<j
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(diff_sq[iu])
    energy = -np.sum(np.log(d))
    
    # Gradient: grad_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # inv_dist_sq[i,j] = 1 / |x_i - x_j|^2
    inv_dist_sq = np.zeros_like(diff_sq)
    np.divide(1.0, diff_sq, out=inv_dist_sq, where=diff_sq < np.inf)
    np.fill_diagonal(inv_dist_sq, 0)
    
    # grad_i = sum_j (x_j - x_i) * inv_dist_sq[i,j]
    # = (sum_j x_j * inv_dist_sq[i,j]) - x_i * sum_j inv_dist_sq[i,j]
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad = inv_dist_sq @ pts - pts * row_sums[:, np.newaxis]
    
    return energy, grad


def _project_and_normalize(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n=N):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n=N, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _simulated_annealing(pts, start_time, deadline, max_iter=2000, T0=0.1, T1=1e-6):
    """Simulated annealing to escape local minima."""
    best_pts = pts.copy()
    best_e = _energy_and_grad(best_pts)[0]
    cur_e = best_e
    cur_pts = best_pts.copy()
    n = N
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Random perturbation: pick a random subset of points and perturb them
        n_perturb = max(1, n // 20)
        idx = np.random.choice(n, size=n_perturb, replace=False)
        noise_scale = 0.05 * (T0 - (T0 - T1) * it / max_iter)
        
        new_pts = cur_pts.copy()
        for i in idx:
            noise = np.random.randn(3) * noise_scale
            new_pts[i] = cur_pts[i] + noise
            new_pts[i] /= np.linalg.norm(new_pts[i])
        
        new_e = _energy_and_grad(new_pts)[0]
        delta_e = new_e - cur_e
        
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / max(T1, 1e-10)):
            cur_pts = new_pts
            cur_e = new_e
            if cur_e < best_e:
                best_pts = cur_pts.copy()
                best_e = cur_e
    
    return best_pts, best_e


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Start from Fibonacci spiral
    x0 = _fibonacci_spiral(N).flatten()
    
    best_pts = x0.copy()
    best_energy = _energy_and_grad(best_pts.reshape(-1, 3))[0]
    
    def objective(x):
        pts = _project_and_normalize(x.reshape(-1, 3))
        return _energy_and_grad(pts)[0]
    
    def grad(x):
        pts = _project_and_normalize(x.reshape(-1, 3))
        e, g = _energy_and_grad(pts)
        # Project gradient onto tangent space
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return g_tangent.flatten()
    
    # Phase 1: L-BFGS-B with moderate tolerance
    try:
        res = minimize(
            objective, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
        )
        if time.time() < deadline:
            cand = _project_and_normalize(res.x.reshape(-1, 3)).flatten()
            cand_e = objective(cand)
            if cand_e < best_energy:
                best_pts = cand
                best_energy = cand_e
    except Exception:
        pass
    
    # Phase 2: Simulated annealing to escape local minima if time permits
    if time.time() < deadline - 60:
        try:
            sa_pts, sa_e = _simulated_annealing(
                best_pts.reshape(-1, 3), start_time, deadline,
                max_iter=1500, T0=0.05, T1=1e-7
            )
            if sa_e < best_energy:
                best_pts = sa_pts.flatten()
                best_energy = sa_e
        except Exception:
            pass
    
    # Phase 3: High-precision polish with tighter tolerances
    if time.time() < deadline - 20:
        try:
            x0 = best_pts.copy()
            res = minimize(
                objective, x0, jac=grad, method='L-BFGS-B',
                options={'maxiter': 15000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxfun': 25000}
            )
            if time.time() < deadline:
                cand = _project_and_normalize(res.x.reshape(-1, 3)).flatten()
                cand_e = objective(cand)
                if cand_e < best_energy:
                    best_pts = cand
                    best_energy = cand_e
        except Exception:
            pass
    
    # Phase 4: Final ultra-tight polish if time permits
    if time.time() < deadline - 5:
        try:
            x0 = best_pts.copy()
            res = minimize(
                objective, x0, jac=grad, method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 10000}
            )
            if time.time() < deadline:
                cand = _project_and_normalize(res.x.reshape(-1, 3)).flatten()
                cand_e = objective(cand)
                if cand_e < best_energy:
                    best_pts = cand
                    best_energy = cand_e
        except Exception:
            pass
    
    # Final check and return
    result = _project_and_normalize(best_pts.reshape(-1, 3))
    
    # Validate: check for near-duplicate points
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(result[i] - result[j])
            if d < 1e-9:
                # Add small perturbation
                result[i] += np.random.normal(0, 1e-8, 3)
                result[i] /= np.linalg.norm(result[i])
    
    # Ensure non-finite values are handled
    if not np.all(np.isfinite(result)):
        result = _fibonacci_spiral(N)
    
    return result
# EVOLVE-BLOCK-END
