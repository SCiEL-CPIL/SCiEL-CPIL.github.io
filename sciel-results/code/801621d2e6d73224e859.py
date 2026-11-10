"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 150

# EVOLVE-BLOCK-START
def _fibonacci_sphere(n, offset=0.0):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on the sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|)
    Gradient is projected onto the tangent space of the sphere.
    """
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    np.fill_diagonal(dist, 1.0)  # avoid log(0)
    
    # Energy: sum over i<j of log(1/dist)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            energy += math.log(1.0 / dist[i, j])
    
    # Gradient computation (vectorized)
    # For each pair (i,j), the gradient of log(1/|x_i - x_j|) w.r.t. x_i is (x_i - x_j)/|x_i - x_j|^2
    # We need to sum over all j != i for each i
    grad = np.zeros_like(pts)
    for i in range(n):
        # diff[i, j] = pts[i] - pts[j]
        # d/dx_i log(1/|x_i - x_j|) = (x_i - x_j) / |x_i - x_j|^2
        for j in range(n):
            if i == j:
                continue
            d = dist[i, j]
            grad[i] += diff[i, j] / (d * d)
    
    # Project gradient onto tangent space: grad_tan = grad - (grad · x) x
    grad_dot_x = np.sum(grad * pts, axis=1)  # (n,)
    grad = grad - np.outer(grad_dot_x, pts)
    
    return energy, grad


def _lbfgs_optimize(pts, max_time=100.0):
    """Optimize points on the sphere using L-BFGS-B with parameterization via angles."""
    from scipy.optimize import minimize
    
    # Convert to spherical coordinates (theta, phi)
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))  # polar angle
    phi = np.arctan2(pts[:, 1], pts[:, 0])  # azimuthal angle
    
    def params_to_points(theta, phi):
        r = np.sqrt(1.0 - np.cos(theta) ** 2)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), np.cos(theta)])
    
    def objective(params):
        theta = params[:N]
        phi = params[N:]
        p = params_to_points(theta, phi)
        e, _ = _energy_and_grad(p)
        return e
    
    # Use L-BFGS-B
    x0 = np.concatenate([theta, phi])
    
    start_time = time.time()
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 5000, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
    )
    
    elapsed = time.time() - start_time
    if result.success or elapsed < max_time:
        best_params = result.x
    else:
        best_params = x0
    
    theta_opt = best_params[:N]
    phi_opt = best_params[N:]
    return params_to_points(theta_opt, phi_opt)


def _simulated_annealing(pts, max_time=80.0, T_start=0.01, T_end=0.0001, 
                          n_iter_per_temp=50, cooling=0.995):
    """Simple simulated annealing on the sphere."""
    current_pts = pts.copy()
    current_energy, _ = _energy_and_grad(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    T = T_start
    start_time = time.time()
    
    while time.time() - start_time < max_time and T > T_end:
        # Propose a move: randomly perturb one point
        idx = np.random.randint(N)
        # Random direction on tangent plane
        random_dir = np.random.randn(3)
        random_dir -= np.dot(random_dir, current_pts[idx]) * current_pts[idx]
        random_dir /= np.linalg.norm(random_dir)
        
        step_size = 0.01 * T / T_start + 0.001
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + step_size * random_dir
        new_pts[idx] /= np.linalg.norm(new_pts[idx])
        
        new_energy, _ = _energy_and_grad(new_pts)
        
        # Acceptance criterion
        if new_energy < current_energy or np.random.random() < math.exp(-(new_energy - current_energy) / T):
            current_pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
        
        T *= cooling
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s limit
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    
    # First pass: L-BFGS optimization
    if time.time() - start_time < deadline - 60:
        pts = _lbfgs_optimize(pts, max_time=min(60.0, deadline - (time.time() - start_time)))
    
    # Second pass: Simulated annealing to escape local minima
    if time.time() - start_time < deadline - 20:
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            pts_sa = _simulated_annealing(pts, max_time=min(remaining - 5, 80.0))
            e_sa, _ = _energy_and_grad(pts_sa)
            e_curr, _ = _energy_and_grad(pts)
            if e_sa < e_curr:
                pts = pts_sa
    
    # Final polish: L-BFGS again if time allows
    if time.time() - start_time < deadline - 10:
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            pts = _lbfgs_optimize(pts, max_time=min(remaining - 5, 50.0))
    
    # Ensure valid output: normalize and check for near-duplicates
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Check for points closer than 1e-9
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                # Separate them slightly
                pts[j] = pts[j] + 1e-8 * np.random.randn(3)
                pts[j] /= np.linalg.norm(pts[j])
    
    return pts
# EVOLVE-BLOCK-END
