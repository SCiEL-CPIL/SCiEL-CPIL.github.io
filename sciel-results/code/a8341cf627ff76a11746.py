"""Rewritten program for logarithmic energy minimization of 100 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
DEADLINE_SECONDS = 118.0  # Safety margin before 120s kill
EPS = 1e-12

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its Euclidean gradient.
    
    Energy E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)
    Gradient: dE/dp_i = sum_{j!=i} (p_j - p_i) / |p_i - p_j|^2
    """
    # Pairwise differences: (N, N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    # Squared distances
    dist2 = np.sum(diff * diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    # Safe distances
    dist = np.sqrt(np.where(mask, np.maximum(dist2, EPS), 1.0))
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    log_terms = np.where(mask, -np.log(dist), 0.0)
    energy = 0.5 * np.sum(log_terms)
    
    # Gradient computation
    inv_dist2 = np.where(mask, 1.0 / np.maximum(dist2, EPS), 0.0)
    
    # grad[i] = sum_j (pts[j] - pts[i]) * inv_dist2[i, j]
    weight_sums = np.sum(inv_dist2, axis=1, keepdims=True)
    weighted_pts = inv_dist2 @ pts
    
    grad = weighted_pts - weight_sums * pts
    
    return energy, grad

def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts

def _retract(pts, step):
    """Move points along step and project back to sphere."""
    new_pts = pts + step
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    norms = np.where(norms < EPS, 1.0, norms)
    return new_pts / norms

def _icosahedron_subdivide(n_points):
    """
    Generate points on sphere by subdividing an icosahedron.
    This provides a more uniform initial distribution than Fibonacci spiral.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Icosahedron edges
    edges = [
        (0, 1), (0, 5), (0, 11), (0, 7), (0, 10),
        (1, 5), (5, 11), (11, 7), (7, 10), (10, 1),
        (1, 5), (5, 11), (11, 7), (7, 10), (10, 1),
        (3, 2), (3, 4), (3, 9), (3, 6), (3, 8),
        (2, 4), (4, 9), (9, 6), (6, 8), (8, 2),
        (4, 0), (9, 5), (6, 11), (8, 7), (2, 10)
    ]
    
    # Subdivide edges
    def subdivide_edge(p1, p2, level):
        if level == 0:
            return [p1]
        mid = (p1 + p2) / 2
        mid = mid / np.linalg.norm(mid)
        return subdivide_edge(p1, mid, level - 1) + subdivide_edge(mid, p2, level - 1)
    
    # Determine subdivision level
    # Icosahedron has 20 faces, each face has 3 edges
    # After subdivision, number of points increases
    level = 0
    while True:
        # Estimate number of points after subdivision
        # Each edge is split into 2^level segments
        # Total edges in icosahedron: 30
        # Points per edge after subdivision: 2^level
        # Total points: 12 + 30 * (2^level - 1)
        n_est = 12 + 30 * (2**level - 1)
        if n_est >= n_points:
            break
        level += 1
    
    # Generate points by subdividing edges
    all_points = list(vertices)
    for i, j in edges:
        points_on_edge = subdivide_edge(vertices[i], vertices[j], level)
        for p in points_on_edge:
            # Check if point already exists
            exists = False
            for existing in all_points:
                if np.linalg.norm(p - existing) < 1e-6:
                    exists = True
                    break
            if not exists:
                all_points.append(p)
    
    # Convert to array and normalize
    all_points = np.array(all_points)
    all_points = all_points / np.linalg.norm(all_points, axis=1, keepdims=True)
    
    # If we have more points than needed, randomly select
    if len(all_points) > n_points:
        indices = np.random.choice(len(all_points), n_points, replace=False)
        all_points = all_points[indices]
    
    # If we have fewer points than needed, add random points
    if len(all_points) < n_points:
        remaining = n_points - len(all_points)
        rng = np.random.RandomState(42)
        random_pts = rng.randn(remaining, 3)
        random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True)
        all_points = np.vstack([all_points, random_pts])
    
    return all_points[:n_points]

def _fibonacci_spiral(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _random_points(n, seed=42):
    """Generate n random points on sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _perturbed_points(base_pts, sigma, seed=0):
    """Perturb base points by adding Gaussian noise and reprojecting."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*base_pts.shape) * sigma
    pts = base_pts + noise
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _riemannian_optimize(pts0, max_iters=500, tol=1e-10, time_limit=None, adaptive=True):
    """
    Riemannian Gradient Descent with adaptive step size and backtracking.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Check convergence
        grad_norm = np.linalg.norm(g_t)
        if grad_norm < tol:
            break
            
        # Adaptive learning rate
        if adaptive:
            lr = min(0.1, max(0.001, 0.01 / (1.0 + 0.01 * it)))
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(25):
            new_pts = _retract(pts, -alpha * g_t)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * g_t):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            # Try smaller step
            alpha = max(0.001, lr * 0.1)
            new_pts = _retract(pts, -alpha * g_t)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                pts = new_pts
                energy = new_energy
            else:
                break
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
    return best_pts, best_energy

def _rcg_polish(pts0, max_iters=300, tol=1e-12, time_limit=None):
    """
    Riemannian Conjugate Gradient for faster convergence.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    # Initial search direction
    d_t = -g_t.copy()
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        if np.linalg.norm(g_t) < tol:
            break
        
        # Line search along d_t
        alpha = 0.01
        improved = False
        for _ in range(30):
            new_pts = _retract(pts, alpha * d_t)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * d_t):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            break
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t_new = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Polak-Ribiere beta
        dot_new = np.sum(g_t_new * g_t)
        dot_old = np.sum(g_t * g_t)
        if dot_old > 1e-16:
            beta = max(0.0, dot_new / dot_old)
        else:
            beta = 0.0
        
        # Update search direction
        d_t = -g_t_new + beta * d_t
        # Ensure descent direction
        if np.sum(g_t_new * d_t) > 0:
            d_t = -g_t_new
        
        g_t = g_t_new
    
    return best_pts, best_energy

def _polish_lbfgs(pts, max_iter=300, gtol=1e-14):
    """Polish using L-BFGS-B with tight tolerances."""
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
    
    x0 = pts.ravel()
    try:
        res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': gtol, 'ftol': 1e-16})
        if res.success or np.isfinite(res.fun):
            x_opt = res.x.reshape(N, 3)
            x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
            e_opt, _ = _energy_and_grad(x_opt)
            if np.isfinite(e_opt):
                return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]

def _simulated_annealing(pts, max_iters=100, time_limit=None):
    """
    Simulated annealing to escape local minima.
    """
    pts = pts.copy()
    energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    start_time = time.time()
    T = 0.1  # Initial temperature
    
    for it in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Generate perturbation
        rng = np.random.RandomState(it)
        noise = rng.randn(*pts.shape) * 0.05
        new_pts = pts + noise
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        new_energy, _ = _energy_and_grad(new_pts)
        
        # Accept if better or with probability exp(-delta_E / T)
        if new_energy < energy or np.random.random() < np.exp(-(new_energy - energy) / T):
            pts = new_pts
            energy = new_energy
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        # Cool down
        T *= 0.95
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    
    # Generate multiple diverse initializations
    inits = []
    
    # 1. Icosahedron subdivision (better initial distribution)
    inits.append(('ico', _icosahedron_subdivide(N)))
    
    # 2. Fibonacci spiral
    inits.append(('fib', _fibonacci_spiral(N)))
    
    # 3. Random with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7]:
        inits.append((f'rand{seed}', _random_points(N, seed=seed)))
    
    # 4. Perturbed icosahedron
    ico_base = _icosahedron_subdivide(N)
    for sigma, seed in [(0.1, 10), (0.05, 11), (0.2, 12)]:
        inits.append((f'ico_pert{sigma}_{seed}', _perturbed_points(ico_base, sigma, seed=seed)))
    
    # 5. Perturbed Fibonacci
    fib_base = _fibonacci_spiral(N)
    for sigma, seed in [(0.1, 13), (0.05, 14)]:
        inits.append((f'fib_pert{sigma}_{seed}', _perturbed_points(fib_base, sigma, seed=seed)))
    
    best_pts = None
    best_energy = np.inf
    
    # Optimize each initialization
    for name, init_pts in inits:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 25:
            break
        
        # Stage 1: Riemannian Gradient Descent
        pts_opt, energy_opt = _riemannian_optimize(init_pts, max_iters=400, time_limit=15)
        
        # Stage 2: RCG Polish
        if time.time() - start_time < DEADLINE_SECONDS - 20:
            pts_opt, energy_opt = _rcg_polish(pts_opt, max_iters=300, time_limit=15)
        
        # Stage 3: Simulated Annealing to escape local minima
        if time.time() - start_time < DEADLINE_SECONDS - 15:
            pts_sa, energy_sa = _simulated_annealing(pts_opt, max_iters=50, time_limit=10)
            if energy_sa < energy_opt:
                pts_opt, energy_opt = pts_sa, energy_sa
                # Re-optimize after SA
                pts_opt, energy_opt = _riemannian_optimize(pts_opt, max_iters=200, time_limit=10)
                pts_opt, energy_opt = _rcg_polish(pts_opt, max_iters=200, time_limit=10)
        
        # Stage 4: L-BFGS-B Polish
        if time.time() - start_time < DEADLINE_SECONDS - 10:
            pts_opt, energy_opt = _polish_lbfgs(pts_opt, max_iter=200, gtol=1e-13)
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Final high-precision polish on the best result
    if best_pts is not None and time.time() - start_time < DEADLINE_SECONDS - 10:
        best_pts, best_energy = _rcg_polish(best_pts, max_iters=400, time_limit=20)
    
    if best_pts is not None and time.time() - start_time < DEADLINE_SECONDS - 5:
        best_pts, best_energy = _polish_lbfgs(best_pts, max_iter=300, gtol=1e-15)
    
    # Final Check and Return
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    return best_pts
