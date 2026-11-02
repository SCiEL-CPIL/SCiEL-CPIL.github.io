"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute log energy sum_{i<j} log(1/|xi-xj|) using math.fsum."""
    import math
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _log_energy_fast(pts):
    """Fast vectorized log energy for internal optimization (not used for scoring)."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist, 1.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _grad_log_energy(pts):
    """Gradient of log energy: d/dx_i sum_{j != i} log(1/|x_i - x_j|) = sum_{j != i} (x_i - x_j)/|x_i - x_j|^2."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_vec = pts[i] - pts[j]
            d2 = np.dot(d_vec, d_vec)
            if d2 < 1e-30:
                continue
            grad[i] += d_vec / d2
    return grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=None):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(start_pts, max_iters=5000, lr_init=0.01, lr_min=1e-6, tol=1e-10):
    """
    Optimize using projected gradient descent with adaptive step size.
    Uses L-BFGS-like momentum for faster convergence.
    """
    pts = start_pts.copy()
    n = len(pts)
    
    # Initial energy
    e0 = _log_energy_fast(pts)
    
    # Momentum (simple Nesterov-like)
    v = np.zeros_like(pts)
    beta = 0.9
    
    # Adaptive learning rate
    lr = lr_init
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Compute gradient
        g = _grad_log_energy(pts)
        
        # Project gradient onto tangent space: g_t = g - (g·x)x
        g_t = g - np.sum(pts * g, axis=1, keepdims=True) * pts
        
        # Update with momentum
        v = beta * v + g_t
        pts_new = pts - lr * v
        pts_new = _project_to_sphere(pts_new)
        
        # Compute new energy
        e_new = _log_energy_fast(pts_new)
        
        # Adaptive step size
        if e_new < e0:
            # Accept
            pts = pts_new
            # Can increase lr slightly
            lr = min(lr * 1.05, 0.1)
            e0 = e_new
        else:
            # Reject, decrease lr
            lr *= 0.5
            if lr < lr_min:
                break
        
        # Check convergence
        if it > 100 and abs(e_new - e0) < tol:
            break
    
    return pts, e0


def _lbfgs_optimize(start_pts, max_iters=3000):
    """
    Use scipy's L-BFGS-B with a penalty method for sphere constraint.
    Actually, let's use a direct approach: parameterize via tangent space updates.
    """
    from scipy.optimize import minimize
    
    pts = start_pts.copy()
    n = len(pts)
    
    def objective_and_grad(x):
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e = _log_energy_fast(p)
        g = _grad_log_energy(p)
        # Project gradient
        g_t = g - np.sum(p * g, axis=1, keepdims=True) * p
        return e, g_t.flatten()
    
    # Use a custom L-BFGS that respects the manifold
    # Actually, let's just do projected gradient with more iterations
    
    best_pts = pts.copy()
    best_e = _log_energy_fast(pts)
    
    lr = 0.05
    v = np.zeros_like(pts)
    beta = 0.9
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        g = _grad_log_energy(pts)
        g_t = g - np.sum(pts * g, axis=1, keepdims=True) * pts
        
        v = beta * v + g_t
        pts_new = pts - lr * v
        pts_new = _project_to_sphere(pts_new)
        
        e_new = _log_energy_fast(pts_new)
        
        if e_new < best_e:
            best_pts = pts_new.copy()
            best_e = e_new
        
        if e_new < _log_energy_fast(pts):
            pts = pts_new
            lr = min(lr * 1.02, 0.1)
        else:
            lr *= 0.5
            if lr < 1e-7:
                break
        
        if it > 200 and lr < 1e-6:
            break
    
    return best_pts, best_e


def _anneal(pts, max_iters=2000):
    """
    Simulated annealing to escape local minima.
    """
    best_pts = pts.copy()
    best_e = _log_energy_fast(pts)
    n = len(pts)
    
    T = 0.1  # initial temperature
    T_min = 1e-6
    cooling = 0.999
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 5:
            break
        
        # Random perturbation in tangent space
        noise = rng.standard_normal((n, 3))
        # Project to tangent space
        noise_t = noise - np.sum(pts * noise, axis=1, keepdims=True) * pts
        # Scale by temperature
        step_size = np.sqrt(T * 2.0 / n)
        pts_new = pts + step_size * noise_t
        pts_new = _project_to_sphere(pts_new)
        
        e_new = _log_energy_fast(pts_new)
        
        # Metropolis criterion
        delta_e = e_new - _log_energy_fast(pts)
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, T_min)):
            pts = pts_new
            if e_new < best_e:
                best_pts = pts_new.copy()
                best_e = e_new
        
        T = max(T * cooling, T_min)
    
    return best_pts, best_e


def construct():
    """
    Construct 100 points on unit sphere minimizing logarithmic energy.
    Strategy: Multi-start with projected gradient descent + simulated annealing.
    """
    t_start = time.time()
    
    # Start with Fibonacci spiral (good initial configuration)
    best_pts = _fibonacci_spiral(N)
    best_e = _log_energy_fast(best_pts)
    
    # First: optimize from Fibonacci start
    if time.time() < DEADLINE - 30:
        opt_pts, opt_e = _lbfgs_optimize(best_pts, max_iters=3000)
        if opt_e < best_e:
            best_pts = opt_pts
            best_e = opt_e
    
    # Try a few random restarts if time permits
    n_restarts = 0
    for seed in [1, 7, 42, 123, 999]:
        if time.time() > DEADLINE - 25:
            break
        
        rand_pts = _random_points(N, seed=seed)
        # Quick optimization from random start
        if time.time() < DEADLINE - 15:
            opt_pts, opt_e = _lbfgs_optimize(rand_pts, max_iters=1500)
            if opt_e < best_e:
                best_pts = opt_pts
                best_e = opt_e
                n_restarts += 1
    
    # Final polish: run more iterations from best
    if time.time() < DEADLINE - 10:
        opt_pts, opt_e = _lbfgs_optimize(best_pts, max_iters=5000)
        if opt_e < best_e:
            best_pts = opt_pts
            best_e = opt_e
    
    # Annealing to escape local minima
    if time.time() < DEADLINE - 5:
        ann_pts, ann_e = _anneal(best_pts, max_iters=1000)
        if ann_e < best_e:
            best_pts = ann_pts
            best_e = ann_e
    
    # Final verification and normalization
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                best_pts[j] += 1e-8 * np.random.default_rng(0).standard_normal(3)
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts

# EVOLVE-BLOCK-END
