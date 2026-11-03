"""Logarithmic energy minimization for N points on the unit sphere.
Uses Basin Hopping global search combined with Riemannian local optimization."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_SAFETY = 10.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-16))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)
    
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _local_minimize(pts, deadline, max_iter=10000):
    """
    Hybrid local minimizer:
    1. Quick Riemannian Gradient Descent to get close to a minimum.
    2. Scipy L-BFGS-B for high-precision polishing.
    """
    current = _project(pts.copy())
    
    # Phase 1: Riemannian GD
    e, g = _energy_grad(current)
    lr = 0.05
    for it in range(500):
        if time.time() > deadline - 5.0:
            break
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-10:
            break
        
        # Simple backtracking
        step = lr
        new_e = float('inf')
        found = False
        for bt in range(10):
            v = -step * g
            v_norms = np.linalg.norm(v, axis=1, keepdims=True)
            v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_unit = v / v_norms_safe
            cos_t = np.cos(v_norms)
            sin_t = np.sin(v_norms)
            new_pts = current * cos_t + v_unit * sin_t
            new_pts = _project(new_pts)
            
            new_e = _energy(new_pts)
            if new_e < e:
                found = True
                break
            step *= 0.5
        
        if found:
            current = new_pts
            e = new_e
            _, g = _energy_grad(current)
            lr = min(step * 1.5, 0.1)
        else:
            break

    # Phase 2: Scipy L-BFGS-B
    def obj(x):
        p = x.reshape(N, 3)
        p = _project(p)
        return _energy(p)
    
    def grad(x):
        p = x.reshape(N, 3)
        p = _project(p)
        e, g = _energy_grad(p)
        return g.flatten()
        
    try:
        x0 = current.flatten()
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxls': 50})
        final_pts = _project(res.x.reshape(N, 3))
        final_e = _energy(final_pts)
        
        if final_e < e:
            return final_pts, final_e
    except Exception:
        pass
        
    return current, e


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin for overhead and final checks
    
    rng = np.random.default_rng(42)
    
    # Initial configuration: Fibonacci spiral is a good starting point
    best_pts = _fibonacci_spiral(N, offset=0.5)
    best_energy = _energy(best_pts)
    
    # Basin Hopping Parameters
    # We use a cooling schedule for the perturbation size
    T_init = 0.5
    T_min = 0.01
    max_hops = 200  # Number of basin hopping iterations
    
    # We will run a few independent basin hopping chains to increase robustness
    # but primarily rely on one strong chain with adaptive step size.
    
    current_pts = best_pts.copy()
    current_energy = best_energy
    
    for hop in range(max_hops):
        if time.time() > deadline - 10.0:
            break
            
        # 1. Local Minimization of current state
        local_pts, local_e = _local_minimize(current_pts, deadline)
        
        if local_e < best_energy:
            best_energy = local_e
            best_pts = local_pts.copy()
        
        # If we have converged tightly, we might need a bigger kick
        # Check gradient norm at the local minimum
        _, g_local = _energy_grad(local_pts)
        g_norm = np.linalg.norm(g_local)
        
        # 2. Perturbation
        # Perturbation size adapts based on temperature
        T = T_init * (T_min / T_init) ** (hop / max_hops)
        
        # Random Gaussian noise
        noise = rng.standard_normal((N, 3))
        
        # Project noise to tangent space of local_pts
        dot_n = np.sum(noise * local_pts, axis=1, keepdims=True)
        noise = noise - dot_n * local_pts
        
        # Scale noise
        # We want the step size to be related to T
        # Typical step size for basin hopping is sqrt(2*T) * dt, but here we use a fixed scale factor
        # adjusted by T
        step_scale = 0.2 * T + 0.01
        
        perturbed_pts = local_pts + step_scale * noise
        perturbed_pts = _project(perturbed_pts)
        
        # 3. Acceptance Criterion (Metropolis-like)
        # We always accept if energy decreases.
        # If energy increases, we accept with probability exp(-(E_new - E_old)/T)
        # But since we are minimizing, and T is cooling, we can be stricter.
        # Actually, standard Basin Hopping accepts the perturbed state as the *start* for the next
        # local minimization, regardless of energy, provided it's not too far.
        # The "acceptance" is implicit in the local minimization step:
        # We minimize the perturbed state. If the resulting minimum is better, we keep it.
        # If it's worse, we might revert to the previous best?
        # Standard BH:
        # x_{k+1} = Minimize( x_k + delta )
        # We don't reject x_{k+1} if it's worse than x_k, we just continue.
        # This allows escaping local minima.
        
        current_pts = perturbed_pts
        
    # Final Polish on the best found configuration
    if time.time() < deadline - 5.0:
        final_pts, final_e = _local_minimize(best_pts, deadline, max_iter=20000)
        if final_e < best_energy:
            best_energy = final_e
            best_pts = final_pts

    # Final Validation and Normalization
    best_pts = _project(best_pts)
    
    # Check for validity (collisions)
    d2 = np.einsum('ij,ij->i', best_pts, best_pts)[:, None] + np.einsum('ij,ij->i', best_pts, best_pts)[None, :] \
         - 2.0 * (best_pts @ best_pts.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        # Fallback to a very well-separated configuration if collision occurs
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)

    return best_pts
