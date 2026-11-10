"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # Safety margin before 150s kill

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _generate_fibonacci_samples(n_samples):
    """Generate well-distributed sample points on the sphere using Fibonacci spiral."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return samples

def _generate_initial_points(method='fibonacci', rng=None):
    """Generate initial point configurations using different methods."""
    if method == 'fibonacci':
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return _normalize(pts)
    elif method == 'random':
        if rng is None:
            rng = np.random.default_rng(42)
        pts = rng.normal(size=(N, 3))
        return _normalize(pts)
    else:
        raise ValueError(f"Unknown method: {method}")

def _covering_radius_sampled(pts, samples):
    """
    Fast estimate of covering radius using sample points.
    Returns value in degrees.
    """
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = dots.max(axis=1)
    min_dot = np.min(max_dots)
    radius_rad = np.arccos(np.clip(min_dot, -1.0, 1.0))
    return np.degrees(radius_rad)

def _soft_max_circumradius(pts, samples, temperature=0.01):
    """
    Differentiable soft-max surrogate for the covering radius.
    This focuses the gradient on the worst-case sample (largest distance).
    Returns the soft-max value in radians.
    """
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = dots.max(axis=1)
    
    # Soft-max over the distances (or rather, over -dots to get max distance)
    # We want to minimize the max distance, which is equivalent to maximizing min dot.
    # Let's use log-sum-exp on -dots to approximate max(-dots) = -min(dots).
    # Then covering radius ~ arccos(-min_dots).
    
    # To make it differentiable and focus on the worst case:
    # f(pts) = -log(sum(exp(temperature * dots))) 
    # As temperature -> inf, this approaches -max(dots) = min(-dots) = cos(max_angle).
    # We want to minimize max_angle, which is equivalent to maximizing min(dot).
    # So we minimize -min(dot).
    # Let's use: -logsumexp(temperature * dots) / temperature
    # This approximates -max(dots) for large temperature.
    # Wait, max(dots) corresponds to min(angle). We want to MINIMIZE max(angle).
    # max(angle) = arccos(min(dots)).
    # So we want to MAXIMIZE min(dots).
    # Equivalent to MINIMIZING -min(dots).
    
    # Soft-min of dots:
    # soft_min(dots) = -1/T * log(sum(exp(T * dots)))
    # As T -> inf, soft_min -> min(dots).
    # We want to maximize soft_min(dots).
    # So objective to minimize is: -soft_min(dots) = 1/T * log(sum(exp(T * dots)))
    
    # Let's compute soft_min directly.
    T = temperature
    scaled = T * dots
    max_scaled = scaled.max(axis=1, keepdims=True)
    # log(sum(exp(scaled))) = max_scaled + log(sum(exp(scaled - max_scaled)))
    log_sum_exp = max_scaled.squeeze() + np.log(np.sum(np.exp(scaled - max_scaled), axis=1))
    soft_min = -log_sum_exp / T
    
    # We want to maximize soft_min, so minimize -soft_min.
    # The value -soft_min approximates -min(dots) = cos(max_angle) is wrong.
    # min(dots) = cos(max_angle).
    # We want to minimize max_angle.
    # Minimizing max_angle is equivalent to maximizing min(dots).
    # So we minimize -min(dots).
    # -soft_min(dots) approximates -min(dots).
    # So objective = -soft_min(dots).
    
    return -soft_min

def _objective_and_gradient(pts_flat, samples, temperature):
    """
    Objective function and gradient for L-BFGS-B.
    pts_flat: 1D array of shape (3N,)
    """
    pts = pts_flat.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute soft-min of dots
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    
    T = temperature
    scaled = T * dots
    max_scaled = scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled, axis=1)
    
    # soft_min = -1/T * (max_scaled + log(sum_exp))
    # d(soft_min)/d(pts) = ?
    # Let's derive gradient of -soft_min.
    # soft_min = -1/T * logsumexp(T * dots)
    # d(soft_min)/d(dot_ij) = -1/T * (T * exp(T*dot_ij) / sum_exp) = -exp(T*dot_ij)/sum_exp
    # d(dot_ij)/d(p_kl) = sample_i_j * p_k_l (roughly, need chain rule)
    
    # Actually, let's just use numerical gradient or a simpler analytic one.
    # For L-BFGS-B, we can provide the gradient.
    # Objective f = -soft_min = 1/T * logsumexp(T * dots)
    # f = 1/T * (max_scaled + log(sum_exp))
    # df/d(dot_ij) = 1/T * (T * exp(T*dot_ij) / sum_exp) = exp(T*dot_ij) / sum_exp
    # Note: exp(T*dot_ij) / sum_exp is the softmax probability.
    
    probs = exp_scaled / sum_exp[:, np.newaxis]
    
    # d(f)/d(p_k) = sum_i df/d(dot_ik) * d(dot_ik)/d(p_k)
    # dot_ik = samples_i . p_k
    # d(dot_ik)/d(p_kl) = samples_il
    # So d(f)/d(p_kl) = sum_i probs[i, k] * samples_il
    
    # grad is shape (N, 3)
    grad = (probs.T @ samples) / T  # Wait, check scaling.
    # f = 1/T * logsumexp(T * dots)
    # d f / d dot = probs (as derived above, the 1/T and T cancel)
    # So d f / d p = probs.T @ samples
    
    # However, we are optimizing on the sphere. We need to project the gradient to the tangent space.
    # grad_tangent = grad - (grad . p) * p
    
    grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    
    return f_val, grad_tangent.flatten()

def _soft_max_obj(pts_flat, samples, temperature):
    pts = pts_flat.reshape(N, 3)
    pts = _normalize(pts)
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    T = temperature
    scaled = T * dots
    max_scaled = scaled.max(axis=1, keepdims=True)
    sum_exp = np.sum(np.exp(scaled - max_scaled), axis=1)
    log_sum_exp = max_scaled.squeeze() + np.log(sum_exp)
    soft_min = -log_sum_exp / T
    return -soft_min

def _soft_max_grad(pts_flat, samples, temperature):
    pts = pts_flat.reshape(N, 3)
    pts = _normalize(pts)
    dots = samples @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    T = temperature
    scaled = T * dots
    max_scaled = scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled, axis=1)
    probs = exp_scaled / sum_exp[:, np.newaxis]
    grad = probs.T @ samples
    grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    return grad_tangent.flatten()

def _optimize_with_lbfgs(pts_init, samples, rng, start_time, deadline):
    """
    Optimize using L-BFGS-B with soft-max surrogate.
    """
    pts = _normalize(pts_init).copy()
    best_pts = pts.copy()
    best_radius = _covering_radius_sampled(pts, samples)
    
    # Increase temperature gradually to focus on worst cases
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for T in temperatures:
        if time.time() > deadline:
            break
        
        x0 = pts.flatten()
        
        def obj(x):
            return _soft_max_obj(x, samples, T)
        
        def grad(x):
            return _soft_max_grad(x, samples, T)
        
        try:
            res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.success or res.fun < _soft_max_obj(x0, samples, T):
                new_pts = _normalize(res.x.reshape(N, 3))
                new_radius = _covering_radius_sampled(new_pts, samples)
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline:
            break

    return best_pts, best_radius

def construct():
    """
    Construct 50 points on unit sphere minimizing covering radius.
    Uses gradient-based optimization with soft-max surrogate.
    """
    start_time = time.time()
    deadline = _DEADLINE
    
    # Generate fixed samples for fast evaluation
    # Use more samples for better accuracy in the surrogate
    samples = _generate_fibonacci_samples(10000)
    
    # Track best result
    best_pts = None
    best_radius = np.inf
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Fibonacci init + L-BFGS-B
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        pts_init = _generate_initial_points('fibonacci', rng)
        remaining = deadline - (time.time() - start_time) - 5
        if remaining > 5:
            sa_pts, sa_radius = _optimize_with_lbfgs(pts_init, samples, rng, start_time, start_time + remaining)
            if sa_radius < best_radius:
                best_radius = sa_radius
                best_pts = sa_pts.copy()
    
    # Strategy 2: Random init + L-BFGS-B (multiple restarts)
    for i in range(5):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        pts_init = _generate_initial_points('random', rng)
        remaining = deadline - (time.time() - start_time) - 5
        if remaining < 5:
            break
        sa_pts, sa_radius = _optimize_with_lbfgs(pts_init, samples, rng, start_time, start_time + remaining)
        if sa_radius < best_radius:
            best_radius = sa_radius
            best_pts = sa_pts.copy()
    
    # Strategy 3: Perturbed Fibonacci + L-BFGS-B
    for i in range(3):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        base = _generate_initial_points('fibonacci', rng)
        noise = rng.normal(0, 0.2, size=base.shape)
        pts_init = _normalize(base + noise)
        remaining = deadline - (time.time() - start_time) - 5
        if remaining < 5:
            break
        sa_pts, sa_radius = _optimize_with_lbfgs(pts_init, samples, rng, start_time, start_time + remaining)
        if sa_radius < best_radius:
            best_radius = sa_radius
            best_pts = sa_pts.copy()
    
    # Final polish: use a very high temperature L-BFGS run on the best candidate
    if best_pts is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 5:
            remaining = deadline - elapsed - 2
            if remaining > 2:
                # Use a high temperature to focus on the absolute worst case
                T_high = 20.0
                x0 = best_pts.flatten()
                
                def obj(x):
                    return _soft_max_obj(x, samples, T_high)
                
                def grad(x):
                    return _soft_max_grad(x, samples, T_high)
                
                try:
                    res = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                                   options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-10})
                    if res.success:
                        new_pts = _normalize(res.x.reshape(N, 3))
                        new_radius = _covering_radius_sampled(new_pts, samples)
                        if new_radius < best_radius:
                            best_radius = new_radius
                            best_pts = new_pts.copy()
                except Exception:
                    pass

    if best_pts is None:
        # Fallback: Fibonacci spiral
        best_pts = _generate_initial_points('fibonacci', rng)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
