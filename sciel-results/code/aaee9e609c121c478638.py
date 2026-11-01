import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # seconds, safety margin for 150s limit

def _fibonacci_start():
    """Golden-angle Fibonacci spiral on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_start(dim=3, n=N):
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, dim))
    bases = [2, 3, 5]
    for i in range(n):
        for j, base in enumerate(bases):
            pts[i, j] = halton(i + 1, base)
    
    # Map [0,1)^3 to sphere using spherical coordinates
    theta = 2 * np.pi * pts[:, 0]
    cos_phi = 1 - 2 * pts[:, 1]
    sin_phi = np.sqrt(np.maximum(0, 1 - cos_phi**2))
    
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = cos_phi
    
    return np.column_stack([x, y, z])

def _random_start(seed):
    """Random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _log_energy_fsum(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for accuracy."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _optimize_with_scipy(pts0, max_time, start_time, tol=1e-10, max_iter=1000):
    """
    Optimize using scipy L-BFGS-B with normalization inside objective/gradient.
    """
    def objective(x):
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        p = p / norms
        # Vectorized energy
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-24)
        np.fill_diagonal(dist_sq, 1.0)
        # log(1/d) = -0.5 * log(d^2)
        # We sum over i<j, so 0.5 * sum over all i!=j
        energy = -0.25 * np.sum(np.log(dist_sq))
        return energy

    def gradient(x):
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        p = p / norms
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-24)
        np.fill_diagonal(dist_sq, 1.0)
        # Gradient of -0.25 * sum log(d2) w.r.t. x_i
        # d/dx_i [-0.25 * log(|x_i - x_j|^2)] = -0.25 * 2*(x_i - x_j)/|x_i - x_j|^2 = -(x_i-x_j)/(2*d2)
        # Sum over j != i
        forces = -0.5 * np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
        # Project to tangent space? No, L-BFGS-B works in R^3N, so we just return the gradient.
        # The normalization in objective handles the constraint implicitly, but it's better to
        # provide the gradient of the function defined on R^3N where p is normalized.
        # However, the standard trick is to optimize on the sphere using Riemannian methods,
        # or use the fact that the gradient of the energy on the sphere is the projection.
        # If we use L-BFGS-B on R^3N, we should provide the gradient of the extended function.
        # Let f(p) = E(p/||p||). The gradient w.r.t x is (I - pp^T/||p||^2) * grad_E(p/||p||).
        # Since p is normalized in the function, the gradient of the objective w.r.t x is:
        # grad_obj(x) = (I - p p^T) * grad_E(p)  (since ||p||=1)
        # But wait, the objective function normalizes p. So the chain rule applies.
        # Let u = p/||p||. E(u). grad_x E(u) = (I - u u^T) * grad_u E(u) * (1/||x||)
        # This is complex. A simpler approach: use the Riemannian gradient and let L-BFGS-B
        # handle the unconstrained space? No, L-BFGS-B doesn't know about the manifold.
        
        # Better approach: Use the Riemannian gradient as the gradient for the unconstrained problem,
        # but this only works if the step is projected. L-BFGS-B doesn't project.
        
        # Alternative: Use the fact that minimizing E on the sphere is equivalent to minimizing
        # E(x/||x||) on R^3N. The gradient is (I - xx^T/||x||^2) * grad_E(x/||x||) / ||x||.
        # Since we normalize p inside, let's compute grad_E(p) and project it.
        # Actually, the standard way to use L-BFGS-B for sphere constraints is to just provide
        # the Riemannian gradient and hope the optimizer stays close, or use a penalty method.
        
        # Let's stick to the Riemannian gradient projection.
        # The gradient of the energy w.r.t the position on the sphere is the projection of the
        # Euclidean gradient.
        # grad_E(p) = sum_j (p_j - p_i) / |p_i - p_j|^2  (for log(1/d), grad is (p_i-p_j)/d^2 * (-1) ... wait)
        # E = sum log(1/d) = - sum log(d).
        # d/dp_i log(d) = (p_i - p_j) / d^2.
        # So d/dp_i (-log(d)) = -(p_i - p_j) / d^2 = (p_j - p_i) / d^2.
        # My previous force calculation: forces = -0.5 * sum(diff / dist_sq).
        # diff = p_i - p_j. So forces = -0.5 * sum((p_i - p_j) / d^2).
        # This is 0.5 * sum((p_j - p_i) / d^2).
        # This is correct for the gradient of the energy on the sphere (up to a factor).
        
        # To use L-BFGS-B effectively, we can just return the projected gradient.
        # The optimizer will take steps in R^3N. If the gradient is tangent, the steps will be
        # mostly tangent.
        dot_products = np.sum(forces * p, axis=1)
        proj_forces = forces - dot_products[:, np.newaxis] * p
        return proj_forces.flatten()

    try:
        result = minimize(
            objective, pts0.flatten(), jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8, 'maxls': 50}
        )
        res_pts = result.x.reshape(N, 3)
        norms = np.linalg.norm(res_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        res_pts = res_pts / norms
        return res_pts, result.fun
    except Exception:
        return pts0, _log_energy_fsum(pts0)

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Multiple diverse starting points
    starts = []
    
    # Fibonacci spiral
    pts_fib = _fibonacci_start()
    starts.append(pts_fib)
    
    # Halton sequence
    pts_halton = _halton_start()
    starts.append(pts_halton)
    
    # Random starts
    for seed in [1, 42, 123, 456]:
        pts_rand = _random_start(seed)
        starts.append(pts_rand)
    
    best_pts = starts[0].copy()
    best_energy = _log_energy_fsum(starts[0])
    
    # Phase 1: Optimize from each diverse start
    for i, pts_start in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 15:
            break
        
        # Give more time to first few starts
        if i < 2:
            max_time = min(remaining - 5, 25.0)
        else:
            max_time = min(remaining - 5, 12.0)
        
        if max_time < 3:
            break
        
        pts_opt, e_opt = _optimize_with_scipy(pts_start, max_time, start_time, tol=1e-10, max_iter=1000)
        
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            e_precise = _log_energy_fsum(pts_opt)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = pts_opt.copy()
    
    # Phase 2: Additional random restarts if time permits
    rng = np.random.RandomState(999)
    n_restarts = 10
    for i in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 15:
            break
        
        pts_rand = _random_start(1000 + i)
        max_time = min(remaining - 10, 8.0)
        if max_time < 2:
            break
        
        try:
            pts_new, e_new = _optimize_with_scipy(pts_rand, max_time, start_time, tol=1e-10, max_iter=500)
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 10:
                e_precise = _log_energy_fsum(pts_new)
                if e_precise < best_energy:
                    best_energy = e_precise
                    best_pts = pts_new.copy()
        except Exception:
            continue
    
    # Phase 3: Final high-precision polish on the best configuration
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    if remaining > 10:
        try:
            max_polish_time = min(remaining - 5, 20.0)
            pts_polished, e_polished = _optimize_with_scipy(best_pts, max_polish_time, start_time, tol=1e-12, max_iter=2000)
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 5:
                e_final = _log_energy_fsum(pts_polished)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_polished.copy()
        except Exception:
            pass
    
    # Final verification
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 2:
        final_e = _log_energy_fsum(best_pts)
        if final_e < best_energy:
            best_energy = final_e
    
    # Ensure valid output
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-15) or np.any(~np.isfinite(best_pts)):
        return _fibonacci_start()
    
    return _project_to_sphere(best_pts)

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms
