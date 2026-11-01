"""Direct approach for minimal logarithmic energy of 150 points on sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150


def _farthest_point_init(n, seed=42):
    """Generate initial points using farthest-point sampling on the sphere."""
    rng = np.random.default_rng(seed)
    pts = [rng.standard_normal(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])
    
    for _ in range(n - 1):
        # Compute distance from each candidate to the farthest existing point
        candidates = rng.standard_normal((2000, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # Find the candidate that maximizes the minimum distance to existing points
        best_idx = 0
        best_min_dist = 0
        for i, cand in enumerate(candidates):
            dists = np.linalg.norm(np.array(pts) - cand, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        
        pts.append(candidates[best_idx])
    
    return np.array(pts)


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-12:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    inv_d2 = 1.0 / (dists ** 2)
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad


def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _halton_sphere(n):
    """Generate Halton sequence points mapped to unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = np.zeros((n, 3))
    for i in range(n):
        # Use 2D Halton (bases 2 and 3) to get (u, v) in [0,1)^2
        u = halton(i + 1, 2)
        v = halton(i + 1, 3)
        # Map to sphere via spherical coordinates with jitter
        theta = 2.0 * np.pi * u
        # Use a non-uniform mapping for v to avoid poles clustering
        phi = np.arccos(1.0 - 2.0 * v)
        pts[i, 0] = np.sin(phi) * np.cos(theta)
        pts[i, 1] = np.sin(phi) * np.sin(theta)
        pts[i, 2] = np.cos(phi)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _riemannian_cg(pts, time_limit=10.0):
    """Riemannian conjugate gradient with momentum on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # Initial search direction: negative gradient (already projected)
    d = -grad.copy()
    
    # Momentum parameter
    momentum = 0.9
    prev_d = np.zeros_like(d)
    
    # Precompute for beta calculation
    prev_grad_norm_sq = np.sum(grad * grad)
    
    for it in range(800):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Apply momentum to search direction
        d = momentum * prev_d + (1 - momentum) * (-grad)
        
        # Project d to tangent space
        dot_prod = np.sum(d * current, axis=1)
        d -= dot_prod[:, np.newaxis] * current
        
        # Line search: minimize f(exp_x(alpha * d))
        # exp_x(v) = x * cos(||v||) + (v / ||v||) * sin(||v||) for unit sphere
        best_alpha = 0.0
        best_energy = energy
        
        alpha = 0.1
        for _ls in range(40):
            # Compute exponential map
            alpha_norm = alpha * np.linalg.norm(d, axis=1)
            # Handle case where d is zero
            mask = alpha_norm > 1e-15
            new_pts = current.copy()
            if np.any(mask):
                cos_a = np.cos(alpha_norm)
                sin_a = np.sin(alpha_norm)
                d_hat = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-15)
                new_pts[mask] = (current[mask] * cos_a[mask, np.newaxis] + 
                                d_hat[mask] * sin_a[mask, np.newaxis])
            new_pts = _normalize(new_pts)
            
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_alpha = alpha
            
            if new_energy < energy - 1e-14:
                break
            alpha *= 0.5
        
        if best_alpha == 0.0:
            # Try smaller step
            alpha = 1e-4
            for _ls in range(20):
                alpha_norm = alpha * np.linalg.norm(d, axis=1)
                mask = alpha_norm > 1e-15
                new_pts = current.copy()
                if np.any(mask):
                    cos_a = np.cos(alpha_norm)
                    sin_a = np.sin(alpha_norm)
                    d_hat = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-15)
                    new_pts[mask] = (current[mask] * cos_a[mask, np.newaxis] + 
                                    d_hat[mask] * sin_a[mask, np.newaxis])
                new_pts = _normalize(new_pts)
                new_energy = _log_energy_fast(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_alpha = alpha
                    break
                alpha *= 0.5
        
        if best_alpha == 0.0:
            break
        
        # Update current point
        alpha_norm = best_alpha * np.linalg.norm(d, axis=1)
        mask = alpha_norm > 1e-15
        new_pts = current.copy()
        if np.any(mask):
            cos_a = np.cos(alpha_norm)
            sin_a = np.sin(alpha_norm)
            d_hat = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-15)
            new_pts[mask] = (current[mask] * cos_a[mask, np.newaxis] + 
                            d_hat[mask] * sin_a[mask, np.newaxis])
        current = _normalize(new_pts)
        
        # Compute new gradient
        new_energy, new_grad = _energy_and_grad(current)
        
        if np.linalg.norm(new_grad) < 1e-14:
            break
        
        grad = new_grad
        energy = new_energy
        
        # Store previous direction for momentum
        prev_d = d.copy()
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-12:
            break
    
    return current, energy


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _lbfgs_optimize(pts, time_limit=30.0):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    
    # Flatten points for L-BFGS-B
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy, grad = _energy_and_grad(p)
        # Project gradient back to flat space
        return energy, grad.flatten()
    
    # Use L-BFGS-B with bounds to keep points reasonable
    # No explicit bounds needed since we normalize after each step
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12,
                            'maxfun': 10000, 'maxls': 50})
    
    # Check time
    if time.time() - start > time_limit:
        return pts, _log_energy_fast(pts)
    
    # Extract optimized points
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy


def _polish(pts, time_limit=5.0):
    """High-precision polish with very small steps and tighter convergence."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 1e-4
    
    for it in range(20000):
        if time.time() - start > time_limit - 0.1:
            break
        
        e, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-18:
            break
        
        # Backtracking line search with very tight tolerance
        alpha = step_size
        accepted = False
        for _ in range(40):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < e - 1e-18:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            step_size = min(alpha * 1.05, 0.01)
        else:
            step_size *= 0.5
            if step_size < 1e-20:
                break
    
    return current, energy


def _grad_descent(pts, time_limit=10.0):
    """Gradient descent with adaptive step size."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    step_size = 0.01
    
    for it in range(5000):
        if time.time() - start > time_limit - 0.1:
            break
        
        e, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-14:
            break
        
        # Backtracking line search
        alpha = step_size
        accepted = False
        for _ in range(50):
            new_pts = current - alpha * grad
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < e - 1e-12:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = new_pts
            energy = new_energy
            # Increase step size if we took a full step
            if alpha == step_size:
                step_size = min(step_size * 1.1, 0.1)
        else:
            step_size *= 0.5
            if step_size < 1e-16:
                break
    
    return current, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    
    best_pts = pts_fib.copy()
    best_energy = _log_energy_fast(pts_fib)
    
    # Generate diverse initializations
    restarts = [
        ("fibonacci", pts_fib.copy()),
        ("halton", _halton_sphere(N)),
        ("farthest", _farthest_point_init(N, seed=42)),
        ("farthest2", _farthest_point_init(N, seed=123)),
    ]
    
    # Random initializations with different seeds
    for seed in range(8):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        restarts.append((f"random_{seed}", pts_rand))
    
    # Perturbed Fibonacci
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 456 + 13)
        noise_level = 0.02 * (seed + 1)
        pts_pert = pts_fib + rng_local.standard_normal((N, 3)) * noise_level
        pts_pert = _normalize(pts_pert)
        restarts.append((f"perturbed_{seed}", pts_pert))
    
    # Grid-based initializations
    for axis in range(3):
        # Create points along great circles
        pts_grid = []
        n_per_circle = 10
        for k in range(15):
            angle_z = np.pi * k / 15
            z = np.cos(angle_z)
            r = np.sin(angle_z)
            for j in range(n_per_circle):
                angle_x = 2 * np.pi * j / n_per_circle
                if axis == 0:
                    x = r * np.cos(angle_x)
                    y = r * np.sin(angle_x)
                elif axis == 1:
                    x = r * np.cos(angle_x)
                    y = z
                    z = r * np.sin(angle_x)
                else:
                    x = z
                    y = r * np.cos(angle_x)
                    z = r * np.sin(angle_x)
                pts_grid.append([x, y, z])
        pts_grid = np.array(pts_grid[:N])
        pts_grid = _normalize(pts_grid)
        restarts.append((f"grid_axis{axis}", pts_grid))
    
    # Optimize each restart
    for name, init_pts in restarts:
        if time.time() > deadline - 25:
            break
        
        # First pass: L-BFGS-B optimization
        time_left = deadline - time.time() - 10
        if time_left < 3:
            break
        
        opt_pts, opt_energy = _lbfgs_optimize(init_pts, time_limit=min(time_left, 12.0))
        
        # Second pass: Riemannian conjugate gradient on the sphere
        if time.time() < deadline - 8:
            rcg_time = min(deadline - time.time() - 3, 8.0)
            if rcg_time > 0.5:
                rcg_pts, rcg_energy = _riemannian_cg(opt_pts, time_limit=rcg_time)
                if rcg_energy < opt_energy:
                    opt_pts = rcg_pts
                    opt_energy = rcg_energy
        
        # Third pass: Gradient descent for fine-tuning
        if time.time() < deadline - 5:
            gd_time = min(deadline - time.time() - 2, 4.0)
            if gd_time > 0.5:
                gd_pts, gd_energy = _grad_descent(opt_pts, time_limit=gd_time)
                if gd_energy < opt_energy:
                    opt_pts = gd_pts
                    opt_energy = gd_energy
        
        # Fourth pass: Polish
        if time.time() < deadline - 3:
            polish_time = min(deadline - time.time() - 1, 3.0)
            if polish_time > 0.3:
                polished_pts, polished_energy = _polish(opt_pts, time_limit=polish_time)
                if polished_energy < opt_energy:
                    opt_pts = polished_pts
                    opt_energy = polished_energy
        
        # Update best
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    # Final high-precision polish on best solution
    if time.time() < deadline - 2:
        final_time = min(deadline - time.time() - 0.5, 5.0)
        if final_time > 0.3:
            # Try Riemannian CG first for better local convergence
            rcg_time = min(final_time * 0.5, 3.0)
            if rcg_time > 0.3:
                rcg_pts, rcg_energy = _riemannian_cg(best_pts, time_limit=rcg_time)
                if rcg_energy < best_energy:
                    best_energy = rcg_energy
                    best_pts = rcg_pts
            
            # Then polish with tighter tolerance
            polish_time = min(deadline - time.time() - 0.5, 3.0)
            if polish_time > 0.3:
                final_pts, final_energy = _polish(best_pts, time_limit=polish_time)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pts = final_pts
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
