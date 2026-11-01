"""
Combined program: Riemannian Conjugate Gradient with Multi-Start and L-BFGS-B Polish
for minimal logarithmic energy of 150 points on the unit sphere.

Strategy:
1. Generate diverse initial configurations (Fibonacci, Halton, Greedy FPS, Random, Perturbed).
2. Use Riemannian Conjugate Gradient (CG) with backtracking line search on the sphere manifold.
   This uses the exponential map to stay on the sphere and leverages second-order-like
   information (via conjugate directions) for efficient descent, escaping shallow local minima.
   Improved line search with dynamic initial step size.
3. Polish each candidate with L-BFGS-B for high-precision convergence.
4. Implement a "Shake and Restart" mechanism to escape deep local minima.
5. Final high-precision polish with multiple L-BFGS-B runs from slightly perturbed best solutions.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE_SECONDS = 150.0
SAFETY_MARGIN = 10.0  # seconds before the deadline to stop and return best

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed (no gradient)."""
    n = pts.shape[0]
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected to tangent space)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on the diagonal
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_ambient_i = sum_j (x_j - x_i) / ||x_i - x_j||^2
    # = sum_j x_j / d^2 - x_i * sum_j 1/d^2
    weighted_sum_pts = inv_d2 @ pts 
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad_ambient = weighted_sum_pts - pts * sum_inv_d2[:, np.newaxis]
    
    # Project gradient to tangent space: g_tangent = g - (g . x) * x
    dot_prod = np.sum(grad_ambient * pts, axis=1)
    grad_tangent = grad_ambient - dot_prod[:, np.newaxis] * pts
    
    return grad_tangent

def _energy_and_grad(pts):
    """Compute both energy and gradient."""
    energy = _log_energy_fast(pts)
    grad = _log_energy_grad(pts)
    return energy, grad

def _exponential_map(pts, vecs):
    """
    Exponential map on the unit sphere.
    pts: (N, 3)
    vecs: (N, 3) tangent vectors
    Returns new points on the sphere.
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    sin_term = np.sin(norms_safe)
    cos_term = np.cos(norms_safe)
    
    # Exp_x(v) = x * cos(||v||) + (v/||v||) * sin(||v||)
    new_pts = pts * cos_term + vecs * (sin_term / norms_safe)
    
    # Normalize to ensure unit length (numerical stability)
    new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
    
    return new_pts

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    points = []
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    for i in range(n):
        z = 1 - (2 * i + 1) / n
        r = math.sqrt(1 - z*z)
        theta = 2 * math.pi * i / phi
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append([x, y, z])
    pts = np.array(points)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _halton_sphere(n, base_start=2):
    """Generate Halton sequence points on unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = []
    for i in range(1, n + 1):
        u1 = halton(i, base_start)
        u2 = halton(i, base_start + 1)
        theta = 2.0 * np.pi * u1
        phi = np.arccos(2.0 * u2 - 1.0)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts.append([x, y, z])
    
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _greedy_fps_sphere(n, max_iters=2000, seed=0):
    """
    Greedy Farthest Point Sampling on the sphere.
    """
    rng = np.random.default_rng(seed)
    pts = []
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    pts.append(v)
    
    for _ in range(1, n):
        candidates = rng.normal(size=(max_iters, 3))
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        diff = candidates[:, np.newaxis, :] - np.array(pts)[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        min_dists = np.min(dists_sq, axis=1)
        
        best_idx = np.argmax(min_dists)
        pts.append(candidates[best_idx])
        
    pts = np.array(pts)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_cg(pts, time_limit=10.0, max_iter=500):
    """
    Riemannian Conjugate Gradient on the sphere using exponential maps.
    Uses Polak-Ribiere beta and backtracking line search.
    Improved: Dynamic initial step size.
    """
    start_time = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    best_pts = current.copy()
    best_energy = energy
    
    # Initial search direction = negative gradient
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit - 0.05:
            break
        
        # Dynamic initial step size: larger if gradient is small, smaller if large
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            alpha = 0.5
        else:
            alpha = min(1.0, 0.1 / grad_norm)
        
        best_ls_energy = energy
        best_ls_pts = current.copy()
        best_ls_alpha = 0.0
        best_ls_grad = grad.copy()
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < best_ls_energy:
                best_ls_energy = new_energy
                best_ls_pts = new_pts
                best_ls_alpha = alpha
                best_ls_grad = _log_energy_grad(new_pts)
            
            # Armijo condition check
            if new_energy <= energy + 1e-4 * alpha * np.sum(grad * d):
                break
                
            alpha *= 0.5
        
        if best_ls_alpha == 0.0:
            break
        
        # Update
        current = best_ls_pts
        energy = best_ls_energy
        grad = best_ls_grad
        
        if energy < best_energy:
            best_energy = energy
            best_pts = current.copy()
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-12:
            break
        
        # Polak-Ribiere beta for conjugate gradient
        d_new = -grad
        beta = 0.0
        if it > 0:
            beta_num = np.sum(grad * (grad - prev_grad))
            beta_den = np.sum(prev_grad * prev_grad)
            if beta_den > 1e-30:
                beta = max(0.0, beta_num / beta_den)
        
        d = d_new + beta * d
        prev_grad = grad.copy()
        
        # Reset direction if beta is too large (restart)
        if beta > 2.0:
            d = -grad.copy()
    
    return best_pts, best_energy

def _lbfgs_optimize(pts, time_limit=5.0, use_penalty=False, min_dist=0.05):
    """Optimize using L-BFGS-B with projected gradient and optional separation penalty."""
    start_time = time.time()
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy = _log_energy_fast(p)
        grad = _log_energy_grad(p)
        
        if use_penalty:
            # Simple penalty: sum of 1/d for d < min_dist
            # This is non-smooth, so we use a smooth approximation: -log(d) for d < min_dist
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            mask = dists < min_dist
            if np.any(mask):
                # Smooth penalty: -log(d/min_dist) for d < min_dist
                d_safe = np.maximum(dists, 1e-15)
                pen_val = -np.sum(np.log(d_safe[mask] / min_dist))
                # Gradient of penalty
                # d/dx_i (-log(d_ij)) = (x_i - x_j) / d_ij^2
                # Sum over j
                pen_grad = np.zeros_like(p)
                for i in range(N):
                    for j in range(N):
                        if i != j and dists[i,j] < min_dist:
                            d_ij = dists[i,j]
                            pen_grad[i] += (p[i] - p[j]) / (d_ij * d_ij)
                # Project to tangent space
                dot_prod = np.sum(pen_grad * p, axis=1)
                pen_grad -= dot_prod[:, np.newaxis] * p
                energy += pen_val
                grad += pen_grad
        
        return energy, grad.flatten()
    
    # Estimate max iterations based on time
    max_iter = 200
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxfun': 5000, 'maxls': 50})
        
        x_opt = res.x.reshape(N, 3)
        x_opt = _normalize(x_opt)
        
        # Compute actual energy (without penalty) for scoring
        energy = _log_energy_fast(x_opt)
        
        return x_opt, energy
    except Exception:
        return pts, _log_energy_fast(pts)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    best_pts = None
    best_energy = float('inf')
    
    # 1. Generate Initializations
    inits = []
    fib_pts = _fibonacci_sphere(N)
    inits.append(fib_pts.copy())
    inits.append(_halton_sphere(N))
    inits.append(_greedy_fps_sphere(N, seed=1))
    inits.append(_greedy_fps_sphere(N, seed=2))
    
    # Random inits
    for seed in [42, 123, 456]:
        rng = np.random.default_rng(seed)
        rand_pts = rng.normal(size=(N, 3))
        inits.append(_normalize(rand_pts))
        
    # Perturbed Fibonacci
    for seed in [7, 8, 9]:
        rng = np.random.default_rng(seed)
        perturbed = fib_pts + rng.normal(0, 0.05, size=(N, 3))
        inits.append(_normalize(perturbed))

    # 2. Multi-Start Optimization
    num_starts = len(inits)
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        remaining = deadline - time.time()
        
        # Phase 1: Riemannian CG (fast, efficient)
        t_cg = min(8.0, remaining * 0.5)
        if t_cg < 1.0:
            continue
            
        try:
            pts_cg, e_cg = _riemannian_cg(init_pts, time_limit=t_cg)
        except Exception:
            continue
            
        # Phase 2: L-BFGS-B Polish (high precision) with penalty for early starts
        # Use penalty for first few starts to avoid clustering, then pure optimization
        use_penalty = (i < 4)  # Use penalty for first 4 starts
        t_lbfgs = min(4.0, remaining - t_cg - 1.0)
        if t_lbfgs > 1.0:
            try:
                pts_opt, e_opt = _lbfgs_optimize(pts_cg, time_limit=t_lbfgs, 
                                                   use_penalty=use_penalty,
                                                   min_dist=0.05)
            except Exception:
                pts_opt, e_opt = pts_cg, e_cg
        else:
            pts_opt, e_opt = pts_cg, e_cg
            
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()

    # 3. Shake and Restart
    if best_pts is not None and time.time() < deadline - 10.0:
        rng_shake = np.random.default_rng(999)
        shake_count = 0
        max_shakes = 3
        
        while time.time() < deadline - 10.0 and shake_count < max_shakes:
            # Create a shake: perturb a random subset of points
            shake_pts = best_pts.copy()
            num_shake = N // 5
            indices = rng_shake.choice(N, size=num_shake, replace=False)
            
            # Add random noise
            noise = rng_shake.normal(0, 0.3, size=(num_shake, 3))
            shake_pts[indices] += noise
            shake_pts = _normalize(shake_pts)
            
            # Optimize the shaken configuration
            t_limit = min(10.0, deadline - time.time() - 5.0)
            if t_limit < 2.0:
                break
                
            try:
                # Use CG for speed, then L-BFGS if time permits
                t_cg = t_limit * 0.6
                t_lbfgs = t_limit * 0.4 - 1.0
                
                pts_cg, e_cg = _riemannian_cg(shake_pts, time_limit=t_cg)
                
                if t_lbfgs > 1.0:
                    pts_opt, e_opt = _lbfgs_optimize(pts_cg, time_limit=t_lbfgs,
                                                      use_penalty=False)
                else:
                    pts_opt, e_opt = pts_cg, e_cg
                    
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = pts_opt.copy()
            except Exception:
                pass
            shake_count += 1

    # 4. Final Polish with L-BFGS-B if time permits (no penalty for final precision)
    if best_pts is not None and time.time() < deadline - 2.0:
        t_final = deadline - time.time() - 1.0
        if t_final > 2.0:
            try:
                # Run L-BFGS-B on the best point
                pts_final, e_final = _lbfgs_optimize(best_pts, time_limit=t_final * 0.5,
                                                      use_penalty=False)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_final.copy()
                
                # Run L-BFGS-B on a slightly perturbed version to escape shallow local minima
                if time.time() < deadline - 1.0:
                    rng_perturb = np.random.default_rng(12345)
                    perturbed_pts = best_pts + rng_perturb.normal(0, 0.01, size=(N, 3))
                    perturbed_pts = _normalize(perturbed_pts)
                    t_remaining = deadline - time.time() - 0.5
                    if t_remaining > 1.0:
                        pts_pert, e_pert = _lbfgs_optimize(perturbed_pts, time_limit=t_remaining,
                                                            use_penalty=False)
                        if e_pert < best_energy:
                            best_energy = e_pert
                            best_pts = pts_pert.copy()
            except Exception:
                pass

    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
    
    # Final projection and validation
    best_pts = _normalize(best_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                rng_fix = np.random.default_rng(i+j)
                best_pts[j] += 1e-8 * rng_fix.randn(3)
                best_pts[j] = _normalize(best_pts[j])
    
    return best_pts
