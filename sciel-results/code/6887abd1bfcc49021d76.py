import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE_MARGIN = 5.0  # seconds to leave before hard timeout

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
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return np.sum(np.log(1.0 / d))

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _project_to_tangent(pts, vecs):
    """Project vectors to tangent space at each point."""
    dot_prods = np.sum(vecs * pts, axis=1, keepdims=True)
    return vecs - dot_prods * pts

def _scipy_lbfgs_sphere(pts, time_limit=10.0, max_iter=500):
    """
    Uses scipy L-BFGS-B on ambient coordinates.
    The objective function normalizes the input points to the sphere.
    The gradient returned is the chain rule gradient: dE/dx = (1/||x||) * (g_tangent).
    """
    start = time.time()
    x0 = pts.flatten().copy()
    
    def objective_and_grad(x):
        if time.time() - start > time_limit:
            return float('inf'), np.zeros_like(x)
        
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        p_norm = p / norms
        
        energy, grad_tan = _energy_and_grad(p_norm)
        
        # Chain rule: grad_x = (1/norms) * grad_tan
        # grad_tan is already tangent to p_norm, so it is orthogonal to p_norm.
        # d/dx (p/||x||) = (1/||x||) I - (1/||x||^3) x x^T
        # grad_E . J = grad_E . ((1/||x||) I - ...) 
        # Since grad_E is tangent to p_norm, grad_E . p_norm = 0.
        # So grad_x = grad_E / ||x||
        
        grad_ambient = grad_tan / norms
        return energy, grad_ambient.flatten()

    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': max_iter,
            'ftol': 1e-15,
            'gtol': 1e-12,
            'maxls': 50,
            'maxfun': 10000
        }
    )
    
    if result.fun < float('inf'):
        best_pts = _normalize(result.x.reshape(N, 3))
        best_energy = _log_energy_fast(best_pts)
    else:
        best_pts = pts.copy()
        best_energy = _log_energy_fast(pts)
        
    return best_pts, best_energy

def _riemannian_cg(pts, time_limit=15.0, max_iter=200):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    d = -grad.copy()
    prev_grad = grad.copy()
    alpha = 0.1
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Backtracking line search (Armijo condition)
        c1 = 1e-4
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Armijo condition
            armijo = new_energy <= energy + c1 * alpha * np.sum(grad * d)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_alpha = alpha
                best_grad = new_grad
            
            if armijo:
                break
            
            alpha *= 0.5
        
        if best_alpha == 0.0:
            # Try a steepest descent step
            alpha_try = 0.01
            for _ in range(10):
                new_pts = _exponential_map(current, -alpha_try * grad)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha_try
                    best_grad = new_grad
                    break
                alpha_try *= 0.5
        
        if best_alpha == 0.0:
            break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
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
    
    return current, energy

def _dual_representation_opt(pts, time_limit=30.0):
    """
    Alternate between Riemannian CG and Scipy L-BFGS.
    """
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    best_pts = current.copy()
    best_energy = energy
    
    phase = 0  # 0: Riemannian CG, 1: Scipy L-BFGS
    max_phases = 6
    
    for p in range(max_phases):
        time_left = time_limit - (time.time() - start)
        if time_left < 3:
            break
        
        if phase % 2 == 0:
            # Riemannian CG phase
            phase_time = min(10.0, time_left - 2)
            if phase_time > 2:
                current, energy = _riemannian_cg(current, time_limit=phase_time, max_iter=100)
        else:
            # Scipy L-BFGS phase
            phase_time = min(10.0, time_left - 2)
            if phase_time > 2:
                current, energy = _scipy_lbfgs_sphere(current, time_limit=phase_time, max_iter=100)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = current.copy()
        
        # Small perturbation to escape potential plateaus
        if p > 0 and p < max_phases - 1:
            rng = np.random.default_rng(p * 7 + 100)
            perturbation = rng.standard_normal((N, 3)) * 0.05
            perturbation = _project_to_tangent(current, perturbation)
            current = _exponential_map(current, perturbation)
            current = _normalize(current)
            energy = _log_energy_fast(current)
        
        phase += 1
    
    return best_pts, best_energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=10):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization with dual-representation
    time_left = time_limit - (time.time() - start)
    if time_left > 8:
        current, energy = _dual_representation_opt(current, time_limit=min(20, time_left - 3))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.3
    decay = 0.8
    
    for rh in range(max_restarts):
        time_left = time_limit - (time.time() - start)
        if time_left < 10:
            break
        
        # Perturb with adaptive scale
        _, grad = _energy_and_grad(current)
        grad_mag = np.linalg.norm(grad)
        
        rng = np.random.default_rng(rh * 13 + 42)
        # Decreasing perturbation scale with restart index
        base_scale = 0.25 * (0.7 ** rh)
        # Larger perturbations when gradient is small (stuck in local min)
        if grad_mag < 0.01:
            adaptive_factor = 1.5
        elif grad_mag < 0.1:
            adaptive_factor = 1.2
        else:
            adaptive_factor = 1.0
        scale = base_scale * adaptive_factor
        
        perturbations = rng.standard_normal((N, 3)) * scale
        perturbations = _project_to_tangent(current, perturbations)
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point with dual-representation
        opt_time = min(12, time_left - 5)
        if opt_time < 4:
            break
        new_opt_pts, new_energy = _dual_representation_opt(new_pts, time_limit=opt_time)
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        accept = delta_e < 0
        if not accept and temp > 1e-6:
            try:
                accept = np.random.rand() < math.exp(-delta_e / temp)
            except OverflowError:
                accept = False
        
        if accept:
            current = new_opt_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = current.copy()
        
        temp *= decay
    
    return best_pts, best_energy

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
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

def _fps_sphere(n, seed=42, batch_size=2000):
    """Fast Farthest Point Sampling on the sphere."""
    rng = np.random.default_rng(seed)
    pts = [rng.standard_normal(3)]
    pts[0] = pts[0] / np.linalg.norm(pts[0])
    pts_arr = np.array(pts)
    
    for _ in range(n - 1):
        candidates = rng.standard_normal((batch_size, 3))
        candidates = _normalize(candidates)
        
        # Compute min distance from each candidate to existing points
        min_dists = np.full(batch_size, np.inf)
        chunk = 50
        for i in range(0, len(pts_arr), chunk):
            chunk_pts = pts_arr[i:i+chunk]
            d = np.linalg.norm(candidates[:, np.newaxis, :] - chunk_pts[np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, np.min(d, axis=1))
        
        best_idx = np.argmax(min_dists)
        pts_arr = np.vstack([pts_arr, candidates[best_idx:best_idx+1]])
        
    return _normalize(pts_arr)

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    
    # Initializations - diverse starting points
    inits = []
    
    # Halton-based initializations (primary, low-discrepancy)
    inits.append(_halton_sphere(N, base_start=2))
    inits.append(_halton_sphere(N, base_start=3))
    
    # Fibonacci sphere
    inits.append(_fibonacci_sphere(N))
    
    # Farthest Point Sampling (Spectral-like uniformity)
    inits.append(_fps_sphere(N, seed=123))
    inits.append(_fps_sphere(N, seed=456))
    
    # Random inits with different seeds
    for s in range(2):
        rng = np.random.default_rng(s * 7 + 1)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Halton
    rng = np.random.default_rng(200)
    pts = _halton_sphere(N) + rng.standard_normal((N, 3)) * 0.1
    pts = _normalize(pts)
    inits.append(pts)

    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    n_inits = len(inits)
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 18:
            break
        
        # Allocate time: give more time to earlier inits, reserve for final polish
        time_remaining_for_inits = remaining - 12
        inits_remaining = n_inits - idx
        time_per_restart = time_remaining_for_inits / inits_remaining
        time_per_restart = min(25.0, max(10.0, time_per_restart))
        
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    if best_pts is None:
        best_pts = _halton_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish - use remaining time for high-precision local optimization
    time_left = deadline - time.time()
    if time_left > 5:
        polish_time = max(4.0, time_left - 2.0)
        try:
            # Final Scipy L-BFGS polish with very tight tolerances
            final_pts, final_energy = _scipy_lbfgs_sphere(best_pts, time_limit=polish_time, max_iter=5000)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
        
        # Second polish pass with Riemannian CG for additional precision
        time_left = deadline - time.time()
        if time_left > 3:
            try:
                cg_pts, cg_energy = _riemannian_cg(best_pts, time_limit=max(2.0, time_left - 1.5), max_iter=500)
                if cg_energy < best_energy:
                    best_energy = cg_energy
                    best_pts = cg_pts
            except Exception:
                pass
        
        # Final normalization and validation
        best_pts = _normalize(best_pts)
        final_check_energy = _log_energy_fast(best_pts)
        if final_check_energy < best_energy:
            best_energy = final_check_energy
    
    return _normalize(best_pts)
