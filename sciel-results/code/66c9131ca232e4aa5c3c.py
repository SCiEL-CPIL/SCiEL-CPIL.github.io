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

def _lbfgs_sphere(pts, time_limit=10.0, max_iter=200, m=10):
    """L-BFGS on the sphere using exponential maps and parallel transport."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # L-BFGS history
    s_history = []  # step vectors (tangent at previous point)
    y_history = []  # gradient differences
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Compute search direction using L-BFGS two-loop recursion
        q = grad.copy()
        rho = []
        
        # Two-loop recursion (backward)
        for k in range(len(s_history) - 1, -1, -1):
            s_k = s_history[k]
            y_k = y_history[k]
            
            ys = np.sum(y_k * s_k)
            if ys < 1e-30:
                rho.append(0.0)
                continue
            rho_k = 1.0 / ys
            rho.append(rho_k)
            alpha_k = rho_k * np.sum(s_k * q)
            q -= alpha_k * y_k
        
        # r = q (standard L-BFGS with H0 = I)
        r = q
        
        # Two-loop recursion (forward)
        d = r.copy()
        for k in range(len(rho)):
            s_k = s_history[k]
            y_k = y_history[k]
            
            if rho[k] == 0.0:
                continue
            ys = np.sum(y_k * s_k)
            if ys < 1e-30:
                continue
            beta_k = (1.0 / ys) * np.sum(y_k * r)
            d += (rho[k] - beta_k) * s_k
        
        # Ensure d is in tangent space
        dot_d = np.sum(d * current, axis=1)
        d -= dot_d[:, np.newaxis] * current
        
        # Check if direction is meaningful
        dir_norm = np.linalg.norm(d)
        if dir_norm < 1e-14:
            break
        
        # Line search (Armijo backtracking with stronger sufficient decrease)
        alpha = 1.0
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        directional_deriv = np.sum(grad * d)
        
        # If directional derivative is positive, flip direction
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            # Armijo condition
            if new_energy > energy + c1 * alpha * directional_deriv:
                break
            
            alpha *= 0.5
            if alpha < 1e-14:
                break
        
        if best_alpha == 0.0:
            # Try smaller step
            alpha_try = 0.01
            for _ in range(5):
                new_pts = _exponential_map(current, alpha_try * d)
                new_energy, new_grad = _energy_and_grad(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_grad = new_grad
                    best_alpha = alpha_try
                    break
                alpha_try *= 0.5
            
            if best_alpha == 0.0:
                break
        
        # Update L-BFGS history
        s_vec = best_pts - current  # approximate step
        y_vec = best_grad - grad
        
        # Project to tangent space
        dot_s = np.sum(s_vec * current, axis=1)
        s_vec -= dot_s[:, np.newaxis] * current
        
        dot_y = np.sum(y_vec * best_pts, axis=1)
        y_vec -= dot_y[:, np.newaxis] * best_pts
        
        s_history.append(s_vec)
        y_history.append(y_vec)
        
        if len(s_history) > m:
            s_history.pop(0)
            y_history.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-12:
            break
    
    return current, energy

def _basin_hopping(pts, time_limit=30.0, max_restarts=10):
    """Perform local optimization followed by basin hopping."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization
    time_left = time_limit - (time.time() - start)
    if time_left > 5:
        current, energy = _lbfgs_sphere(current, time_limit=min(15, time_left - 2))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 0.8
    decay = 0.82
    
    for rh in range(max_restarts):
        time_left = time_limit - (time.time() - start)
        if time_left < 5:
            break
        
        # Perturb with adaptive scale - start larger, decay
        rng = np.random.default_rng(rh + 42)
        base_scale = 0.25 * (0.75 ** rh)
        scale = base_scale * (1.0 + 0.3 * math.sin(rh * 1.3))
        perturbations = rng.standard_normal((N, 3)) * scale
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point
        opt_time = min(15, time_left - 3)
        if opt_time < 3:
            break
        new_opt_pts, new_energy = _lbfgs_sphere(new_pts, time_limit=opt_time)
        
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
    
    # Final polish within basin hopping
    time_left = time_limit - (time.time() - start)
    if time_left > 3:
        try:
            polished, pol_energy = _lbfgs_sphere(best_pts, time_limit=max(2.0, time_left - 1.0), max_iter=300, m=20)
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_pts = polished
        except Exception:
            pass
    
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

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    
    # Initializations - more diverse starting points
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # Random inits with different seeds
    for s in range(3):
        rng = np.random.default_rng(s * 7 + 1)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed Fibonacci with varying scales
    for s in range(3):
        rng = np.random.default_rng(s * 13 + 200)
        scale = 0.1 + 0.08 * s
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        inits.append(pts)
    
    # Spherical design-like init using scipy
    try:
        from scipy.spatial import ConvexHull
        # Generate more points and select a subset
        rng = np.random.default_rng(999)
        many_pts = rng.standard_normal((N * 3, 3))
        many_pts = _normalize(many_pts)
        # Use greedy farthest-point sampling
        selected = [many_pts[0].copy()]
        min_dists = np.full(N * 3, np.inf)
        for _ in range(N - 1):
            d = np.linalg.norm(many_pts[:, np.newaxis, :] - selected[-1][np.newaxis, :, :], axis=2)
            min_dists = np.minimum(min_dists, d)
            next_idx = np.argmax(min_dists)
            selected.append(many_pts[next_idx].copy())
        inits.append(np.array(selected))
    except Exception:
        pass
    
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    n_inits = len(inits)
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 15:
            break
        
        # Allocate time: give more time to earlier inits, reserve for final polish
        time_remaining_for_inits = remaining - 10  # reserve 10s for final polish
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
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish - use remaining time for a high-precision local optimization
    time_left = deadline - time.time()
    if time_left > 2:
        polish_time = max(2.0, time_left - 1.0)
        try:
            final_pts, final_energy = _lbfgs_sphere(best_pts, time_limit=polish_time, max_iter=800, m=20)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
        
        # Second polish pass if time remains
        time_left2 = deadline - time.time()
        if time_left2 > 3:
            try:
                final_pts2, final_energy2 = _lbfgs_sphere(best_pts, time_limit=max(2.0, time_left2 - 1.0), max_iter=500, m=25)
                if final_energy2 < best_energy:
                    best_energy = final_energy2
                    best_pts = final_pts2
            except Exception:
                pass
        
        # Third polish: tighter tolerance, more iterations
        time_left3 = deadline - time.time()
        if time_left3 > 2:
            try:
                final_pts3, final_energy3 = _lbfgs_sphere(best_pts, time_limit=max(1.5, time_left3 - 0.5), max_iter=600, m=30)
                if final_energy3 < best_energy:
                    best_energy = final_energy3
                    best_pts = final_pts3
            except Exception:
                pass
            
    return _normalize(best_pts)
