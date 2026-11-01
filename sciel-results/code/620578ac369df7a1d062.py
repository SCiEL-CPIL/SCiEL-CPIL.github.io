"""Direct approach for minimal logarithmic energy of 150 points on sphere.
Combines L-BFGS with Riemannian CG using exponential maps for curvature-aware updates,
Halton and Fibonacci initializations, and a final high-precision polish."""
import time
import math
import numpy as np

N = 150


def _log_energy(pts):
    """Compute logarithmic energy using math.fsum for precision."""
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
    dists = np.sqrt(dists_sq)
    
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    valid_mask = d_upper > 1e-12
    if not np.all(valid_mask):
        return float('inf'), np.zeros_like(pts)
    
    terms = [math.log(1.0 / d) for d in d_upper]
    energy = math.fsum(terms)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum = inv_d2 @ pts
    sum_inv = np.sum(inv_d2, axis=1)
    grad = pts * sum_inv[:, np.newaxis] - weighted_sum
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _exp_map(pts, v):
    """Exponential map on sphere: move points along tangent vectors."""
    norms_v = np.linalg.norm(v, axis=1, keepdims=True)
    safe_norms = np.where(norms_v < 1e-15, 1.0, norms_v)
    cos_t = np.cos(norms_v)
    sin_t = np.sin(norms_v)
    return pts * cos_t + v * (sin_t / safe_norms)


def _retr_tract(pts, v):
    """Retraction: project pts + v to sphere."""
    return _project_to_sphere(pts + v)


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _halton_sphere(n):
    """Generate n points on sphere using Halton sequence in 3D."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    points = []
    for i in range(1, n + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        z = halton(i, 5)
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        if norm > 1e-15:
            points.append(p / norm)
        else:
            points.append(np.array([1.0, 0.0, 0.0]))
    return np.array(points)


def _lbgfs_sphere(pts0, max_iter=500, m=10, time_limit=None, start_time=None, tol=1e-12):
    """L-BFGS optimization on the sphere with projected gradients and Wolfe line search."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    # L-BFGS memory
    s_list = []
    y_list = []
    
    step_size = 0.5
    min_step = 1e-14
    c1 = 1e-4
    c2 = 0.9
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # L-BFGS two-loop recursion
        if len(s_list) == 0:
            direction = -grad
        else:
            q = grad.copy()
            rhos = []
            alphas = []
            
            for i in range(len(s_list) - 1, -1, -1):
                si = s_list[i]
                yi = y_list[i]
                si_yi = np.sum(si * yi)
                if abs(si_yi) < 1e-16:
                    rho = 0.0
                else:
                    rho = 1.0 / si_yi
                rhos.append(rho)
                ai = np.sum(si * q) * rho
                alphas.append(ai)
                q -= ai * yi
            
            # Initial curvature estimate
            if len(s_list) > 0:
                last_s = s_list[-1]
                last_y = y_list[-1]
                sy = np.sum(last_s * last_y)
                if abs(sy) > 1e-16:
                    gamma = np.sum(last_s * last_s) / sy
                    if gamma < 1e-10 or gamma > 1e10:
                        gamma = 1.0
                else:
                    gamma = 1.0
            else:
                gamma = 1.0
            
            r = q * gamma
            
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta = rho * np.sum(yi * r)
                r += si * (alpha_i - beta)
            
            direction = -r
        
        # Ensure direction is on tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        # Wolfe line search
        alpha = step_size
        accepted = False
        grad_dir = np.sum(grad * direction)
        
        if grad_dir >= 0:
            direction = -grad
            dot_dir = np.sum(direction * pts, axis=1)
            direction -= dot_dir[:, np.newaxis] * pts
            grad_dir = np.sum(grad * direction)
        
        for ls_iter in range(50):
            new_pts = _retr_tract(pts, alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if not np.isfinite(new_energy):
                alpha *= 0.5
                continue
            
            armijo = new_energy <= energy + c1 * alpha * grad_dir
            curvature = np.sum(new_grad * direction) >= c2 * grad_dir
            
            if armijo:
                accepted = True
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            step_size *= 0.5
            if step_size < min_step:
                break
            # Try a small step anyway
            alpha = max(step_size, min_step)
            new_pts = _retr_tract(pts, alpha * direction)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if np.isfinite(new_energy) and new_energy < energy:
                accepted = True
            else:
                break
        
        if not accepted:
            break
        
        # Update L-BFGS memory
        s = alpha * direction
        y = new_grad - grad
        sy = np.sum(s * y)
        
        if sy > 1e-14:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
            s_list.append(s)
            y_list.append(y)
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
        step_size = min(alpha * 1.5, 2.0)
    
    return pts, energy


def _cg_sphere(pts0, max_iter=500, time_limit=None, start_time=None, tol=1e-12):
    """Conjugate gradient optimization on the sphere using exponential maps."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    # CG state
    p = -grad
    p = p - np.sum(p * pts, axis=1, keepdims=True) * pts
    
    beta = 0.0
    old_grad_norm_sq = np.sum(grad * grad)
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if np.linalg.norm(grad) < tol:
            break
        
        # Line search along p using exponential map
        alpha = 0.5
        accepted = False
        grad_dir = np.sum(grad * p)
        
        if grad_dir >= 0:
            p = -grad
            p = p - np.sum(p * pts, axis=1, keepdims=True) * pts
            grad_dir = np.sum(grad * p)
        
        for _ in range(50):
            new_pts = _exp_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if not np.isfinite(new_energy):
                alpha *= 0.5
                continue
            
            if new_energy < energy + 1e-4 * alpha * grad_dir:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            alpha = max(0.01, 1e-14)
            new_pts = _exp_map(pts, alpha * p)
            new_energy, new_grad = _energy_and_grad(new_pts)
            if np.isfinite(new_energy) and new_energy < energy:
                accepted = True
            else:
                break
        
        if not accepted:
            break
        
        # Update CG
        new_grad_norm_sq = np.sum(new_grad * new_grad)
        beta = new_grad_norm_sq / max(old_grad_norm_sq, 1e-30)
        p = -new_grad + beta * p
        p = p - np.sum(p * pts, axis=1, keepdims=True) * pts
        
        pts = new_pts
        grad = new_grad
        energy = new_energy
        old_grad_norm_sq = new_grad_norm_sq
    
    return pts, energy


def _alternating_optimize(pts0, max_iter=1000, time_limit=None, start_time=None, 
                          lbgfs_iters=50, cg_iters=30, m=10):
    """Alternate between L-BFGS and CG to escape shallow local minima."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, _ = _energy_and_grad(pts)
    
    if not np.isfinite(energy):
        return pts, energy
    
    phase = 0  # 0 = L-BFGS, 1 = CG
    total_iters = 0
    
    while total_iters < max_iter:
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        if phase == 0:
            pts, energy = _lbgfs_sphere(
                pts, max_iter=lbgfs_iters, m=m,
                time_limit=time_limit, start_time=start_time
            )
            total_iters += lbgfs_iters
            phase = 1
        else:
            pts, energy = _cg_sphere(
                pts, max_iter=cg_iters,
                time_limit=time_limit, start_time=start_time
            )
            total_iters += cg_iters
            phase = 0
        
        if not np.isfinite(energy):
            break
    
    return pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Multiple diverse initializations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_fibonacci_sphere(N))
    
    # 2. Halton sequence
    inits.append(_halton_sphere(N))
    
    # 3. Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)
    
    # 4. Perturbed Fibonacci with different perturbation strengths
    for seed, scale in [(7, 0.05), (99, 0.1), (200, 0.02)]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * scale
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Run alternating optimization on all initializations
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - time.time()
        time_per_init = min(remaining / max(1, len(inits) - idx), 20.0)
        
        pts, energy = _alternating_optimize(
            init_pts,
            max_iter=200,
            lbgfs_iters=40,
            cg_iters=25,
            m=10,
            time_limit=time_per_init,
            start_time=time.time()
        )
        
        if np.isfinite(energy) and energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Refine the best result with more iterations
    if best_pts is not None and time.time() < deadline - 15:
        remaining = deadline - time.time()
        best_pts, best_energy = _alternating_optimize(
            best_pts,
            max_iter=300,
            lbgfs_iters=50,
            cg_iters=30,
            m=15,
            time_limit=remaining - 5,
            start_time=time.time()
        )
    
    # Phase 3: Final high-precision polish with L-BFGS only
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        best_pts, best_energy = _lbgfs_sphere(
            best_pts,
            max_iter=int(remaining * 15),
            m=20,
            time_limit=remaining - 2,
            start_time=time.time(),
            tol=1e-14
        )
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    best_pts = best_pts.copy()
    best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Verify all points are valid and no two points are too close
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(best_pts[i] - best_pts[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-9:
            valid = False
    
    if not valid:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
