"""Direct approach for minimal logarithmic energy of 150 points on sphere."""
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


def _halton_sphere(n, base=2):
    """Generate n points on sphere using Halton sequence in 3D."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((n, 3))
    for i in range(n):
        idx = i + 1
        pts[i, 0] = 2 * halton(idx, 2) - 1
        pts[i, 1] = 2 * halton(idx, 3) - 1
        pts[i, 2] = 2 * halton(idx, 5) - 1
    
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    pts /= norms
    return pts


def _energy_and_grad(pts):
    """Vectorized energy and gradient (projected to tangent space)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    
    # Energy: sum over i<j of log(1/d)
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    energy = np.sum(np.log(1.0 / d_upper))
    
    # Gradient: for point i, grad_i = sum_{j!=i} (pts_i - pts_j) / d_ij^2
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    weighted_sum = inv_d2 @ pts
    sum_inv = np.sum(inv_d2, axis=1)
    grad = pts * sum_inv[:, np.newaxis] - weighted_sum
    
    # Project to tangent space: grad_i -= dot(grad_i, pts_i) * pts_i
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return energy, grad


def _newton_polish(pts, max_iter=20, time_limit=None, start_time=None):
    """Newton-style polish using inverse Hessian approximation on tangent space."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts.copy()
    n = len(pts)
    energy, grad = _energy_and_grad(pts)
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Build a simplified Newton step: use diagonal approximation of Hessian
        # H_ii,jj approx ~ sum_k (1/d_ik^2 + 1/d_jk^2) for i!=j
        # We'll use a damped Newton step with line search
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        
        # Approximate Hessian diagonal: for each point i,
        # H_ii ~ sum_{j!=i} (1/d_ij^2) * I_3 - (2/d_ij^2) * (p_i-p_j)(p_i-p_j)^T projected
        # Simplified: use scalar damping based on gradient magnitude
        
        # Damped Newton: step = -alpha * grad where alpha adapts
        # Use a simple adaptive step size
        alpha = 0.5 / (grad_norm + 1e-10)
        
        # Try several step sizes
        best_new_pts = None
        best_new_energy = energy
        for trial_alpha in [alpha, alpha * 0.5, alpha * 0.25, alpha * 0.1]:
            new_pts = pts + trial_alpha * (-grad)
            new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < best_new_energy:
                best_new_energy = new_energy
                best_new_pts = new_pts
        
        if best_new_pts is None or best_new_energy >= energy - 1e-15:
            break
        
        pts = best_new_pts
        energy, grad = _energy_and_grad(pts)
    
    return pts, energy


def _fibonacci_sphere(n):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _exp_map(pts, tangent):
    """Exponential map on the sphere: move from pts along tangent vectors."""
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    # For small steps, exp map ~ retract; for large, use exact formula
    # p_new = cos(theta)*p + sin(theta)*v/||v||
    sin_t = np.sin(norms)
    cos_t = np.cos(norms)
    new_pts = cos_t * pts + sin_t * (tangent / norms)
    new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
    return new_pts


def _retract_to_sphere(pts):
    """Retract points to the sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _lbgfs_sphere(pts0, max_iter=500, m=10, time_limit=None, start_time=None):
    """L-BFGS optimization on the sphere with projected gradients and Wolfe line search."""
    if start_time is None:
        start_time = time.time()
    
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    
    # L-BFGS memory
    s_list = []  # step directions
    y_list = []  # gradient differences
    
    step_size = 0.1
    min_step = 1e-12
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-13:
            break
        
        # L-BFGS two-loop recursion to find search direction
        if len(s_list) == 0:
            direction = -grad
        else:
            # Two-loop recursion
            q = grad.copy()
            rhos = []
            alphas = []
            for i in range(len(s_list) - 1, -1, -1):
                si = s_list[i]
                yi = y_list[i]
                ri = np.sum(si * yi)
                if abs(ri) < 1e-16:
                    rhos.append(0.0)
                    alphas.append(0.0)
                else:
                    rho = 1.0 / ri
                    rhos.append(rho)
                    ai = np.sum(si * q) * rho
                    alphas.append(ai)
                    q -= ai * yi
            
            # Scale by initial curvature
            y0 = y_list[0]
            s0 = s_list[0]
            g0_dot_y0 = np.sum(grad * y0)
            g0_dot_s0 = np.sum(grad * s0)
            gamma = g0_dot_s0 / g0_dot_y0 if abs(g0_dot_y0) > 1e-16 else 1.0
            r = q * gamma
            
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho_i = rhos[len(s_list) - 1 - i]
                beta = rho_i * np.sum(yi * r)
                r += si * (alphas[len(s_list) - 1 - i] - beta)
            
            direction = -r
        
        # Ensure direction is on tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        # Wolfe line search with backtracking
        c1 = 1e-4
        c2 = 0.9
        alpha = step_size
        accepted = False
        new_pts = None
        new_grad = None
        new_energy = None
        
        for _ in range(40):
            new_pts = pts + alpha * direction
            new_pts = _retract_to_sphere(new_pts)
            
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Armijo condition
            grad_dot_dir = np.sum(grad * direction)
            if new_energy <= energy + c1 * alpha * grad_dot_dir:
                # Strong Wolfe: curvature condition
                new_grad_dot_dir = np.sum(new_grad * direction)
                if new_grad_dot_dir >= c2 * grad_dot_dir:
                    accepted = True
                    break
                # If curvature fails, still accept if Armijo holds (relaxed)
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            step_size *= 0.5
            if step_size < min_step:
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
        step_size = min(alpha * 1.2, 0.5)
    
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
    for seed in [42, 123, 456]:
        rng = np.random.default_rng(seed)
        pts_rand = rng.standard_normal((N, 3))
        pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        inits.append(pts_rand)
    
    # 4. Perturbed Fibonacci
    for seed in [7, 99]:
        rng = np.random.default_rng(seed)
        pts_fib = _fibonacci_sphere(N)
        pts_pert = pts_fib + rng.standard_normal((N, 3)) * 0.05
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        inits.append(pts_pert)
    
    # 5. Perturbed Halton
    rng = np.random.default_rng(314)
    pts_hal = _halton_sphere(N)
    pts_hal_pert = pts_hal + rng.standard_normal((N, 3)) * 0.03
    pts_hal_pert /= np.linalg.norm(pts_hal_pert, axis=1, keepdims=True)
    inits.append(pts_hal_pert)
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: L-BFGS from multiple starts
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - time.time()
        time_limit = max(remaining - 10, 5)
        
        pts, energy = _lbgfs_sphere(
            init_pts,
            max_iter=600,
            m=10,
            time_limit=time_limit,
            start_time=time.time()
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 2: Newton polish on best result
    if best_pts is not None and time.time() < deadline - 10:
        remaining = deadline - time.time()
        time_limit = max(remaining - 5, 2)
        best_pts, best_energy = _newton_polish(
            best_pts,
            max_iter=50,
            time_limit=time_limit,
            start_time=time.time()
        )
    
    # Phase 3: Final L-BFGS polish
    if best_pts is not None and time.time() < deadline - 5:
        remaining = deadline - time.time()
        time_limit = max(remaining - 2, 1)
        best_pts, best_energy = _lbgfs_sphere(
            best_pts,
            max_iter=200,
            m=10,
            time_limit=time_limit,
            start_time=time.time()
        )
    
    # Ensure valid output
    best_pts = _retract_to_sphere(best_pts)
    
    # Verify all points are valid
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            best_pts = _fibonacci_sphere(N)
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            best_pts = _fibonacci_sphere(N)
            break
    
    # Final check: no points too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                rng = np.random.default_rng(i * 1000 + j)
                best_pts[j] += rng.standard_normal(3) * 1e-6
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    return best_pts
