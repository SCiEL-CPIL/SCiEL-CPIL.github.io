"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.optimize import minimize

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs * diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    return float(np.sum(np.log(1.0 / d)))

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i, k] = sum_j (-(pts[i,k] - pts[j,k]) / dists_sq[i,j])
    # = -pts[i,k] * sum_j(1/d2) + sum_j(pts[j,k] / d2)
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return grad

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_norms = np.cos(norms)
    sin_norms_over_norms = np.sin(norms) / norms
    return _normalize(cos_norms * pts + sin_norms_over_norms * tangent_vecs)

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _farthest_point_init(n, seed=42):
    """Generate points using farthest-point sampling for good initial spread."""
    rng = np.random.RandomState(seed)
    pts = np.zeros((n, 3))
    
    # Start with a random point
    pts[0] = rng.randn(3)
    pts[0] = _normalize(pts[0].reshape(1, 3))[0]
    
    min_dist = np.full(n, np.inf)
    
    for i in range(1, n):
        # Compute distance from each candidate to the closest existing point
        diff = pts[i-1:i, :] - pts[:i, :]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        min_dist[i-1] = np.min(dists)
        
        # Pick the point with maximum minimum distance
        # Use a greedy approach: sample candidates and pick best
        candidates = rng.randn(200, 3)
        candidates = _normalize(candidates)
        
        best_candidate = None
        best_min_dist = -1
        
        for c in candidates:
            dists_to_existing = np.sqrt(np.sum((c - pts[:i]) ** 2, axis=1))
            min_d = np.min(dists_to_existing)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_candidate = c
        
        pts[i] = best_candidate
    
    return _normalize(pts)

def _icosahedral_init(n):
    """Generate points using icosahedral symmetry as starting point."""
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of icosahedron
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    base = _normalize(base)
    
    # Generate additional points by perturbing and using symmetry
    pts = [base.copy()]
    
    # Add perturbed versions for diversity
    rng = np.random.RandomState(42)
    for _ in range(3):
        perturbed = base + rng.randn(12, 3) * 0.1
        pts.append(_normalize(perturbed))
    
    # Combine all points and select n points evenly
    all_pts = np.vstack(pts)
    
    # Select n points from all_pts using stratified sampling
    indices = np.linspace(0, len(all_pts) - 1, n, dtype=int)
    selected = all_pts[indices]
    
    return _normalize(selected)

def _halton_sphere(n):
    """Generate Halton sequence points on sphere"""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = np.zeros((n, 3))
    for i in range(n):
        u = halton(i + 1, 2)
        v = halton(i + 1, 3)
        z = 1.0 - 2.0 * v
        theta = 2.0 * np.pi * u
        r = np.sqrt(1.0 - z * z)
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return _normalize(points)

def _riemannian_cg(pts, time_limit=15.0, max_iter=200):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = _normalize(pts.copy())
    energy = _log_energy_fast(current)
    grad = _log_energy_grad(current)
    
    # Initial search direction = negative gradient (already in tangent space)
    d = -grad.copy()
    prev_grad = grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.1:
            break
        
        # Ensure d is in tangent space
        d_dot_pts = np.sum(d * current, axis=1)
        d = d - d_dot_pts[:, np.newaxis] * current
        
        # Backtracking line search with Wolfe conditions
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_alpha = 0.0
        best_grad = grad.copy()
        c1 = 1e-4  # Armijo condition
        c2 = 0.9    # Curvature condition
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy = _log_energy_fast(new_pts)
            new_grad = _log_energy_grad(new_pts)
            
            # Armijo condition
            if new_energy < energy + c1 * alpha * np.sum(grad * d):
                # Check curvature condition
                if np.sum(new_grad * d) <= c2 * alpha * np.sum(grad * d):
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = new_grad
                    break
                else:
                    # Accept if it's still an improvement
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_pts = new_pts
                        best_alpha = alpha
                        best_grad = new_grad
            
            if new_energy > energy:
                break
            
            alpha *= 0.7
        
        if best_alpha == 0.0:
            # Try steepest descent step
            alpha = 0.01
            for ls in range(20):
                new_pts = _exponential_map(current, -alpha * grad)
                new_energy = _log_energy_fast(new_pts)
                if new_energy < energy:
                    best_energy = new_energy
                    best_pts = new_pts
                    best_alpha = alpha
                    best_grad = _log_energy_grad(new_pts)
                    break
                alpha *= 0.5
            
            if best_alpha == 0.0:
                break
            d = -grad.copy()
        
        # Update
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        # Check convergence
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
        # Project d back to tangent space
        d_dot_pts = np.sum(d * current, axis=1)
        d = d - d_dot_pts[:, np.newaxis] * current
        prev_grad = grad.copy()
    
    return current, energy

def _lbfgs_optimize(pts, time_limit=20.0, maxiter=2000):
    """Optimize using L-BFGS-B with projected gradient."""
    start = time.time()
    
    x0 = _normalize(pts.copy()).flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _normalize(p)
        energy = _log_energy_fast(p)
        grad = _log_energy_grad(p)
        return energy, grad.flatten()
    
    # Check time budget
    elapsed = time.time() - start
    remaining = time_limit - elapsed
    if remaining < 1.0:
        return _normalize(pts), _log_energy_fast(_normalize(pts))
    
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                   options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12,
                            'maxfun': 5000, 'maxls': 50})
    
    x_opt = res.x.reshape(N, 3)
    x_opt = _normalize(x_opt)
    energy = _log_energy_fast(x_opt)
    
    return x_opt, energy

def _polish(pts, time_limit=5.0):
    """High-precision polish with adaptive step size and Newton-type refinement."""
    start = time.time()
    current = _normalize(pts.copy())
    energy = _log_energy_fast(current)
    step_size = 1e-2
    
    for it in range(8000):
        if time.time() - start > time_limit - 0.1:
            break
        
        grad = _log_energy_grad(current)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-15:
            break
        
        alpha = step_size
        accepted = False
        best_new_energy = energy
        best_new_pts = current.copy()
        
        for _ in range(50):
            new_pts = _exponential_map(current, -alpha * grad)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy < best_new_energy:
                best_new_energy = new_energy
                best_new_pts = new_pts
            
            if new_energy < energy - 1e-14:
                accepted = True
                break
            alpha *= 0.5
        
        if accepted:
            current = best_new_pts
            energy = best_new_energy
            # Adaptively adjust step size
            if alpha > 1e-6:
                step_size = min(alpha * 1.1, 0.05)
            else:
                step_size = alpha * 0.8
        else:
            step_size *= 0.4
            if step_size < 1e-25:
                break
    
    return current, energy

def construct():
    """Main function to construct optimal point configuration"""
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin before 150s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate initial configurations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Icosahedral-based initialization
    ico_pts = _icosahedral_init(N)
    initializations.append(ico_pts)
    
    # 3. Halton sequence
    halton_pts = _halton_sphere(N)
    initializations.append(halton_pts)
    
    # 4. Farthest-point sampling
    for seed in [42, 123, 456]:
        fpp = _farthest_point_init(N, seed=seed)
        initializations.append(fpp)
    
    # 5. Random uniform on sphere
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 6. Perturbed Fibonacci
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 7. Perturbed Icosahedral
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 789 + 23)
        pts_pert = ico_pts.copy() + rng_local.standard_normal((N, 3)) * 0.03
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 8. Perturbed farthest-point
    for seed in range(2):
        rng_local = np.random.default_rng(seed * 321 + 99)
        fpp_base = _farthest_point_init(N, seed=seed + 100)
        pts_pert = fpp_base + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # Evaluate each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 25:
            break
        
        current_pts = init_pts.copy()
        current_energy = _log_energy_fast(current_pts)
        
        # Phase 1: L-BFGS-B for coarse optimization
        try:
            if time.time() < deadline - 15:
                current_pts, current_energy = _lbfgs_optimize(
                    current_pts, time_limit=12.0, maxiter=1500)
        except Exception:
            pass
        
        # Phase 2: Riemannian CG for fine optimization
        try:
            if time.time() < deadline - 8:
                current_pts, current_energy = _riemannian_cg(
                    current_pts, time_limit=8.0, max_iter=150)
        except Exception:
            pass
        
        # Phase 3: Final L-BFGS-B polish
        try:
            if time.time() < deadline - 4:
                current_pts, current_energy = _lbfgs_optimize(
                    current_pts, time_limit=4.0, maxiter=500)
        except Exception:
            pass
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pts = current_pts.copy()
    
    # Final high-precision polish on best result
    if best_pts is not None and time.time() < deadline - 2:
        try:
            best_pts, best_energy = _polish(best_pts, time_limit=4.0)
        except Exception:
            pass
        
        # One more short L-BFGS pass if time allows
        if time.time() < deadline - 1:
            try:
                best_pts, best_energy = _lbfgs_optimize(
                    best_pts, time_limit=1.5, maxiter=200)
            except Exception:
                pass
    
    # Ensure valid output
    if best_pts is None:
        best_pts = fib_pts
    
    best_pts = _normalize(best_pts)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        return fib_pts
    
    return best_pts
# EVOLVE-BLOCK-END
