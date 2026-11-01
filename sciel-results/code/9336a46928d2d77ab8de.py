import time
import math
import numpy as np

N = 150

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
    
    # Project gradient to tangent space: g - (g . p) p
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

def _cartesian_projected_grad(pts, lr=0.001, max_iter=50, time_limit=5.0):
    """Cartesian projected gradient descent with projection back to sphere.
    This can escape local minima that Riemannian methods get stuck in."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Take a step in Cartesian space
        step = lr * grad
        new_pts = current - step
        
        # Project back to sphere
        new_pts = _normalize(new_pts)
        
        new_energy, new_grad = _energy_and_grad(new_pts)
        
        # Simple backtracking if energy increases
        if new_energy > energy:
            lr *= 0.5
            continue
        
        current = new_pts
        energy = new_energy
        grad = new_grad
        
        if np.linalg.norm(grad) < 1e-12:
            break
        
    return current, energy

def _riemannian_lbfgs(pts, time_limit=10.0, max_iter=200, m=10):
    """Riemannian L-BFGS on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    # L-BFGS memory
    S = []  # step vectors (tangent)
    Y = []  # gradient differences
    W = []  # scaling factors
    
    # Initial search direction
    d = -grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # L-BFGS two-loop recursion to compute new direction
        if len(S) > 0:
            r = -grad.copy()
            alphas = []
            betas = []
            # Backward loop
            for i in range(len(S) - 1, -1, -1):
                si = S[i]
                yi = Y[i]
                wi = W[i]
                yi_norm_sq = np.sum(yi * yi)
                if yi_norm_sq > 1e-30:
                    beta_i = np.sum(si * r) / yi_norm_sq
                else:
                    beta_i = 0.0
                r = r - beta_i * yi
                alphas.append(beta_i)
                betas.append(0.0)
            
            # Initial Hessian approximation
            if len(S) > 0:
                s_last = S[-1]
                y_last = Y[-1]
                gamma = np.sum(s_last * y_last) / np.sum(y_last * y_last)
                if gamma < 1e-30:
                    gamma = 1.0
            else:
                gamma = 1.0
            
            r = gamma * r
            
            # Forward loop
            for i in range(len(S)):
                si = S[i]
                yi = Y[i]
                wi = W[i]
                alpha_i = alphas[i]
                if np.sum(yi * yi) > 1e-30:
                    beta_i = np.sum(yi * r) / np.sum(yi * yi)
                else:
                    beta_i = 0.0
                r = r + (alpha_i - beta_i) * si
                betas[i] = beta_i
            
            d = r
        
        # Backtracking line search
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        gd = np.sum(grad * d)
        
        for ls in range(40):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Sufficient decrease (Armijo)
            if new_energy > energy + c1 * alpha * gd:
                break
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            if new_energy > energy:
                break
                
            alpha *= 0.7
            
            if alpha < 1e-12:
                break
                
        if best_alpha == 0.0:
            break
            
        # Update L-BFGS memory
        step = best_alpha * d
        grad_diff = best_grad - grad
        step_norm = np.linalg.norm(step)
        grad_diff_norm = np.linalg.norm(grad_diff)
        
        if step_norm > 1e-10 and grad_diff_norm > 1e-10:
            S.append(step)
            Y.append(grad_diff)
            W.append(step_norm)
            if len(S) > m:
                S.pop(0)
                Y.pop(0)
                W.pop(0)
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-13:
            break
        
    return current, energy

def _dual_optimizer(pts, time_limit=20.0, max_rounds=5):
    """Alternate between Cartesian projected gradient and Riemannian CG.
    Cartesian steps can escape certain local minima, while Riemannian CG
    provides precise convergence on the sphere."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    for rnd in range(max_rounds):
        remaining = time_limit - (time.time() - start) - 1.0
        if remaining < 3.0:
            break
        
        # Phase 1: Riemannian CG for precise optimization
        rcg_time = min(remaining * 0.5, 10.0)
        if rcg_time > 2.0:
            current, energy = _riemannian_cg(current, time_limit=rcg_time, max_iter=200)
        
        remaining = time_limit - (time.time() - start) - 1.0
        if remaining < 2.0:
            break
        
        # Phase 2: Cartesian projected gradient to potentially escape
        cart_time = min(remaining * 0.4, 5.0)
        if cart_time > 1.0:
            current, cart_energy = _cartesian_projected_grad(
                current, lr=0.0005, max_iter=30, time_limit=cart_time
            )
            if cart_energy < energy:
                energy = cart_energy
        
        remaining = time_limit - (time.time() - start) - 1.0
        if remaining < 2.0:
            break
        
        # Phase 3: Final Riemannian polish
        polish_time = remaining
        if polish_time > 1.0:
            current, energy = _riemannian_cg(current, time_limit=polish_time, max_iter=100)
    
    return current, _log_energy_fast(current)

def _basin_hopping(pts, time_limit=30.0, max_restarts=10):
    """Perform local optimization followed by basin hopping with adaptive perturbation."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization using dual optimizer
    time_left_init = time_limit - (time.time() - start) - 3.0
    if time_left_init > 5.0:
        current, energy = _dual_optimizer(current, time_limit=min(12.0, time_left_init))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop with larger initial perturbations
    temp = 2.0
    decay = 0.85
    
    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2.0:
            break
        
        # Adaptive perturbation: larger early, smaller later
        # Start with larger perturbations to explore, shrink over time
        base_scale = 0.15 * temp + 0.03 * (1.0 / (1.0 + rh * 0.5))
        perturb_scale = max(0.05, min(0.4, base_scale))
        
        # Generate random tangent vectors
        rng = np.random.default_rng(rh * 7 + 13)
        perturbations = rng.standard_normal((N, 3))
        # Project to tangent space
        dot_prods = np.sum(perturbations * current, axis=1, keepdims=True)
        perturbations -= dot_prods * current
        # Normalize to unit length then scale
        pert_norms = np.linalg.norm(perturbations, axis=1, keepdims=True)
        pert_norms = np.maximum(pert_norms, 1e-15)
        perturbations = perturbations / pert_norms * perturb_scale
        
        new_pts = _exponential_map(current, perturbations)
        new_pts = _normalize(new_pts)
        
        # Optimize the perturbed point
        time_left = time_limit - (time.time() - start) - 1.0
        if time_left < 4.0:
            break
        
        opt_time = min(8.0, time_left * 0.7)
        new_opt_pts, new_energy = _dual_optimizer(new_pts, time_limit=opt_time)
        new_energy = _log_energy_fast(new_opt_pts)
        
        # Metropolis Criterion
        delta_e = new_energy - energy
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / max(temp, 1e-10)):
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

def _spherical_design_init(n, seed=42):
    """Generate points using a more structured approach with random rotations."""
    rng = np.random.default_rng(seed)
    # Start with Fibonacci, then apply random rotations
    pts = _fibonacci_sphere(n)
    # Apply a few random rotations to break symmetry
    for _ in range(3):
        # Random rotation matrix
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        pts = pts @ R.T
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initializations - diverse set
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_spherical_design_init(N, seed=42))
    inits.append(_spherical_design_init(N, seed=123))
    inits.append(_spherical_design_init(N, seed=999))
    
    # Random inits with different seeds
    for s in range(3):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed structured inits
    for s in range(3):
        rng = np.random.default_rng(s + 100)
        base = _fibonacci_sphere(N) if s % 2 == 0 else _halton_sphere(N)
        pts = base + rng.standard_normal((N, 3)) * 0.2
        pts = _normalize(pts)
        inits.append(pts)
        
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        # Time budget: give more time to structured inits
        if idx < 5:
            time_per_restart = 16.0
        else:
            time_per_restart = 12.0
        
        try:
            opt_pts, opt_energy = _basin_hopping(init_pts, time_limit=time_per_restart)
            opt_energy = _log_energy_fast(opt_pts)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
            
    # If no valid points found, fallback to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final high-precision polish with L-BFGS
    if time.time() < deadline - 8.0:
        polish_time = deadline - time.time() - 5.0
        try:
            final_pts, final_energy = _riemannian_lbfgs(best_pts, time_limit=polish_time, max_iter=300, m=15)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    # One more polish if time allows
    if time.time() < deadline - 4.0:
        polish_time = deadline - time.time() - 2.0
        try:
            final_pts, final_energy = _riemannian_lbfgs(best_pts, time_limit=polish_time, max_iter=200, m=15)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
    
    return _normalize(best_pts)
