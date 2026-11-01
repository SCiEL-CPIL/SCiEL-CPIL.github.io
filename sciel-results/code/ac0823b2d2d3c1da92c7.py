import time
import math
import numpy as np
from scipy.optimize import minimize

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

def _riemannian_cg(pts, time_limit=10.0, max_iter=300):
    """Riemannian conjugate gradient on the sphere using exponential maps."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    prev_grad = grad.copy()
    
    # Initial search direction
    d = -grad.copy()
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # Backtracking line search
        alpha = 0.1
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        
        for ls in range(60):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            # Sufficient decrease (Armijo)
            if new_energy > energy + c1 * alpha * np.sum(grad * d):
                break
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            if new_energy > energy:
                break
                
            alpha *= 0.7
            
            if alpha < 1e-14:
                break
                
        if best_alpha == 0.0:
            # Try a smaller step in the negative gradient direction
            alpha = 0.01
            new_pts = _exponential_map(current, alpha * (-grad))
            new_energy, new_grad = _energy_and_grad(new_pts)
            if new_energy < energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            else:
                break
        
        current = best_pts
        energy = best_energy
        grad = best_grad
        
        if np.linalg.norm(grad) < 1e-14:
            break
        
        # Polak-Ribiere beta
        d_new = -grad
        beta = 0.0
        if it > 0:
            num = np.sum(grad * (grad - prev_grad))
            den = np.sum(prev_grad * prev_grad)
            if den > 1e-30:
                beta = max(0.0, num / den)
        else:
            beta = 0.0
            
        d = d_new + beta * d
        prev_grad = grad.copy()
        
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

def _basin_hopping(pts, time_limit=30.0, max_restarts=8):
    """Perform local optimization followed by basin hopping with adaptive perturbation."""
    start = time.time()
    current = pts.copy()
    energy = _log_energy_fast(current)
    
    # Initial optimization using dual optimizer
    time_left_init = time_limit - (time.time() - start) - 3.0
    if time_left_init > 5.0:
        current, energy = _dual_optimizer(current, time_limit=min(15.0, time_left_init))
    
    best_pts = current.copy()
    best_energy = energy
    
    # Basin Hopping Loop
    temp = 1.0
    decay = 0.9
    
    for rh in range(max_restarts):
        if time.time() - start > time_limit - 2.0:
            break
        
        # Adaptive perturbation scale based on gradient magnitude
        _, grad = _energy_and_grad(current)
        grad_norm = np.linalg.norm(grad)
        perturb_scale = max(0.05, min(0.3, 0.1 * temp + 0.05 * grad_norm))
        
        # Generate random tangent vectors
        rng = np.random.default_rng()
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
        
        new_opt_pts, new_energy = _dual_optimizer(new_pts, time_limit=min(10.0, time_left))
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

def _tangent_basis(pts):
    """Create an orthonormal tangent basis for each point."""
    n = len(pts)
    bases = np.zeros((n, 3, 3))
    for i in range(n):
        p = pts[i]
        # Find a vector not parallel to p
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(p, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        t1 = ref - np.dot(p, ref) * p
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(p, t1)
        t2 /= np.linalg.norm(t2)
        bases[i, 0] = t1
        bases[i, 1] = t2
        bases[i, 2] = p
    return bases

def _tangent_to_cartesian(pts, bases, coords):
    """Convert tangent coordinates to Cartesian points on sphere."""
    n = len(pts)
    # coords: (n, 2) - angles in tangent plane
    theta = coords[:, 0:1]
    phi = coords[:, 1:2]
    # Exponential map: move along tangent vectors
    t1 = bases[:, 0]
    t2 = bases[:, 1]
    p = pts
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # Direction in tangent plane
    dir_t = np.cos(phi) * t1 + np.sin(phi) * t2
    # Exponential map on sphere
    new_pts = cos_t * p + sin_t * dir_t
    return _normalize(new_pts)

def _cartesian_to_tangent(pts, bases, new_pts):
    """Convert Cartesian points back to tangent coordinates."""
    n = len(pts)
    # Compute angle between old and new points
    cos_angle = np.sum(pts * new_pts, axis=1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    theta = np.arccos(cos_angle)
    # Direction in tangent plane
    t1 = bases[:, 0]
    t2 = bases[:, 1]
    # Project new_pts - cos(theta)*pts onto tangent plane
    residual = new_pts - cos_angle[:, np.newaxis] * pts
    # Project onto t1, t2
    proj_t1 = np.sum(residual * t1, axis=1)
    proj_t2 = np.sum(residual * t2, axis=1)
    phi = np.arctan2(proj_t2, proj_t1)
    return np.column_stack([theta, phi])

def _lbfgs_tangent_polish(pts, time_limit=30.0):
    """High-precision polish using L-BFGS-B in tangent coordinates."""
    start = time.time()
    n = len(pts)
    
    # Create tangent basis
    bases = _tangent_basis(pts)
    
    # Initial tangent coordinates (all zeros)
    x0 = np.zeros(2 * n)
    
    def objective_and_grad(x):
        # Reshape to (n, 2)
        coords = x.reshape(n, 2)
        # Convert to Cartesian
        new_pts = _tangent_to_cartesian(pts, bases, coords)
        # Compute energy and gradient
        energy, grad_cart = _energy_and_grad(new_pts)
        
        # Convert gradient to tangent coordinates
        # grad_cart is already projected to tangent space
        # We need to map it back to the (theta, phi) coordinate system
        # This is approximate but works well for small steps
        t1 = bases[:, 0]
        t2 = bases[:, 1]
        
        # Gradient with respect to theta and phi
        # For small angles, dE/dtheta ≈ grad . dir_t
        # dE/dphi ≈ grad . (perpendicular direction)
        
        # More careful: use finite differences or analytical mapping
        # For now, use a simple approximation
        grad_theta = np.zeros(n)
        grad_phi = np.zeros(n)
        
        for i in range(n):
            p = pts[i]
            t1_i = t1[i]
            t2_i = t2[i]
            g = grad_cart[i]
            
            # Gradient in tangent plane
            g_t1 = np.dot(g, t1_i)
            g_t2 = np.dot(g, t2_i)
            
            # Map to (theta, phi) coordinates
            # This is a local approximation
            grad_theta[i] = g_t1 * np.cos(coords[i, 1]) + g_t2 * np.sin(coords[i, 1])
            grad_phi[i] = -g_t1 * np.sin(coords[i, 1]) + g_t2 * np.cos(coords[i, 1])
        
        grad = np.concatenate([grad_theta, grad_phi])
        return energy, grad
    
    # Use L-BFGS-B
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if result.success or result.fun < _log_energy_fast(pts):
            # Convert back to Cartesian
            coords = result.x.reshape(n, 2)
            new_pts = _tangent_to_cartesian(pts, bases, coords)
            new_energy = _log_energy_fast(new_pts)
            if new_energy < _log_energy_fast(pts):
                return new_pts, new_energy
    
    except Exception:
        pass
    
    return pts, _log_energy_fast(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initializations - diverse set
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    inits.append(_spherical_design_init(N, seed=42))
    inits.append(_spherical_design_init(N, seed=123))
    
    # Random inits with different seeds
    for s in range(4):
        rng = np.random.default_rng(s)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        inits.append(pts)
        
    # Perturbed structured inits
    for s in range(3):
        rng = np.random.default_rng(s + 100)
        base = _fibonacci_sphere(N) if s % 2 == 0 else _halton_sphere(N)
        pts = base + rng.standard_normal((N, 3)) * 0.15
        pts = _normalize(pts)
        inits.append(pts)
        
    # Additional structured inits with different rotation seeds
    for s in [456, 789]:
        inits.append(_spherical_design_init(N, seed=s))
        
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
            
        # Time budget: give more time to structured inits
        if idx < 4:
            time_per_restart = 18.0
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
        
    # Final polish with tight tolerance using Riemannian CG
    if time.time() < deadline - 10.0:
        polish_time = min(15.0, deadline - time.time() - 8.0)
        try:
            final_pts, final_energy = _riemannian_cg(best_pts, time_limit=polish_time, max_iter=500)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    # High-precision L-BFGS polish in tangent coordinates
    if time.time() < deadline - 5.0:
        polish_time = min(10.0, deadline - time.time() - 3.0)
        try:
            final_pts, final_energy = _lbfgs_tangent_polish(best_pts, time_limit=polish_time)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
            
    # One more Riemannian polish if time allows
    if time.time() < deadline - 3.0:
        polish_time = deadline - time.time() - 2.0
        try:
            final_pts, final_energy = _riemannian_cg(best_pts, time_limit=polish_time, max_iter=300)
            final_energy = _log_energy_fast(final_pts)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass
    
    return _normalize(best_pts)
