import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE_MARGIN = 5.0

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
    """
    Compute energy and gradient for points on the unit sphere.
    Energy: E = sum_{i<j} log(1/||x_i - x_j||)
    Gradient (Euclidean): grad_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(pts)
    # Compute all pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid division by zero
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # Gradient: sum_j (x_i - x_j) * (1/||x_i - x_j||^2)
    # This is equivalent to: x_i * sum_j (1/d2_ij) - sum_j (x_j * 1/d2_ij)
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = pts * sum_inv_d2[:, np.newaxis] - weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    # Compute energy (upper triangle only)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dists_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = np.sum(np.log(1.0 / d))
    
    return energy, grad

def _exponential_map(pts, tangent_vecs):
    """Apply exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    # Handle zero norms
    norms_safe = np.where(norms < 1e-15, 1.0, norms)
    cos_norms = np.cos(norms_safe)
    sin_norms_over_norms = np.sin(norms_safe) / norms_safe
    # If norm is 0, sin/norm is 0, cos is 1, so result is pts
    # But we need to be careful with the 0/0 case.
    # Actually, if norm is 0, tangent_vec is 0, so result should be pts.
    # cos(0)=1, sin(0)/0 is limit 1? No, sin(x)/x -> 1.
    # Wait, the formula is cos(||v||) x + (sin(||v||)/||v||) v.
    # If v=0, term is 1*x + 1*0 = x. Correct.
    return cos_norms * pts + sin_norms_over_norms * tangent_vecs

def _project_to_tangent(pts, vecs):
    """Project vectors to tangent space at each point."""
    dot_prods = np.sum(vecs * pts, axis=1, keepdims=True)
    return vecs - dot_prods * pts

def _riemannian_lbfgs(pts, time_limit=10.0, max_iter=200, m=10):
    """
    Riemannian L-BFGS on the sphere.
    """
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    s_history = []
    y_history = []
    
    for it in range(max_iter):
        if time.time() - start > time_limit - 0.5:
            break
        
        # L-BFGS Two-Loop Recursion
        q = grad.copy()
        rho = []
        
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
        
        r = q
        d = r.copy()
        
        for k in range(len(rho)):
            s_k = s_history[k]
            y_k = y_history[k]
            ys = np.sum(y_k * s_k)
            if ys < 1e-30:
                continue
            rho_k = 1.0 / ys
            beta_k = rho_k * np.sum(y_k * r)
            d += (rho[k] - beta_k) * s_k
        
        # Ensure d is in tangent space
        d = _project_to_tangent(current, d)
        
        dir_norm = np.linalg.norm(d)
        if dir_norm < 1e-14:
            break
        
        # Line Search
        alpha = 0.5
        best_energy = energy
        best_pts = current.copy()
        best_grad = grad
        best_alpha = 0.0
        
        c1 = 1e-4
        directional_deriv = np.sum(grad * d)
        
        if directional_deriv > 0:
            d = -d
            directional_deriv = -directional_deriv
        
        for ls in range(30):
            new_pts = _exponential_map(current, alpha * d)
            new_energy, new_grad = _energy_and_grad(new_pts)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = new_pts
                best_grad = new_grad
                best_alpha = alpha
            
            if new_energy > energy + c1 * alpha * directional_deriv:
                break
            
            alpha *= 0.7
            if alpha < 1e-12:
                break
        
        if best_alpha == 0.0:
            break
        
        # Update History
        s_vec = best_pts - current
        y_vec = best_grad - grad
        
        s_vec = _project_to_tangent(current, s_vec)
        y_vec = _project_to_tangent(best_pts, y_vec)
        
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

def _repulsion_init(pts, steps=50):
    """
    Pre-spread points using repulsive forces.
    This helps escape bad local minima by spreading points out.
    """
    current = pts.copy()
    for _ in range(steps):
        energy, grad = _energy_and_grad(current)
        # Gradient of energy points towards lower energy (higher repulsion)
        # We want to minimize energy, so we move in direction of -grad
        # But grad is already the gradient of the energy to minimize.
        # So we move in direction -grad.
        step_size = 0.1
        tangent_step = -step_size * grad
        current = _exponential_map(current, tangent_step)
    return current

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

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    
    # Initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # Random inits
    for s in range(5):
        rng = np.random.default_rng(s * 7 + 1)
        pts = rng.standard_normal((N, 3))
        pts = _normalize(pts)
        # Pre-spread
        pts = _repulsion_init(pts, steps=20)
        inits.append(pts)
        
    # Perturbed Fibonacci
    for s in range(3):
        rng = np.random.default_rng(s * 13 + 200)
        scale = 0.1 + 0.05 * s
        pts = _fibonacci_sphere(N) + rng.standard_normal((N, 3)) * scale
        pts = _normalize(pts)
        pts = _repulsion_init(pts, steps=10)
        inits.append(pts)
        
    best_pts = None
    best_energy = float('inf')
    
    n_inits = len(inits)
    
    for idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        time_per_restart = min(20.0, max(5.0, (remaining - 5.0) / (n_inits - idx)))
        
        try:
            # Basin Hopping
            current = init_pts.copy()
            energy = _energy_and_grad(current)[0]
            
            # Local Opt
            opt_time = time_per_restart * 0.5
            current, energy = _riemannian_lbfgs(current, time_limit=opt_time)
            
            best_current = current.copy()
            best_e_current = energy
            
            # Basin Hopping Loop
            temp = 0.5
            decay = 0.8
            rh = 0
            while time.time() - start_time < deadline - 5.0 and rh < 10:
                time_left = deadline - time.time()
                if time_left < 5: break
                
                rng = np.random.default_rng(rh + 1000 + idx)
                scale = 0.2 * (0.9 ** rh)
                perturbations = rng.standard_normal((N, 3)) * scale
                perturbations = _project_to_tangent(current, perturbations)
                
                new_pts = _exponential_map(current, perturbations)
                
                opt_time = min(10.0, time_left - 2.0)
                if opt_time < 1: break
                
                new_opt_pts, new_energy = _riemannian_lbfgs(new_pts, time_limit=opt_time)
                
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
                    if energy < best_e_current:
                        best_e_current = energy
                        best_current = current.copy()
                
                temp *= decay
                rh += 1
            
            if best_e_current < best_energy:
                best_energy = best_e_current
                best_pts = best_current.copy()
                
        except Exception as e:
            pass

    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy = _log_energy_fast(best_pts)
        
    # Final Polish
    time_left = deadline - time.time()
    if time_left > 2:
        try:
            final_pts, final_energy = _riemannian_lbfgs(best_pts, time_limit=max(2.0, time_left - 1.0), max_iter=500, m=15)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts
        except Exception:
            pass

    return _normalize(best_pts)
