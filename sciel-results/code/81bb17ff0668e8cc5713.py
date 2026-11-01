import time
import math
import numpy as np

N = 150
DEADLINE = 150.0
MARGIN = 5.0  # Safety margin in seconds

def _energy_grad(pts):
    """
    Compute logarithmic energy and its R3 gradient.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t x_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, 1.0)
    
    d = np.sqrt(d2)
    
    iu = np.triu_indices(n, 1)
    d_upper = d[iu]
    
    if np.any(d_upper < 1e-12):
        return float('inf'), np.zeros_like(pts)
    
    # Energy calculation using fsum for precision
    terms = np.log(1.0 / d_upper)
    energy = math.fsum(terms.tolist())
    
    # Gradient calculation
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    col_sum = np.sum(inv_d2, axis=1)
    weighted = inv_d2 @ pts
    grad = pts * col_sum[:, None] - weighted
    
    return energy, grad

def _project(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms

def _retract(pts, v):
    """Retraction: project pts + v to the sphere."""
    return _project(pts + v)

def _fibonacci_sphere(n):
    """Fibonacci spiral initialization."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(pts)

def _halton_sphere(n):
    """Halton sequence initialization."""
    def halton(idx, base):
        result = 0.0
        f = 1.0
        i = idx
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    pts = []
    for i in range(1, n + 1):
        x, y, z = halton(i, 2), halton(i, 3), halton(i, 5)
        p = np.array([x, y, z])
        norm = np.linalg.norm(p)
        pts.append(p / norm if norm > 1e-15 else np.array([1.0, 0.0, 0.0]))
    return np.array(pts)

def _random_sphere(n, seed):
    """Random uniform initialization."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project(pts)

def _riemannian_lbfgs(pts0, time_limit, maxiter=1000, m=10, tol=1e-12):
    """
    Riemannian L-BFGS-B optimization on the sphere.
    """
    start = time.time()
    n = len(pts0)
    
    pts = pts0.copy()
    energy, grad_r3 = _energy_grad(pts)
    
    # Project gradient to tangent space
    dot = np.sum(pts * grad_r3, axis=1, keepdims=True)
    grad_t = grad_r3 - dot * pts
    
    if np.linalg.norm(grad_t) < tol:
        return pts, energy
    
    s_list = []
    y_list = []
    step_size = 1.0
    min_step = 1e-14
    
    for it in range(maxiter):
        if time.time() - start > time_limit:
            break
        
        gnorm = np.linalg.norm(grad_t)
        if gnorm < tol:
            break
        
        # Two-loop recursion for L-BFGS direction
        if len(s_list) == 0:
            direction = -grad_t
        else:
            q = grad_t.copy()
            rhos = []
            alphas = []
            
            # First loop (reverse)
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
            
            # Initial Hessian approximation
            last_s = s_list[-1]
            last_y = y_list[-1]
            sy = np.sum(last_s * last_y)
            if abs(sy) > 1e-16:
                gamma = np.sum(last_s * last_s) / sy
                if gamma < 1e-10 or gamma > 1e10:
                    gamma = 1.0
            else:
                gamma = 1.0
            
            r = q * gamma
            
            # Second loop (forward)
            for i in range(len(s_list)):
                si = s_list[i]
                yi = y_list[i]
                rho = rhos[len(s_list) - 1 - i]
                alpha_i = alphas[len(s_list) - 1 - i]
                beta = rho * np.sum(yi * r)
                r += si * (alpha_i - beta)
            
            direction = -r
        
        # Ensure direction is in tangent space
        dot_dir = np.sum(direction * pts, axis=1)
        direction -= dot_dir[:, np.newaxis] * pts
        
        # Line search (Backtracking)
        alpha = step_size
        accepted = False
        grad_dir = np.sum(grad_t * direction)
        
        for _ in range(40):
            new_pts = _retract(pts, alpha * direction)
            new_energy, new_grad_r3 = _energy_grad(new_pts)
            
            if new_energy == float('inf'):
                alpha *= 0.5
                continue
            
            # Armijo condition
            if new_energy < energy + 1e-4 * alpha * grad_dir:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            # Try a very small step
            alpha = max(step_size * 0.01, min_step)
            new_pts = _retract(pts, alpha * direction)
            new_energy, new_grad_r3 = _energy_grad(new_pts)
            
            if new_energy < energy and new_energy != float('inf'):
                accepted = True
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
        
        if not accepted:
            break
        
        # Update L-BFGS memory
        s = alpha * direction
        dot_new = np.sum(new_pts * new_grad_r3, axis=1, keepdims=True)
        new_grad_t = new_grad_r3 - dot_new * new_pts
        y = new_grad_t - grad_t
        sy = np.sum(s * y)
        
        if sy > 1e-14:
            if len(s_list) >= m:
                s_list.pop(0)
                y_list.pop(0)
            s_list.append(s)
            y_list.append(y)
        
        pts = new_pts
        grad_t = new_grad_t
        energy = new_energy
        step_size = min(alpha * 1.5, 2.0)
    
    return pts, energy

def _high_precision_polish(pts0, time_limit, maxiter=500, tol=1e-14):
    """
    High-precision final polish using gradient descent with very small steps.
    """
    start = time.time()
    pts = pts0.copy()
    energy, grad_r3 = _energy_grad(pts)
    dot = np.sum(pts * grad_r3, axis=1, keepdims=True)
    grad_t = grad_r3 - dot * pts
    
    if np.linalg.norm(grad_t) < tol:
        return pts, energy
    
    step_size = 1e-3
    min_step = 1e-16
    
    for it in range(maxiter):
        if time.time() - start > time_limit:
            break
        
        gnorm = np.linalg.norm(grad_t)
        if gnorm < tol:
            break
        
        direction = -grad_t / (gnorm + 1e-15)
        
        alpha = step_size
        accepted = False
        
        for _ in range(50):
            new_pts = _retract(pts, alpha * direction)
            new_energy, new_grad_r3 = _energy_grad(new_pts)
            
            if new_energy == float('inf'):
                alpha *= 0.5
                continue
            
            if new_energy < energy:
                accepted = True
                break
            alpha *= 0.5
        
        if not accepted:
            step_size *= 0.5
            if step_size < min_step:
                break
            continue
        
        dot_new = np.sum(new_pts * new_grad_r3, axis=1, keepdims=True)
        new_grad_t = new_grad_r3 - dot_new * new_pts
        
        pts = new_pts
        energy = new_energy
        grad_t = new_grad_t
        step_size = min(alpha * 1.2, 0.1)
    
    return pts, energy

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - MARGIN
    
    # Generate initializations
    inits = []
    inits.append(_fibonacci_sphere(N))
    inits.append(_halton_sphere(N))
    
    # Add a few random initializations
    for seed in [1, 42, 123]:
        inits.append(_random_sphere(N, seed))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Optimize each initialization
    # Allocate 60% of the time for this phase
    phase1_time = (DEADLINE - MARGIN) * 0.60
    per_init_time = phase1_time / len(inits)
    
    for idx, init_pts in enumerate(inits):
        current_time = time.time()
        time_left = deadline - current_time
        if time_left < 5.0:
            break
        
        # Use remaining time or allocated time, whichever is smaller
        t_alloc = min(per_init_time, time_left - 10.0) # Keep 10s buffer for polish
        if t_alloc < 1.0:
            break
        
        pts, energy = _riemannian_lbfgs(init_pts, t_alloc, maxiter=2000, m=15, tol=1e-12)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
        best_energy, _ = _energy_grad(best_pts)
    
    # Phase 2: High-precision polish on the best result
    current_time = time.time()
    time_left = deadline - current_time
    if time_left > 5.0:
        # Use most of the remaining time for polishing
        polish_time = time_left - 2.0
        pts, energy = _high_precision_polish(best_pts, polish_time, maxiter=1000, tol=1e-14)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final validation
    best_pts = _project(best_pts)
    valid = True
    for i in range(N):
        if not np.all(np.isfinite(best_pts[i])):
            valid = False
            break
        if np.linalg.norm(best_pts[i]) < 1e-10:
            valid = False
            break
    
    if valid:
        diff = best_pts[:, None, :] - best_pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        iu = np.triu_indices(N, 1)
        min_d = np.sqrt(d2[iu]).min()
        if min_d < 1e-9:
            valid = False
    
    if not valid:
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
