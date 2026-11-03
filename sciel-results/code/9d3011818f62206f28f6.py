"""Logarithmic energy minimization for N points on the unit sphere.
Direct approach: use structured initialization followed by iterative gradient descent
with adaptive step sizes, exploiting the symmetry and smoothness of the energy landscape."""
import time
import math
import numpy as np

N = 100
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-16))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of -sum log(d_ij) with respect to points
    # d/dx_i [-log(d_ij)] = -(x_i - x_j) / d_ij^2
    # Sum over j != i: -sum_j (x_i - x_j)/d_ij^2 = -sum_j x_i/d_ij^2 + sum_j x_j/d_ij^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j = -W.sum(axis=1)*x_i + W @ x
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)
    
    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_like(n, seed=0):
    """Generate well-distributed points using a simple quasi-random approach."""
    rng = np.random.default_rng(seed)
    points = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        t = (i + 0.5) / n
        z = 1.0 - 2.0 * t
        angle = 2.0 * np.pi * i * (1.0 + 1.0 / golden_ratio) + rng.standard_normal() * 0.1
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        points.append([r * np.cos(angle), r * np.sin(angle), z])
    return _project(np.array(points))


def _gradient_descent(pts, deadline, max_iter=5000, lr_init=0.01, lr_min=1e-12):
    """Adaptive gradient descent with Riemannian exponential map updates."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    lr = lr_init
    n = N
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-13:
            break
        
        # Adaptive step size
        step = lr / max(g_norm, 1e-10)
        
        # Riemannian exponential map: move along geodesic
        # For the sphere, exp_p(v) = p*cos(|v|) + (v/|v|)*sin(|v|)
        v = -step * g  # negative gradient direction
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
        v_unit = v / v_norms_safe
        cos_t = np.cos(v_norms)
        sin_t = np.sin(v_norms)
        # Handle near-zero norms
        new_pts = current * cos_t + v_unit * sin_t
        new_pts = _project(new_pts)
        
        new_e = _energy(new_pts)
        
        if new_e < best_e - 1e-14:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            lr = min(lr * 1.05, 0.05)  # modest increase
        elif new_e < best_e:
            current = new_pts
            lr = lr * 0.95
        else:
            lr = lr * 0.5
            if lr < lr_min:
                break
    
    return best_pts, best_e


def _langevin_verlet(pts, deadline, rng, max_iter=2000, dt=0.01, gamma=0.5, T_init=0.5, T_min=1e-5):
    """
    Langevin dynamics with Velocity Verlet integrator on the sphere using Riemannian updates.
    """
    current = pts.copy()
    velocity = np.zeros_like(current)
    best_e = _energy(current)
    best_pts = current.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Cooling schedule
        T = T_init * (T_min / T_init) ** (it / max_iter)
        
        # 1. Half step velocity update with force
        e, g = _energy_grad(current)
        F = -g
        velocity += 0.5 * dt * F
        
        # Project velocity to tangent space
        dot_v = np.sum(velocity * current, axis=1, keepdims=True)
        velocity = velocity - dot_v * current
        
        # 2. Full step position update using exponential map
        v = dt * velocity
        v_norms = np.linalg.norm(v, axis=1, keepdims=True)
        v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
        v_unit = v / v_norms_safe
        cos_t = np.cos(v_norms)
        sin_t = np.sin(v_norms)
        new_pts = current * cos_t + v_unit * sin_t
        new_pts = _project(new_pts)
        
        # 3. Compute new force
        new_e, new_g = _energy_grad(new_pts)
        new_F = -new_g
        
        # 4. Half step velocity update with new force
        velocity += 0.5 * dt * new_F
        
        # 5. Apply friction and noise (Langevin kick)
        # Project velocity to tangent space of new_pts
        dot_v = np.sum(velocity * new_pts, axis=1, keepdims=True)
        velocity = velocity - dot_v * new_pts
        
        # Damping
        velocity *= (1.0 - gamma * dt)
        
        # Noise
        if T > T_min:
            noise_scale = np.sqrt(2.0 * gamma * T * dt)
            noise = noise_scale * rng.standard_normal((N, 3))
            # Project noise to tangent space
            dot_n = np.sum(noise * new_pts, axis=1, keepdims=True)
            noise = noise - dot_n * new_pts
            velocity += noise
        
        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()
            
        current = new_pts
        
    return best_pts, best_e

def _newton_like(pts, deadline, max_iter=2000, tol=1e-12):
    """Momentum-based gradient descent with Riemannian updates and adaptive LR."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    
    velocity = np.zeros_like(current)
    beta = 0.9
    lr = 0.03
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            break
        
        # Project velocity to tangent space
        dot_v = np.sum(velocity * current, axis=1, keepdims=True)
        velocity = velocity - dot_v * current
        
        velocity = beta * velocity - lr * g
        v_norms = np.linalg.norm(velocity, axis=1, keepdims=True)
        v_norms_safe = np.where(v_norms < 1e-12, 1.0, v_norms)
        v_unit = velocity / v_norms_safe
        cos_t = np.cos(v_norms)
        sin_t = np.sin(v_norms)
        new_pts = current * cos_t + v_unit * sin_t
        new_pts = _project(new_pts)
        
        new_e = _energy(new_pts)
        
        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()
            current = new_pts
            lr = min(lr * 1.02, 0.1)
        else:
            current = best_pts.copy()
            velocity *= 0.5
            lr *= 0.8
    
    return best_pts, best_e


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _scipy_polish(pts, deadline, max_iter=10000):
    """High-precision polish using scipy L-BFGS-B on flattened coordinates."""
    from scipy.optimize import minimize
    
    x0 = pts.flatten()
    
    def obj(x):
        p = x.reshape(N, 3)
        p = _project(p)
        return _energy(p)
    
    def grad(x):
        p = x.reshape(N, 3)
        p = _project(p)
        e, g = _energy_grad(p)
        # The gradient with respect to the unconstrained x is just g projected
        # since we re-project internally, this is approximate
        return g.flatten()
    
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxls': 50})
        best_pts = _project(res.x.reshape(N, 3))
        best_e = _energy(best_pts)
        return best_pts, best_e
    except Exception:
        return pts, _energy(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(10):
        jitter_scale = 0.02 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # Sobol-like points
    for s in range(5):
        initializations.append(_sobol_like(N, seed=s))
    
    best_pts = None
    best_energy = float('inf')
    
    # First pass: quick evaluation of all initializations
    for init_pts in initializations:
        if time.time() > deadline - 20.0:
            break
        
        e_quick = _energy(init_pts)
        if e_quick < best_energy:
            best_energy = e_quick
            best_pts = init_pts.copy()
    
    # Second pass: optimize the top candidates using gradient descent
    if best_pts is not None and time.time() < deadline - 15.0:
        # Sort initializations by quick energy and take top ones
        scored_inits = []
        for init_pts in initializations:
            e_quick = _energy(init_pts)
            scored_inits.append((e_quick, init_pts))
        scored_inits.sort(key=lambda x: x[0])
        
        # Optimize top 8 candidates
        top_candidates = scored_inits[:8]
        for e_quick, init_pts in top_candidates:
            if time.time() > deadline - 10.0:
                break
            
            # Try both methods
            gd_pts, gd_e = _gradient_descent(init_pts, deadline, max_iter=3000, lr_init=0.01)
            if gd_e < best_energy:
                best_energy = gd_e
                best_pts = gd_pts
            
            nl_pts, nl_e = _newton_like(init_pts, deadline, max_iter=1500)
            if nl_e < best_energy:
                best_energy = nl_e
                best_pts = nl_pts

            # Try Langevin dynamics
            if time.time() < deadline - 5.0:
                lv_pts, lv_e = _langevin_verlet(init_pts, deadline, rng, max_iter=1000)
                if lv_e < best_energy:
                    best_energy = lv_e
                    best_pts = lv_pts
    
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 10.0:
        refined, ref_e = _gradient_descent(best_pts, deadline, max_iter=5000, lr_init=0.005, lr_min=1e-14)
        if ref_e < best_energy:
            best_energy = ref_e
            best_pts = refined
    
    # Another refinement pass with momentum method
    if best_pts is not None and time.time() < deadline - 8.0:
        refined, ref_e = _newton_like(best_pts, deadline, max_iter=3000, tol=1e-14)
        if ref_e < best_energy:
            best_energy = ref_e
            best_pts = refined

        # Final Langevin polish
        if time.time() < deadline - 5.0:
            lv_pts, lv_e = _langevin_verlet(best_pts, deadline, rng, max_iter=1000, T_init=0.1)
            if lv_e < best_energy:
                best_energy = lv_e
                best_pts = lv_pts
    
    # High-precision scipy polish
    if best_pts is not None and time.time() < deadline - 2.0:
        refined, ref_e = _scipy_polish(best_pts, deadline, max_iter=5000)
        if ref_e < best_energy:
            best_energy = ref_e
            best_pts = refined
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
