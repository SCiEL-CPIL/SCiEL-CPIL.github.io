import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
TIME_LIMIT = 150.0
DEADLINE_FRACTION = 0.90

def _energy_fast(pts):
    """Fast vectorized log energy for optimization."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    iu = np.triu_indices(n, k=1)
    return -0.5 * np.sum(np.log(dists2[iu]))

def _energy_grad_fast(pts):
    """Vectorized log energy and gradient (ambient)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, 1.0)
    
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists2[iu]))
    
    inv_dists2 = 1.0 / dists2
    np.fill_diagonal(inv_dists2, 0.0)
    
    s = (diff * inv_dists2[:, :, np.newaxis]).sum(axis=1)
    grad = -s
    
    return energy, grad

def _energy_fsum(pts):
    """Exact scoring energy using math.fsum."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    return math.fsum(np.log(1.0 / d).tolist())

def _project(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _tangent_project(pts, grad):
    coeffs = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - coeffs * pts

def _fibonacci_spiral(n=N):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _project(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _random_sphere(n=N, seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project(pts)

def _icosahedron_subdivide(n=N, seed=None):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _project(verts)
    if n <= 12:
        return verts[:n]
    rng = np.random.default_rng(seed)
    extra = rng.standard_normal((n - 12, 3))
    extra = _project(extra)
    return np.vstack([verts, extra])

def _riemannian_flow(pts, time_budget, initial_lr=0.05, min_lr=1e-10, patience=3000):
    """Riemannian gradient descent with backtracking line search."""
    pts = _project(pts.copy())
    current_energy = _energy_fast(pts)
    lr = initial_lr
    no_improve = 0
    start = time.time()
    
    while time.time() - start < time_budget:
        e, g = _energy_grad_fast(pts)
        g_t = _tangent_project(pts, g)
        grad_norm = np.linalg.norm(g_t)
        
        if grad_norm < 1e-12:
            break
        
        alpha = lr
        found = False
        for _ in range(40):
            new_pts = _project(pts - alpha * g_t)
            new_e = _energy_fast(new_pts)
            
            if new_e < current_energy - 1e-16 * alpha * grad_norm:
                pts = new_pts
                current_energy = new_e
                found = True
                break
            alpha *= 0.5
        
        if found:
            no_improve = 0
            lr = min(alpha * 1.3, 1.0)
        else:
            no_improve += 1
            lr = alpha * 0.5
            if no_improve > patience or lr < min_lr:
                break
    
    return pts

def _lbfgs_polish(pts, deadline):
    """L-BFGS-B polish with tight tolerances."""
    pts = _project(pts)
    x0 = pts.flatten()
    
    def func(x):
        p = _project(x.reshape(-1, 3))
        e, _ = _energy_grad_fast(p)
        return e
    
    def jac(x):
        p = _project(x.reshape(-1, 3))
        _, g = _energy_grad_fast(p)
        g_t = _tangent_project(p, g)
        return g_t.flatten()
    
    options = {
        'maxiter': 8000,
        'gtol': 1e-14,
        'ftol': 1e-18,
        'disp': False,
        'maxfun': 30000,
    }
    
    try:
        result = minimize(func, x0, jac=jac, method='L-BFGS-B', options=options)
        if result.fun < 1e17:
            return _project(result.x.reshape(-1, 3))
    except Exception:
        pass
    
    return pts

def _simulated_annealing(pts, deadline, seed=None):
    """Simulated annealing with rotation moves."""
    rng = np.random.default_rng(seed)
    pts = _project(pts.copy())
    current_energy = _energy_fast(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    T = 0.04
    min_T = 1e-7
    
    while time.time() < deadline:
        if T < min_T:
            break
        
        n_moves = rng.integers(1, 4)
        indices = rng.choice(N, size=n_moves, replace=False)
        
        new_pts = pts.copy()
        for idx in indices:
            angle = rng.uniform(-np.pi, np.pi) * T
            axis = rng.standard_normal(3)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-10:
                continue
            axis = axis / axis_norm
            
            c = math.cos(angle)
            s = math.sin(angle)
            k = axis
            k_outer_k = np.outer(k, k)
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * k_outer_k
            new_pts[idx] = R @ new_pts[idx]
        
        new_pts = _project(new_pts)
        new_energy = _energy_fast(new_pts)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(T, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        T *= 0.9995
    
    return best_pts

def _high_precision_polish(pts, deadline):
    """Final high-precision polish with very tight tolerances."""
    pts = _project(pts.copy())
    current_energy = _energy_fast(pts)
    
    lr = 0.01
    min_lr = 1e-15
    patience = 0
    
    while time.time() < deadline:
        e, g = _energy_grad_fast(pts)
        g_t = _tangent_project(pts, g)
        grad_norm = np.linalg.norm(g_t)
        
        if grad_norm < 1e-14:
            break
        
        alpha = lr
        found = False
        for _ in range(60):
            new_pts = _project(pts - alpha * g_t)
            new_e = _energy_fast(new_pts)
            
            if new_e < current_energy - 1e-18 * alpha * grad_norm:
                pts = new_pts
                current_energy = new_e
                found = True
                break
            alpha *= 0.5
        
        if found:
            patience = 0
            lr = min(alpha * 1.5, 0.1)
        else:
            lr = alpha
            patience += 1
            if patience > 100 or lr < min_lr:
                break
    
    return pts

def construct():
    global _start_time
    _start_time = time.time()
    deadline = _start_time + TIME_LIMIT * DEADLINE_FRACTION
    
    best_pts = None
    best_energy = float('inf')
    
    # Multiple diverse starting configurations
    rng = np.random.default_rng(12345)
    starts = [
        _fibonacci_spiral(N),
        _project(_fibonacci_spiral(N) + rng.normal(0, 0.05, (N, 3))),
        _project(_fibonacci_spiral(N) + rng.normal(0, 0.15, (N, 3))),
        _project(_fibonacci_spiral(N) + rng.normal(0, 0.3, (N, 3))),
        _random_sphere(N, seed=1),
        _random_sphere(N, seed=2),
        _random_sphere(N, seed=3),
        _random_sphere(N, seed=4),
        _icosahedron_subdivide(N, seed=42),
        _project(rng.standard_normal((N, 3))),
    ]
    
    # Remove very similar starts
    unique_starts = [starts[0]]
    for s in starts[1:]:
        if all(np.linalg.norm(s - u) > 0.3 for u in unique_starts):
            unique_starts.append(s)
    starts = unique_starts
    
    # Phase 1: Riemannian flow from multiple starts (45% of time)
    total_time = deadline - _start_time
    flow_deadline = _start_time + total_time * 0.45
    
    for i, start in enumerate(starts):
        if time.time() > flow_deadline:
            break
        
        # Quick Riemannian flow
        budget = min(8.0, max(1.0, (flow_deadline - time.time()) / max(1, len(starts) - i)))
        refined = _riemannian_flow(start, budget)
        e = _energy_fast(refined)
        
        if e < best_energy:
            best_energy = e
            best_pts = refined
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_fast(best_pts)
    
    # Phase 2: Simulated annealing to escape local minima (25% of time)
    sa_deadline = _start_time + total_time * 0.70
    if time.time() < sa_deadline:
        sa_result = _simulated_annealing(best_pts, sa_deadline, seed=999)
        sa_energy = _energy_fast(sa_result)
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_pts = sa_result
    
    # Phase 3: L-BFGS-B polish (20% of time)
    lbfgs_deadline = _start_time + total_time * 0.90
    if time.time() < lbfgs_deadline:
        polished = _lbfgs_polish(best_pts, lbfgs_deadline)
        pe = _energy_fast(polished)
        if pe < best_energy:
            best_energy = pe
            best_pts = polished
    
    # Phase 4: Final high-precision polish (remaining time)
    if time.time() < deadline - 2.0:
        final = _high_precision_polish(best_pts, deadline)
        fe = _energy_fast(final)
        if fe < best_energy:
            best_energy = fe
            best_pts = final
    
    best_pts = _project(best_pts)
    
    # Final validity check
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_spiral(N)
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    if min_d < 1e-9:
        return _fibonacci_spiral(N)
    
    return best_pts
