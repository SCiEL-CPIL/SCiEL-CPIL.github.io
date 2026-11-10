"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _energy_and_grad(x):
    """
    Compute logarithmic energy and gradient for N points on unit sphere.
    Uses dot-product based squared distances for speed.
    x: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is (N, 3).
    """
    # Compute squared distances via dot products: |x_i - x_j|^2 = 2 - 2*x_i·x_j
    dots = x @ x.T  # (N, N)
    dist_sq = 2.0 - 2.0 * dots  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # Avoid log(0) on diagonal
    
    # Energy: sum over i<j of -log(sqrt(dist_sq[i,j])) = -0.5 * sum over i<j of log(dist_sq[i,j])
    iu = np.triu_indices(N, k=1)
    energy = -0.5 * np.sum(np.log(dist_sq[iu]))
    
    # Gradient: grad[i] = -sum_{j != i} (x_i - x_j) / dist[i,j]^2
    # = -sum_{j != i} (x_i - x_j) / dist_sq[i,j]
    # Note: dist[i,j]^2 = dist_sq[i,j], so we divide by dist_sq not dist
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j inv_dist_sq[i,j] * (x_i - x_j)
    # = -x_i * sum_j inv_dist_sq[i,j] + sum_j inv_dist_sq[i,j] * x_j
    S = np.sum(inv_dist_sq, axis=1)  # (N,)
    W = inv_dist_sq @ x  # (N, 3)
    grad = -x * S[:, np.newaxis] + W
    
    return energy, grad

def _project_and_normalize(x):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return x / norms

def _fmin_on_sphere(x0, maxiter=10000, tol=1e-12, gtol=1e-8):
    """
    Minimize energy on the sphere using L-BFGS-B with sphere projection.
    """
    x = x0.copy()
    x = _project_and_normalize(x)
    
    def objective_and_grad(x_flat):
        x_2d = x_flat.reshape(N, 3)
        # Project to sphere
        x_2d = _project_and_normalize(x_2d)
        e, g = _energy_and_grad(x_2d)
        # Project gradient to tangent space
        g_tangent = g - (g * x_2d).sum(axis=1, keepdims=True) * x_2d
        return e, g_tangent.flatten()
    
    res = minimize(
        objective_and_grad,
        x.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': gtol, 'maxls': 50}
    )
    
    x_opt = _project_and_normalize(res.x.reshape(N, 3))
    return x_opt

def _simulated_annealing(x0, start_time, deadline, n_iterations=5000):
    """
    Simulated annealing to escape local minima with gradient-guided moves.
    """
    x = x0.copy()
    x = _project_and_normalize(x)
    current_e, current_g = _energy_and_grad(x)
    # Project gradient to tangent space
    current_g_tangent = current_g - (current_g * x).sum(axis=1, keepdims=True) * x
    best_x = x.copy()
    best_e = current_e
    
    # Temperature schedule
    T0 = 0.05
    T_min = 1e-7
    alpha = 0.998
    
    # Adaptive step size
    step_size = 0.03
    accept_count = 0
    total_count = 0
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        T = T0 * (alpha ** it)
        if T < T_min:
            break
        
        # Mix of gradient-guided and random moves
        if np.random.rand() < 0.5:
            # Gradient-guided move: move against gradient with some noise
            perturbation = -current_g_tangent * 0.1 + np.random.randn(N, 3) * step_size * 0.5
        else:
            # Pure random move
            perturbation = np.random.randn(N, 3) * step_size
        
        x_new = _project_and_normalize(x + perturbation)
        
        e_new, g_new = _energy_and_grad(x_new)
        g_new_tangent = g_new - (g_new * x_new).sum(axis=1, keepdims=True) * x_new
        delta_e = e_new - current_e
        
        # Acceptance criterion
        accepted = False
        if delta_e < 0 or (T > 0 and np.random.rand() < np.exp(-delta_e / T)):
            x = x_new
            current_e = e_new
            current_g_tangent = g_new_tangent
            accepted = True
            
            if current_e < best_e:
                best_e = current_e
                best_x = x.copy()
        
        # Adapt step size based on acceptance rate every 50 iterations
        total_count += 1
        if accepted:
            accept_count += 1
        
        if total_count % 50 == 0:
            accept_rate = accept_count / total_count
            if accept_rate > 0.65:
                step_size *= 1.15
            elif accept_rate < 0.35:
                step_size *= 0.85
            step_size = np.clip(step_size, 0.005, 0.2)
            accept_count = 0
            total_count = 0
    
    return best_x, best_e

def _energy(x):
    """Compute energy only."""
    e, _ = _energy_and_grad(x)
    return e

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_x = x0.copy()
    best_e = _energy(best_x)
    
    # Try multiple diverse initializations and optimize each
    np.random.seed(42)
    
    # Generate diverse starting points
    starts = [x0.copy()]
    
    # Perturbed Fibonacci spirals
    for pert_scale in [0.01, 0.05, 0.1, 0.15, 0.25]:
        x_pert = x0 + np.random.randn(N, 3) * pert_scale
        x_pert = _project_and_normalize(x_pert)
        starts.append(x_pert)
    
    # Halton sequence points
    def _halton_sequence(n, base):
        seq = []
        f = 1.0
        r = 0.0
        for i in range(1, n + 1):
            i_val = i
            while i_val > 0:
                f /= base
                r += f * (i_val % base)
                i_val //= base
            seq.append(r)
            r = 0.0
            f = 1.0
        return seq
    
    # 3D Halton points projected to sphere
    h1 = _halton_sequence(N, 2)
    h2 = _halton_sequence(N, 3)
    h3 = _halton_sequence(N, 5)
    # Map to sphere using spherical coordinates
    theta = 2 * np.pi * np.array(h1)
    phi = np.arccos(2 * np.array(h2) - 1)
    x_halton = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    starts.append(x_halton)
    
    # Random points
    for _ in range(8):
        x_rand = np.random.randn(N, 3)
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        starts.append(x_rand)
    
    # Antipodal pairs with slight perturbation
    x_anti = np.zeros((N, 3))
    for i in range(N // 2):
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
        x_anti[2*i] = v
        x_anti[2*i+1] = -v
    x_anti = x_anti + np.random.randn(N, 3) * 0.1
    x_anti = _project_and_normalize(x_anti)
    starts.append(x_anti)
    
    # Optimize each starting point
    for x_init in starts:
        if time.time() > deadline - 15.0:
            break
        
        try:
            x_opt = _fmin_on_sphere(x_init, maxiter=3000, tol=1e-10, gtol=1e-8)
            e_opt = _energy(x_opt)
            
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            continue
    
    # Simulated annealing to escape local minima - run multiple passes
    sa_deadline = deadline - 25.0
    if time.time() < sa_deadline:
        try:
            sa_x, sa_e = _simulated_annealing(best_x, start_time, sa_deadline, n_iterations=4000)
            if sa_e < best_e:
                best_e = sa_e
                best_x = sa_x.copy()
        except Exception:
            pass
    
    # If we still have time, do another SA pass from the best
    if time.time() < deadline - 35.0:
        try:
            sa_x2, sa_e2 = _simulated_annealing(best_x, start_time, deadline - 30.0, n_iterations=3000)
            if sa_e2 < best_e:
                best_e = sa_e2
                best_x = sa_x2.copy()
        except Exception:
            pass
    
    # Final high-precision polish with tighter tolerances
    if time.time() < deadline - 15.0:
        try:
            x_final = _fmin_on_sphere(best_x, maxiter=15000, tol=1e-14, gtol=1e-11)
            e_final = _energy(x_final)
            if e_final < best_e:
                best_e = e_final
                best_x = x_final
        except Exception:
            pass
    
    # One more polish if time permits
    if time.time() < deadline - 8.0:
        try:
            x_final2 = _fmin_on_sphere(best_x, maxiter=10000, tol=1e-16, gtol=1e-13)
            e_final2 = _energy(x_final2)
            if e_final2 < best_e:
                best_e = e_final2
                best_x = x_final2
        except Exception:
            pass
    
    # Ensure points are on unit sphere and valid
    best_x = _project_and_normalize(best_x)
    
    # Check for very close points
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # This should not happen, but if it does, return the Fibonacci spiral
        return x0
    
    return best_x
# EVOLVE-BLOCK-END
