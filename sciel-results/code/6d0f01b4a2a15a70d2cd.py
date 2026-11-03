import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
import time
import math

N = 100
DEADLINE = 145.0  # Safety margin before 150s kill

def _energy_and_grad(points):
    """Compute logarithmic energy and Riemannian gradient for points on unit sphere."""
    n = points.shape[0]
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    
    log_d2 = np.log(d2)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    
    d2_safe = d2.copy()
    np.fill_diagonal(d2_safe, 1.0)
    W = 1.0 / d2_safe
    np.fill_diagonal(W, 0.0)
    
    row_sums = W.sum(axis=1, keepdims=True)
    grad = W @ points - points * row_sums
    
    radial = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - radial * points
    
    return energy, grad

def _energy_only(points):
    """Just the energy value (no gradient)."""
    n = points.shape[0]
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    log_d2 = np.log(d2)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_d2[iu])
    return energy

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _project_tangent(vec, point):
    """Project vector to tangent space at point on unit sphere."""
    radial = np.sum(vec * point, axis=1, keepdims=True)
    return vec - radial * point

def _exp_map(points, tangents):
    """Exponential map from tangent space to sphere."""
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    cos_factor = np.cos(norms)
    sin_factor = np.sin(norms) / norms
    return cos_factor * points + sin_factor * tangents

def _fibonacci_spiral(n, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _sobol_points(n, seed=42):
    """Generate well-distributed points using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed, scramble=True)
    samples = sampler.random(n)
    z = 2.0 * samples[:, 0] - 1.0
    theta = 2.0 * np.pi * samples[:, 1]
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return _normalize(pts)

def _lbfgs_optimize(points, max_iter=2000, ftol=1e-12, gtol=1e-8, deadline=None):
    """Run L-BFGS-B optimization with deadline guard."""
    start = time.time()
    
    def fun(x):
        if deadline is not None and time.time() > deadline:
            raise Exception("Deadline exceeded")
        return _energy_only(_normalize(x.reshape(N, 3)))
    
    def jac(x):
        if deadline is not None and time.time() > deadline:
            raise Exception("Deadline exceeded")
        pts = _normalize(x.reshape(N, 3))
        _, grad = _energy_and_grad(pts)
        return grad.flatten()
    
    try:
        result = minimize(
            fun=fun,
            x0=points.flatten(),
            jac=jac,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
        if result.fun < np.inf:
            return _normalize(result.x.reshape(N, 3)), result.fun
    except Exception:
        pass
    return points, _energy_only(points)

def _simulated_annealing(points, start_time, deadline, rng):
    """
    Simulated Annealing / Langevin Dynamics on the sphere.
    Uses a cooling schedule to explore the landscape.
    """
    current_pts = points.copy()
    current_energy = _energy_only(current_pts)
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    # Cooling schedule parameters
    # Start with high noise, decay to low noise
    initial_temp = 0.5
    final_temp = 0.001
    initial_lr = 0.01
    final_lr = 0.0001
    
    n_steps_est = int((deadline - start_time) * 1000) # Estimate steps
    if n_steps_est < 1000:
        n_steps_est = 1000
        
    # We will run until deadline
    step = 0
    while time.time() < deadline:
        # Calculate progress (0 to 1)
        elapsed = time.time() - start_time
        total_time = deadline - start_time
        if total_time <= 0: break
        progress = elapsed / total_time
        
        # Linearly interpolate temperature and learning rate
        temp = initial_temp * (1 - progress) + final_temp * progress
        lr = initial_lr * (1 - progress) + final_lr * progress
        
        # Compute gradient
        energy, grad = _energy_and_grad(current_pts)
        
        # Generate random tangent vectors
        noise = rng.standard_normal((N, 3))
        noise = _project_tangent(noise, current_pts)
        
        # Langevin update: v = -lr * grad + sqrt(2 * lr * temp) * noise
        # Note: The factor sqrt(2 * lr * temp) ensures correct equilibrium distribution
        noise_scale = np.sqrt(2.0 * lr * temp)
        v = -lr * grad + noise_scale * noise
        
        # Exponential map to stay on sphere
        new_pts = _exp_map(current_pts, v)
        new_pts = _normalize(new_pts)
        
        # Compute new energy
        new_energy = _energy_only(new_pts)
        
        # Metropolis-Hastings acceptance criterion
        if new_energy < current_energy:
            accept = True
        else:
            delta_e = new_energy - current_energy
            # Probability of accepting worse state
            # Avoid overflow in exp
            if delta_e > 0 and temp > 1e-10:
                accept_prob = math.exp(-delta_e / temp)
                accept = (rng.random() < accept_prob)
            else:
                accept = False
        
        if accept:
            current_pts = new_pts
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        step += 1
        
        # Check time frequently to avoid overshooting deadline
        if time.time() > deadline:
            break
            
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Standard Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Sobol points
    for seed in [42, 123, 456]:
        initializations.append(_sobol_points(N, seed=seed))
        
    # 3. Random points
    for _ in range(3):
        pts = rng.standard_normal((N, 3))
        initializations.append(_normalize(pts))
        
    # Pick the best initialization by energy
    best_init = initializations[0]
    best_init_energy = _energy_only(best_init)
    for init in initializations[1:]:
        e = _energy_only(init)
        if e < best_init_energy:
            best_init_energy = e
            best_init = init
            
    # Phase 1: Simulated Annealing / Langevin Dynamics
    # Allocate 80% of time to exploration
    sa_deadline = start_time + (DEADLINE * 0.8)
    
    sa_pts, sa_energy = _simulated_annealing(best_init, start_time, sa_deadline, rng)
    
    current_best_pts = sa_pts
    current_best_energy = sa_energy
    
    # Phase 2: Polish with L-BFGS-B
    # Allocate remaining 20% (minus safety margin) for polishing
    polish_deadline = deadline - 5.0
    
    if time.time() < polish_deadline:
        polish_pts, polish_energy = _lbfgs_optimize(
            current_best_pts, 
            max_iter=5000, 
            ftol=1e-15, 
            gtol=1e-14, 
            deadline=polish_deadline
        )
        if polish_energy < current_best_energy:
            current_best_energy = polish_energy
            current_best_pts = polish_pts
            
    # Ensure valid output
    current_best_pts = _normalize(current_best_pts)
    
    # Check for any points too close together and nudge if needed
    for _nudge in range(20):
        min_d = np.inf
        min_pair = (0, 1)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(current_best_pts[i] - current_best_pts[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
        if min_d >= 1e-9:
            break
        i, j = min_pair
        direction = current_best_pts[i] - current_best_pts[j]
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-15:
            direction = direction / norm_dir
        else:
            direction = rng.standard_normal(3)
            direction = direction / np.linalg.norm(direction)
        current_best_pts[i] = current_best_pts[i] + 1e-8 * direction
        current_best_pts[j] = current_best_pts[j] - 1e-8 * direction
        current_best_pts = _normalize(current_best_pts)
    
    # Final verification
    final_energy = _energy_only(current_best_pts)
    if not np.isfinite(final_energy):
        # Fallback to Fibonacci spiral
        current_best_pts = _fibonacci_spiral(N)
        
    return current_best_pts
