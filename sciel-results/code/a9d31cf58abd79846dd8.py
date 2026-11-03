"""Minimal logarithmic energy of 100 points on the sphere."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _log_energy(x):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)"""
    n = x.shape[0]
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    return -math.fsum(np.log(d).tolist())


def _log_energy_fast(x):
    """Fast energy computation using vectorized operations for inner optimization."""
    n = x.shape[0]
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    return float(-np.sum(np.log(d)))


def _log_energy_grad(x):
    """Compute gradient of logarithmic energy with respect to x."""
    n = x.shape[0]
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    coeff = 1.0 / dist_sq
    np.fill_diagonal(coeff, 0.0)
    
    coeff_sum = coeff.sum(axis=1, keepdims=True)
    grad = coeff @ x - coeff_sum * x
    return grad


def _log_energy_and_grad_fast(x):
    """Compute both energy and gradient in one pass for efficiency."""
    n = x.shape[0]
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    # Energy
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    e = float(-np.sum(np.log(d)))
    
    # Gradient
    coeff = 1.0 / dist_sq
    np.fill_diagonal(coeff, 0.0)
    coeff_sum = coeff.sum(axis=1, keepdims=True)
    grad = coeff @ x - coeff_sum * x
    
    return e, grad


def _project_to_sphere(x):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _golden_spiral(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed_offset)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(n, rng):
    """Generate random points on the unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _sobol_sphere(n, rng):
    """Generate quasi-random points on the sphere using a low-discrepancy sequence."""
    # Use a simple stratified approach for better coverage
    # Divide sphere into latitudinal bands and place points in each band
    i = np.arange(n)
    # Stratified latitudes
    z = 1.0 - 2.0 * (i + 0.5) / n
    # Add small random perturbation to break symmetry
    z = z + 0.01 * rng.standard_normal(n)
    z = np.clip(z, -0.9999, 0.9999)
    # Azimuthal angles with golden ratio spacing
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.standard_normal(n) * 0.1
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _tangent_project(grad, x):
    """Project gradient to tangent space of sphere at each point."""
    return grad - np.sum(grad * x, axis=1, keepdims=True) * x


def _rgeodesic_step(x, v, step_size):
    """
    Geodesic step on the sphere using the exponential map.
    v is a tangent vector at x.
    """
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    angles = step_size * norms
    # exp(v) = cos(angle) * x + sin(angle) * (v / |v|)
    sin_a = np.sin(angles) / norms
    cos_a = np.cos(angles)
    return cos_a * x + sin_a * v


def _optimize_lbfgs(x0, time_budget, maxiter=2000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from x0 with a time budget."""
    start_time = time.time()
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        e, g = _log_energy_and_grad_fast(x)
        g_tangent = _tangent_project(g, x)
        return e, g_tangent.flatten()
    
    x_flat = x0.flatten()
    best_x = x0.copy()
    best_e = _log_energy_fast(x0)
    
    try:
        res = minimize(
            lambda xf: objective_and_grad(xf),
            x_flat,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol, 'gtol': gtol}
        )
        x_opt = _project_to_sphere(res.x.reshape(-1, 3))
        e_opt = _log_energy_fast(x_opt)
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < time_budget * 0.5:
        try:
            res = minimize(
                lambda xf: objective_and_grad(xf),
                best_x.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol * 0.01, 'gtol': gtol * 0.01}
            )
            x_opt = _project_to_sphere(res.x.reshape(-1, 3))
            e_opt = _log_energy_fast(x_opt)
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x, best_e


def _rgeodesic_descent(x0, time_budget, lr_init=0.01, momentum=0.9):
    """
    Riemannian geodesic gradient descent with momentum.
    Uses the exponential map for proper sphere geometry.
    """
    start_time = time.time()
    x = x0.copy()
    e = _log_energy_fast(x)
    best_x = x.copy()
    best_e = e
    
    n = x.shape[0]
    v_mom = np.zeros_like(x)
    lr = lr_init
    lr_min = 1e-8
    
    for step in range(5000):
        if time.time() - start_time > time_budget:
            break
        
        e, grad = _log_energy_and_grad_fast(x)
        grad_tangent = _tangent_project(grad, x)
        
        # Adaptive learning rate
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm > 1e-12:
            lr = max(lr_min, lr_init / (1.0 + step * 0.01))
        
        # Momentum update in tangent space
        v_mom = momentum * v_mom + (1 - momentum) * grad_tangent
        
        # Geodesic step using exponential map
        x = _rgeodesic_step(x, -v_mom, lr)
        x = _project_to_sphere(x)  # Extra projection for numerical stability
        
        if e < best_e:
            best_e = e
            best_x = x.copy()
        
        # Early termination if convergence
        if grad_norm < 1e-10 and step > 100:
            break
    
    return best_x, best_e


def _simulated_annealing(x0, time_budget, rng, T_start=0.3, T_end=1e-8):
    """
    Simulated annealing on the sphere using Metropolis-Hastings with geodesic moves.
    """
    start_time = time.time()
    x = x0.copy()
    e = _log_energy_fast(x)
    best_x = x.copy()
    best_e = e
    
    n = x.shape[0]
    n_steps = 6000
    
    # Precompute step schedule
    temps = np.geomspace(T_start, T_end, n_steps)
    
    # Adaptive step size
    base_step = 0.03
    step_scale = 1.0
    accepted = 0
    total = 0
    window_size = 150
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        T = temps[step]
        
        # Adaptive step size: adjust based on recent acceptance rate
        if total > 0 and total % window_size == 0:
            accept_rate = accepted / total
            if accept_rate < 0.25:
                step_scale *= 1.4
            elif accept_rate > 0.55:
                step_scale *= 0.75
            step_scale = np.clip(step_scale, 0.1, 5.0)
            accepted = 0
            total = 0
        
        # Step size scales with temperature
        step_size = base_step * step_scale * (T + 0.01)
        
        # Generate random tangent vectors
        perturb = rng.standard_normal((n, 3)) * step_size
        # Project to tangent space
        perturb = perturb - np.sum(perturb * x, axis=1, keepdims=True) * x
        
        # Geodesic move
        x_prop = _rgeodesic_step(x, perturb, 1.0)
        x_prop = _project_to_sphere(x_prop)
        
        e_prop = _log_energy_fast(x_prop)
        
        # Metropolis criterion
        delta_e = e_prop - e
        total += 1
        if delta_e < 0 or (T > 1e-10 and rng.random() < math.exp(-delta_e / T)):
            x = x_prop
            e = e_prop
            accepted += 1
            
            if e < best_e:
                best_e = e
                best_x = x.copy()
    
    return best_x, best_e


def _anneal_and_polish(x0, time_budget, rng):
    """
    Combined simulated annealing followed by Riemannian geodesic descent polish.
    """
    start_time = time.time()
    sa_budget = time_budget * 0.5
    x_sa, e_sa = _simulated_annealing(x0, sa_budget, rng)
    
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2:
        x_pol, e_pol = _rgeodesic_descent(x_sa, remaining, lr_init=0.005)
        if e_pol < e_sa:
            return x_pol, e_pol
    
    return x_sa, e_sa


def _multi_point_swap(x0, time_budget, rng, n_swaps=100):
    """
    Try swapping pairs of points to escape local minima.
    """
    start_time = time.time()
    x = x0.copy()
    e = _log_energy_fast(x)
    best_x = x.copy()
    best_e = e
    
    n = x.shape[0]
    
    for _ in range(n_swaps):
        if time.time() - start_time > time_budget:
            break
        
        # Pick two random points to swap
        i, j = rng.choice(n, size=2, replace=False)
        
        # Try swapping
        x_swap = x.copy()
        x_swap[i], x_swap[j] = x_swap[j], x_swap[i]
        x_swap = _project_to_sphere(x_swap)
        
        e_swap = _log_energy_fast(x_swap)
        
        if e_swap < e:
            x = x_swap
            e = e_swap
            if e < best_e:
                best_e = e
                best_x = x.copy()
    
    return best_x, best_e


def _diffeomorphism_refinement(x0, time_budget, rng):
    """
    Refine using a parameterized diffeomorphism approach.
    Represent points as images of a reference grid under a smooth map.
    """
    start_time = time.time()
    
    # Reference grid: golden spiral
    ref = _golden_spiral(N)
    
    # Initialize deformation field: start with identity (zero deformation)
    # Use a small random initial deformation
    x = x0.copy()
    e = _log_energy_fast(x)
    best_x = x.copy()
    best_e = e
    
    # Perform gradient-based refinement with periodic random restarts
    for restart in range(5):
        if time.time() - start_time > time_budget:
            break
        
        # Small random perturbation for diversity
        if restart > 0:
            perturb = rng.standard_normal((N, 3)) * 0.05
            perturb = perturb - np.sum(perturb * x, axis=1, keepdims=True) * x
            x = _project_to_sphere(x + perturb)
            e = _log_energy_fast(x)
        
        # Riemannian geodesic descent
        remaining = time_budget - (time.time() - start_time)
        if remaining < 2:
            break
        
        x_opt, e_opt = _rgeodesic_descent(x, remaining * 0.8, lr_init=0.008)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
            x = x_opt
    
    return best_x, best_e


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # Golden spiral (baseline)
    inits.append(_golden_spiral(N))
    
    # Perturbed golden spirals
    for scale in [0.15, 0.08, 0.04, 0.02, 0.01]:
        perturbed = _golden_spiral(N) + scale * rng.standard_normal((N, 3))
        inits.append(_project_to_sphere(perturbed))
    
    # Sobol-like quasi-random
    inits.append(_sobol_sphere(N, rng))
    
    # Random points
    for _ in range(4):
        inits.append(_random_sphere_points(N, rng))
    
    # Different golden spiral offsets
    for offset in [1.0, 2.0, 0.5, 3.0, 1.5, 4.0, 0.3, 2.5]:
        inits.append(_golden_spiral(N, seed_offset=offset))
    
    # More random with different seeds
    for seed in [123, 456, 789, 1024, 2048]:
        r = np.random.default_rng(seed)
        inits.append(_random_sphere_points(N, r))
    
    best_x = None
    best_e = float('inf')
    
    num_inits = len(inits)
    
    for i, x0 in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed
        if remaining < 15:
            break
        
        # Allocate time budget for this optimization
        progress = i / max(1, num_inits)
        # Give more time to early inits, less to later ones
        time_budget = max(5.0, remaining * (1.2 - progress * 0.4) / max(1, num_inits - i))
        
        # Primary: Riemannian geodesic descent (more geometrically correct)
        x_opt, e_opt = _rgeodesic_descent(x0, time_budget * 0.5, lr_init=0.01)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
        
        # Check if we're running low on time
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 35:
            break
        
        # Secondary: L-BFGS-B polish
        remaining = TIME_LIMIT - elapsed
        if remaining > 15:
            x_lbfgs, e_lbfgs = _optimize_lbfgs(x_opt, min(8.0, remaining * 0.4), maxiter=1500, ftol=1e-14, gtol=1e-11)
            if e_lbfgs < best_e:
                best_e = e_lbfgs
                best_x = x_lbfgs.copy()
                x_opt = x_lbfgs
        
        # Tertiary: Simulated annealing for global exploration
        remaining = TIME_LIMIT - elapsed
        if remaining > 25:
            sa_budget = min(12.0, remaining * 0.35)
            x_sa, e_sa = _anneal_and_polish(x_opt, sa_budget, rng)
            
            if e_sa < best_e:
                best_e = e_sa
                best_x = x_sa.copy()
                x_opt = x_sa
        
        # Quaternary: Multi-point swap for escaping local minima
        remaining = TIME_LIMIT - elapsed
        if remaining > 20 and i < 10:
            swap_budget = min(5.0, remaining * 0.25)
            x_swap, e_swap = _multi_point_swap(x_opt, swap_budget, rng)
            
            if e_swap < best_e:
                best_e = e_swap
                best_x = x_swap.copy()
        
        # Quinary: Diffeomorphism refinement
        remaining = TIME_LIMIT - elapsed
        if remaining > 25 and i < 8:
            diff_budget = min(10.0, remaining * 0.3)
            x_diff, e_diff = _diffeomorphism_refinement(x_opt, diff_budget, rng)
            
            if e_diff < best_e:
                best_e = e_diff
                best_x = x_diff.copy()
    
    # Final polish with very tight tolerances - multiple passes
    remaining = TIME_LIMIT - time.time()
    if remaining > 10 and best_x is not None:
        # Riemannian geodesic descent polish
        x_final, e_final = _rgeodesic_descent(best_x, min(10.0, remaining * 0.5), lr_init=0.003)
        if e_final < best_e:
            best_e = e_final
            best_x = x_final.copy()
        
        # L-BFGS-B final polish
        remaining = TIME_LIMIT - time.time()
        if remaining > 5 and best_x is not None:
            x_final2, e_final2 = _optimize_lbfgs(best_x, min(6.0, remaining * 0.7), maxiter=3000, ftol=1e-16, gtol=1e-14)
            if e_final2 < best_e:
                best_e = e_final2
                best_x = x_final2.copy()
        
        # One more geodesic polish
        remaining = TIME_LIMIT - time.time()
        if remaining > 3 and best_x is not None:
            x_final3, e_final3 = _rgeodesic_descent(best_x, min(3.0, remaining * 0.8), lr_init=0.001)
            if e_final3 < best_e:
                best_e = e_final3
                best_x = x_final3.copy()
    
    if best_x is None:
        best_x = _golden_spiral(N)
    
    return best_x
