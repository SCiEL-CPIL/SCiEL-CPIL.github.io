"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100

# EVOLVE-BLOCK-START
def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy E = sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    grad_ambient = W @ pts - W.sum(axis=1, keepdims=True) * pts
    dot = np.sum(grad_ambient * pts, axis=1, keepdims=True)
    grad_tangent = grad_ambient - dot * pts
    
    return energy, grad_tangent

def _fast_energy(pts):
    """Compute energy quickly for initialization sorting."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _precise_energy(pts):
    """Compute energy using math.fsum for final scoring accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return pts / norms

def _generate_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_sobol(n, seed=42):
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed)
    samples = sampler.random(n)
    samples = samples - 0.5
    return _normalize(samples)

def _generate_random(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline):
    """
    Perform gradient flow with annealing and adaptive step control.
    Uses tangent-space gradient descent with backtracking line search.
    """
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _fast_energy(pts)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        # More aggressive initial learning rate decay
        lr = lr_start * (lr_end / lr_start) ** progress
        
        e, g = _energy_and_grad(current)
        
        # Add small noise for exploration (annealed)
        # Reduce noise magnitude to allow for more precise convergence later
        noise_scale = 0.005 * (1.0 - progress) ** 2.0
        noise = rng.standard_normal((n, 3)) * noise_scale
        
        # Try a few step sizes with backtracking
        accepted = False
        # Increase backtracking attempts to ensure we don't get stuck due to noise
        for attempt in range(8):
            trial_lr = lr * (0.5 ** attempt)
            # Ensure step is tangent
            trial = current - trial_lr * g
            trial = _normalize(trial)
            
            # Add noise only if the step is large enough, or always but scaled?
            # Better: Add noise to the direction before normalization?
            # Let's stick to adding noise to the point after normalization to keep it simple
            trial_noisy = trial + noise * (1.0 if attempt == 0 else 0.5)
            trial_noisy = _normalize(trial_noisy)
            
            e_trial = _fast_energy(trial_noisy)
            
            # Armijo condition: sufficient decrease
            # e_trial < e - 1e-4 * trial_lr * ||g||^2
            # For simplicity, just check if it's better than current
            if e_trial < e:
                current = trial_noisy
                e = e_trial
                accepted = True
                break
        
        if not accepted:
            # If no step improved, try a very small step without noise
            trial = current - (lr * 0.1) * g
            trial = _normalize(trial)
            e_trial = _fast_energy(trial)
            if e_trial < e:
                current = trial
                e = e_trial
            else:
                # Stagnation: accept noise step
                current = _normalize(current + noise)
                e = _fast_energy(current)
        
        if e < best_energy:
            best_energy = e
            best = current.copy()
    
    return best, best_energy

def _optimize_single(pts, deadline):
    """Run L-BFGS-B optimization from a given start."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.reshape(-1)
    
    x0 = pts.reshape(-1)
    
    # Phase 1: Standard L-BFGS-B
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 3000,
                'maxfun': 15000,
                'ftol': 1e-14,
                'gtol': 1e-10,
            }
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_pts, best_energy
    
    # Phase 2: Polish with tighter tolerances
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 5000,
                'maxfun': 25000,
                'ftol': 1e-16,
                'gtol': 1e-12,
            }
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
    except Exception:
        pass
    
    return best_pts, best_energy

def _final_polish(pts, deadline):
    """
    Final ultra-precise polish pass with very tight tolerances.
    Uses multiple L-BFGS restarts from slightly perturbed versions.
    """
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.reshape(-1)
    
    # Strategy: 
    # 1. Polish the current best.
    # 2. Generate several random perturbations of the best and polish them.
    # This helps if the current best is in a shallow local minimum that L-BFGS 
    # from the previous phase couldn't escape, but a small random kick + polish can.
    
    rng = np.random.default_rng(12345)
    
    # List of (start_points, description)
    starts = []
    starts.append((best_pts, "Original"))
    
    # Add perturbations
    # Scales chosen to be small enough to stay in the basin of attraction
    # but large enough to potentially hop out of a very flat region
    scales = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    for scale in scales:
        # Try 2 random perturbations per scale
        for _ in range(2):
            noise = rng.standard_normal((n, 3)) * scale
            start = _normalize(best_pts + noise)
            starts.append((start, f"Perturb_{scale}"))
    
    # Sort starts by energy (quick estimate) to prioritize good ones?
    # Or just process them. Since they are all close to best_pts, order doesn't matter much.
    # But processing the original first ensures we have a baseline.
    
    for start_pts, desc in starts:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective_and_grad,
                start_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 20000,
                    'maxfun': 100000,
                    'ftol': 1e-18,
                    'gtol': 1e-15,
                }
            )
            if result.fun < best_energy:
                new_pts = _normalize(result.x.reshape(n, 3))
                e_precise = _precise_energy(new_pts)
                if e_precise < best_energy:
                    best_pts = new_pts
                    best_energy = e_precise
        except Exception:
            pass
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral
    initializations.append(_generate_fibonacci(N))
    
    # 2. Sobol Sequences - better uniform coverage
    for seed in [1, 2, 3, 4, 5, 6, 7]:
        initializations.append(_generate_sobol(N, seed=seed))
    
    # 3. Random Points
    for _ in range(8):
        initializations.append(_generate_random(N, rng))
        
    # 4. Perturbed Fibonacci
    base = _generate_fibonacci(N)
    for scale in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))
    
    # 5. More structured initializations
    # Golden ratio based with different offsets
    for offset in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        i = np.arange(N) + 0.5 + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        initializations.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # 6. Lat-long grid with jitter
    lat = np.linspace(-np.pi/2, np.pi/2, 10)
    lon = np.linspace(0, 2*np.pi, 10)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    pts_grid = np.column_stack([
        np.cos(lat_grid) * np.cos(lon_grid),
        np.cos(lat_grid) * np.sin(lon_grid),
        np.sin(lat_grid)
    ]).reshape(-1, 3)
    pts_jittered = _normalize(pts_grid + rng.standard_normal((100, 3)) * 0.05)
    initializations.append(pts_jittered)
    
    # Sort initializations by quick energy to prioritize good starts
    scored_inits = []
    for init in initializations:
        e_quick = _fast_energy(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase A: Gradient flow exploration on top few initializations
    # Reduce flow time slightly to allow more time for L-BFGS and Polish
    flow_deadline = start_time + 25.0
    for e_quick, init_pts in scored_inits[:10]:
        if time.time() > flow_deadline:
            break
        
        # Increase steps in gradient flow since we have more precise backtracking
        flowed_pts, e_precise = _gradient_flow(init_pts, rng, 800, 0.05, 0.0003, flow_deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = flowed_pts
    
    # Phase B: L-BFGS optimization on all initializations (including flowed ones)
    if best_pts is not None:
        scored_inits.insert(0, (best_energy, best_pts))
    
    # Reserve 40s for final polish (it's crucial for precision)
    lbfgs_deadline = deadline - 40.0
    for e_quick, init_pts in scored_inits:
        if time.time() > lbfgs_deadline:
            break
            
        optimized_pts, e_precise = _optimize_single(init_pts, lbfgs_deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts

    if best_pts is None:
        best_pts = _generate_fibonacci(N)
    
    # Phase C: Final ultra-precise polish
    polish_deadline = deadline - 2.0  # 2s safety margin
    best_pts, best_energy = _final_polish(best_pts, polish_deadline)
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _generate_fibonacci(N)
        best_pts = _normalize(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
