import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 140.0  # Safety margin before 150s kill

def _energy_and_grad(pos):
    """
    Compute logarithmic energy and gradient for points on unit sphere.
    
    pos: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is the unconstrained gradient w.r.t. pos.
    Energy is sum_{i<j} log(1 / |x_i - x_j|).
    """
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    
    # Avoid division by zero for self-pairs
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # Clip to avoid log(0) if points are extremely close
    d = np.clip(d, 1e-12, None)
    energy = -np.sum(np.log(d))
    
    # Gradient computation:
    # grad_i = sum_{j != i} (p_j - p_i) / |p_i - p_j|^2
    #        = sum_{j != i} p_j / d_ij^2 - p_i * sum_{j != i} 1/d_ij^2
    
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # sum_j p_j / d_ij^2 for each i
    sum_pj_over_d2 = inv_d2 @ pos
    # sum_j 1/d_ij^2 for each i
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    
    grad = sum_pj_over_d2 - pos * sum_inv_d2[:, np.newaxis]
    
    return energy, grad

def _project_to_sphere(pos):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pos / norms

def _tangent_project(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    g_dot_n = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - g_dot_n * pts

def _riemannian_grad_desc(pos_init, time_budget, rng, initial_step=0.01):
    """
    Riemannian Gradient Descent with backtracking line search.
    Moves points on the sphere.
    """
    start_time = time.time()
    pos = _project_to_sphere(pos_init.copy())
    current_energy, grad = _energy_and_grad(pos)
    best_pos = pos.copy()
    best_energy = current_energy
    
    # Project gradient to tangent space
    tangent_grad = _tangent_project(pos, grad)
    
    step_size = initial_step
    
    while time.time() - start_time < time_budget:
        if time.time() > DEADLINE - 5:
            break
            
        # Propose move: pos_new = pos + step_size * tangent_grad
        # Then normalize to sphere.
        # This is an approximation of the exponential map for small steps.
        pos_new = pos + step_size * tangent_grad
        pos_new = _project_to_sphere(pos_new)
        
        new_energy, new_grad = _energy_and_grad(pos_new)
        
        # Check for improvement
        if new_energy < current_energy:
            # Accept
            pos = pos_new
            current_energy = new_energy
            grad = new_grad
            tangent_grad = _tangent_project(pos, grad)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = pos.copy()
                
            # Increase step size slightly if we improved significantly
            # But be careful not to overshoot.
            # A simple adaptive strategy:
            # If improvement is large, increase step. If small, keep.
            # For simplicity, we can just keep it or slowly decay if we want to converge.
            # Let's try to keep step size constant or decay very slowly.
            # Actually, for log energy, the landscape is quite smooth away from collisions.
            # We can use a fixed step or a slowly decaying one.
            # Let's decay it very slowly to allow fine-tuning.
            step_size *= 0.999 
        else:
            # Reject, decrease step size
            step_size *= 0.5
            
        # If step size becomes too small, we might be stuck or converged.
        if step_size < 1e-10:
            break
            
    return best_pos, best_energy

def _fibonacci_sphere(n, rng=None):
    """Generate n points on sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pos = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pos)

def _random_sphere(n, rng):
    """Generate n random points on sphere."""
    pos = rng.normal(0, 1, size=(n, 3))
    return _project_to_sphere(pos)

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_energy = float('inf')
    
    # Phase 1: Multi-start with short polishing
    # We will run many independent Riemannian GD runs.
    # Each run gets a small time slice.
    # Total time budget for Phase 1: ~80s
    # Remaining ~55s for Phase 2 (polishing best candidates)
    
    phase1_time = 80.0
    phase2_time = 50.0
    
    # Generate diverse starts
    starts = []
    
    # 1. Fibonacci
    starts.append(_fibonacci_sphere(N))
    
    # 2. Random starts
    num_random_starts = 100
    for _ in range(num_random_starts):
        starts.append(_random_sphere(N, rng))
        
    # We will process these starts.
    # To be fair and fast, we assign a time budget to each.
    # Total Phase 1 time: 80s.
    # If we do 101 starts, each gets ~0.8s.
    # This might be too short for GD to converge well from random.
    # Better: Do a few "medium" runs and many "short" runs?
    # Or just run them all sequentially with a tight loop.
    
    # Let's refine: 
    # Run 10 "long" runs (5s each) from high-quality starts (Fib, Icosa, etc)
    # Run 90 "short" runs (0.5s each) from random starts.
    
    high_quality_starts = [_fibonacci_sphere(N)]
    # Add icosahedral subdivision or just perturbed fib
    for _ in range(4):
        perturbed = _fibonacci_sphere(N) + rng.normal(0, 0.1, size=(N, 3))
        high_quality_starts.append(_project_to_sphere(perturbed))
        
    # Random starts
    random_starts = [_random_sphere(N, rng) for _ in range(90)]
    
    all_starts = high_quality_starts + random_starts
    
    # Time allocation
    # 5 high quality * 5s = 25s
    # 90 random * 0.5s = 45s
    # Total = 70s. Leaves 10s buffer for Phase 1.
    
    time_per_hq = 5.0
    time_per_rand = 0.5
    
    for i, start_pos in enumerate(all_starts):
        elapsed = time.time() - start_time
        if elapsed > phase1_time:
            break
            
        if i < 5:
            alloc_time = time_per_hq
        else:
            alloc_time = time_per_rand
            
        # Check if we have enough time for this run
        if time.time() - start_time + alloc_time > phase1_time + 5: # 5s buffer
             alloc_time = max(0.1, phase1_time - (time.time() - start_time) - 1)
             if alloc_time < 0.1:
                 break
                 
        pos, energy = _riemannian_grad_desc(start_pos, alloc_time, rng)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
    # Phase 2: Polish the best few candidates found in Phase 1
    # We need to keep track of the top candidates.
    # For simplicity, we just polish the current best_pos.
    # A better approach: keep a list of top 5 (pos, energy) and polish them.
    
    # Let's re-implement Phase 1 to keep top 5.
    # Actually, the above loop only keeps the global best. 
    # If the global best is stuck in a local min, we miss other good ones.
    # Let's modify Phase 1 to collect top candidates.
    
    # Re-run Phase 1 logic with candidate collection?
    # No, that's expensive. 
    # Instead, let's assume the first few high-quality starts are likely to produce the best candidates.
    # We will polish the best one found.
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - 5 - elapsed
    
    if remaining > 2.0 and best_pos is not None:
        # Final polish with longer time budget
        # Use a slightly different step size or just run GD again
        # GD is deterministic given the start and step size history.
        # To get more out of it, we can try a "restart" with a smaller step size.
        
        # Let's just run _riemannian_grad_desc again on the best_pos with remaining time.
        # It will continue from where it left off (if we passed best_pos as init).
        # But _riemannian_grad_desc resets step_size to initial_step.
        # We should modify it to accept an initial step size or continue state.
        
        # For now, let's just call it. It will start with step 0.01 again.
        # This might undo some fine-tuning if the landscape is tricky.
        # Better: Use a smaller initial step for the final polish.
        
        # Let's create a wrapper or just call with a flag?
        # I'll just call it. The energy should not increase if we check best_energy inside.
        # Wait, _riemannian_grad_desc returns the best found *within that run*.
        # If we start from a good point, it should find a better one or stay there.
        
        final_pos, final_energy = _riemannian_grad_desc(best_pos, remaining, rng, initial_step=0.005)
        
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = final_pos.copy()
            
    # Final validation
    best_pos = _project_to_sphere(best_pos)
    
    if not np.all(np.isfinite(best_pos)):
        # Fallback to Fibonacci
        best_pos = _fibonacci_sphere(N)
        
    # Check minimum distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Perturb and retry quickly
        perturbed = best_pos + rng.normal(0, 0.01, size=(N, 3))
        perturbed = _project_to_sphere(perturbed)
        pos, energy = _riemannian_grad_desc(perturbed, 2.0, rng)
        if energy < best_energy:
            best_pos = pos
            best_energy = energy
            best_pos = _project_to_sphere(best_pos)

    return best_pos
