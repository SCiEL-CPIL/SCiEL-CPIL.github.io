import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 50
DEADLINE = 140.0  # safety margin before 150s kill

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall = np.min(0.5 - np.abs(centers))
    if wall <= 0:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    i_idx, j_idx = np.triu_indices(N, k=1)
    min_pair = np.min(dist[i_idx, j_idx]) / 2.0
    return min(wall, min_pair)

def _min_dist_and_grad(centers):
    """
    Return the minimum distance (radius) and its subgradient.
    The minimum distance is min(wall_dists, pair_dists/2).
    We return a vector that points in a direction that increases the minimum.
    """
    wall_dists = 0.5 - np.abs(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-20)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dist[i_idx, j_idx] / 2.0
    
    all_dists = np.concatenate([wall_dists.ravel(), d_ij])
    min_d = np.min(all_dists)
    
    # Find which constraints are active (within tolerance)
    tol = 1e-8
    active = all_dists <= min_d + tol
    
    grad = np.zeros_like(centers)
    
    if np.any(active[:N*3]):
        # Wall constraints active
        wall_active = active[:N*3].reshape(N, 3)
        # For wall: d = 0.5 - |c|, grad_d/dc = -sign(c)
        # We want to increase d, so move in direction +sign(c)
        grad += wall_active * np.sign(centers)
    
    if np.any(active[N*3:]):
        # Pair constraints active
        pair_active = active[N*3:]
        i_act = i_idx[pair_active]
        j_act = j_idx[pair_active]
        vecs = centers[i_act] - centers[j_act]
        d_masked = dist[i_act, j_act]
        # d = |c_i - c_j|/2, grad_d/dc_i = vec/d, grad_d/dc_j = -vec/d
        coeff = 1.0 / d_masked
        np.add.at(grad, i_act, coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_act, -coeff[:, np.newaxis] * vecs)
    
    # Normalize gradient
    norm = np.linalg.norm(grad)
    if norm > 1e-15:
        grad /= norm
    
    return min_d, grad

def _generate_initial_guess(seed=0):
    """Generate diverse initial guesses."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((N, 3))
    strategy = seed % 8
    
    if strategy == 0:
        # 5x5x2 grid
        pts = []
        for i in range(5):
            for j in range(5):
                for k in range(2):
                    x = -0.4 + i * 0.2
                    y = -0.4 + j * 0.2
                    z = -0.2 + k * 0.4
                    pts.append([x, y, z])
        centers[:50] = np.array(pts[:50])
        centers += rng.normal(0, 0.02, (N, 3))
        
    elif strategy == 1:
        # BCC-like
        pts = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.4
                    y = (j - 1) * 0.4
                    z = (k - 1) * 0.4
                    pts.append([x, y, z])
                    if i < 2 and j < 2 and k < 2:
                        pts.append([x + 0.2, y + 0.2, z + 0.2])
        centers[:min(35, N)] = np.array(pts[:min(35, N)])
        if N > 35:
            centers[35:] = rng.uniform(-0.35, 0.35, (N-35, 3))
            
    elif strategy == 2:
        # Random uniform
        centers = rng.uniform(-0.4, 0.4, (N, 3))
        
    elif strategy == 3:
        # Two-layer 5x5 staggered
        bottom = []
        top = []
        for i in range(5):
            for j in range(5):
                x = -0.4 + i * 0.2
                y = -0.4 + j * 0.2
                bottom.append([x, y, -0.15])
                tx = -0.3 + i * 0.2
                ty = -0.3 + j * 0.2
                top.append([tx, ty, 0.15])
        centers[:25] = np.array(bottom)
        centers[25:] = np.array(top)
        centers += rng.normal(0, 0.015, (N, 3))
        
    elif strategy == 4:
        # FCC-like
        pts = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x = (i - 1) * 0.35
                    y = (j - 1) * 0.35
                    z = (k - 1) * 0.35
                    pts.append([x, y, z])
                    if i < 2:
                        pts.append([x + 0.175, y, z])
                    if j < 2:
                        pts.append([x, y + 0.175, z])
                    if k < 2:
                        pts.append([x, y, z + 0.175])
        pts = np.array(pts[:N])
        centers = pts + rng.normal(0, 0.02, (N, 3))
        
    elif strategy == 5:
        # Hexagonal layers
        pts = []
        layer_z = [-0.3, -0.1, 0.1, 0.3]
        for li, z in enumerate(layer_z):
            n_per_layer = 13 if li % 2 == 0 else 12
            for i in range(n_per_layer):
                row = i // 4
                col = i % 4
                x = -0.36 + col * 0.24 + (0.12 if row % 2 else 0)
                y = -0.3 + row * 0.2
                pts.append([x, y, z])
        centers[:N] = np.array(pts[:N])
        centers += rng.normal(0, 0.02, (N, 3))
        
    elif strategy == 6:
        # 4x4x3 + 14 extras
        pts = []
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    x = -0.375 + i * 0.25
                    y = -0.375 + j * 0.25
                    z = -0.333 + k * 0.333
                    pts.append([x, y, z])
        centers[:48] = np.array(pts[:48])
        centers[48:] = rng.uniform(-0.3, 0.3, (2, 3))
        centers += rng.normal(0, 0.01, (N, 3))
        
    else:
        # Simple cubic 4x4x3 with jitter
        pts = []
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    x = -0.375 + i * 0.25
                    y = -0.375 + j * 0.25
                    z = -0.333 + k * 0.333
                    pts.append([x, y, z])
        centers[:48] = np.array(pts[:48])
        centers[48:] = rng.uniform(-0.3, 0.3, (2, 3))
        centers += rng.normal(0, 0.015, (N, 3))
    
    centers = np.clip(centers, -0.48, 0.48)
    return centers

def _simulated_annealing(centers, time_budget, rng):
    """
    Simulated annealing on the minimum distance objective.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Initial step size
    step = 0.03
    T = 0.01  # temperature
    
    # Adaptive: shrink step as we improve
    improvement = 0.0
    stagnation = 0
    
    while time.time() - start_time < time_budget:
        # Generate perturbation
        delta = rng.normal(0, step, (N, 3))
        proposed = current + delta
        proposed = np.clip(proposed, -0.48, 0.48)
        
        proposed_r = _compute_radius(proposed)
        
        # Accept if better, or with probability exp(-delta/T)
        delta_r = current_r - proposed_r
        if proposed_r > current_r or (T > 1e-10 and rng.random() < np.exp(-delta_r / T)):
            current = proposed
            current_r = proposed_r
            stagnation = 0
        else:
            stagnation += 1
        
        if current_r > best_r:
            best_r = current_r
            best = current.copy()
            improvement = current_r - best_r
            # Cool down
            T *= 0.95
            # Shrink step slightly on improvement
            step *= 0.995
        else:
            # If stagnating, occasionally reheat
            if stagnation > 500:
                T = max(T, 0.005)
                step = max(step, 0.01)
                stagnation = 0
        
        # Periodically cool
        if stagnation > 100:
            T *= 0.9
            step *= 0.98
            stagnation = 0
        
        # Floor on step size
        step = max(step, 0.0005)
        T = max(T, 1e-8)
    
    return best, best_r

def _nelder_mead_polish(centers, time_budget):
    """
    Polish using Nelder-Mead on the negative minimum distance.
    """
    start_time = time.time()
    
    def objective(x):
        c = x.reshape(N, 3)
        c = np.clip(c, -0.48, 0.48)
        return -_compute_radius(c)
    
    x0 = centers.ravel()
    bounds = [(-0.48, 0.48)] * (N * 3)
    
    try:
        res = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True,
            }
        )
        polished = res.x.reshape(N, 3)
        polished = np.clip(polished, -0.48, 0.48)
        return polished
    except Exception:
        return centers

def _local_refine(centers, time_budget):
    """
    Local refinement using gradient-based ascent on the minimum distance.
    Uses the subgradient to move in a direction that increases the minimum.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    step = 0.005
    
    while time.time() - start_time < time_budget:
        min_d, grad = _min_dist_and_grad(current)
        
        if np.linalg.norm(grad) < 1e-12:
            break
        
        proposed = current + step * grad
        proposed = np.clip(proposed, -0.48, 0.48)
        proposed_r = _compute_radius(proposed)
        
        if proposed_r > current_r:
            current = proposed
            current_r = proposed_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
            step *= 1.05  # increase step on success
            step = min(step, 0.02)
        else:
            step *= 0.7  # decrease step on failure
            if step < 0.0001:
                break
    
    return best, best_r

def construct():
    """Construct optimal sphere packing."""
    start_time = time.time()
    
    best_radius = 0.0
    best_centers = None
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated annealing from multiple starts
    num_sa_restarts = 12
    sa_time_per = (DEADLINE - 40) / num_sa_restarts  # reserve 40s for polishing
    
    candidates = []
    
    for restart_idx in range(num_sa_restarts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 40:
            break
        
        centers = _generate_initial_guess(seed=restart_idx * 7 + 3)
        
        remaining = DEADLINE - 40 - elapsed
        time_per = min(sa_time_per, remaining / max(1, num_sa_restarts - restart_idx))
        
        try:
            sa_centers, sa_r = _simulated_annealing(centers, time_per, rng)
            candidates.append((sa_r, sa_centers))
            
            if sa_r > best_radius:
                best_radius = sa_r
                best_centers = sa_centers.copy()
                
        except Exception:
            pass
    
    # Phase 2: Local gradient refinement on top candidates
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_k = min(5, len(candidates))
        refine_time_per = max(1.0, (DEADLINE - 20 - elapsed) / max(1, top_k))
        
        for r, c in candidates[:top_k]:
            if time.time() - start_time > DEADLINE - 15:
                break
            try:
                refined, refined_r = _local_refine(c, refine_time_per)
                if refined_r > best_radius:
                    best_radius = refined_r
                    best_centers = refined.copy()
            except Exception:
                pass
    
    # Phase 3: Nelder-Mead polish on the best result
    if best_centers is not None and time.time() - start_time < DEADLINE - 10:
        try:
            nm_time = min(20.0, DEADLINE - 5 - (time.time() - start_time))
            if nm_time > 2.0:
                polished = _nelder_mead_polish(best_centers, nm_time)
                polished_r = _compute_radius(polished)
                if polished_r > best_radius:
                    best_radius = polished_r
                    best_centers = polished
        except Exception:
            pass
    
    # Final validation
    if best_centers is None:
        best_centers = _generate_initial_guess(seed=99)
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.48, 0.48)
    if not np.all(np.isfinite(best_centers)):
        best_centers = _generate_initial_guess(seed=99)
    
    return best_centers
