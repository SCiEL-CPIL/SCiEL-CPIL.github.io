"""Bootstrap search program for packing 50 equal circles in a unit square."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
import random

N = 50

# Best known radius ~0.071377103865 (Boll et al. 2000)
# Seed grid gives radius = 0.5/7 ≈ 0.071428... wait let me check.
# m = ceil(sqrt(50)) = 8. Ticks spacing = 1/8 = 0.125. 
# min dist between adjacent = 0.125, so r = 0.0625.
# But the last column/row may be partially filled.
# Actually grid[:50] from an 8x8 grid: 8 full rows of 8 = 64, take 50.
# The spacing is 1/8 = 0.125, so radius = 0.0625.
# Score for seed = 0.0625.

def compute_radius(centres):
    """Compute the radius for a given set of centres."""
    if centres.shape != (N, 2):
        return 0.0
    if np.any(~np.isfinite(centres)):
        return 0.0
    # Check boundaries: all coords must be in [-0.5, 0.5]
    if np.any(np.abs(centres) > 0.5 + 1e-9):
        return 0.0
    # Compute min distance between pairs
    dists = pdist(centres, metric='euclidean')
    min_pair_dist = np.min(dists)
    # Compute min distance to boundary
    dist_to_walls = np.array([
        0.5 - np.abs(centres[:, 0]),  # left/right
        0.5 - np.abs(centres[:, 1])   # top/bottom
    ])
    min_wall_dist = np.min(dist_to_walls)
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return max(0.0, r)


def _energy(centres, target_r=None):
    """
    Energy function to minimize. Lower is better.
    Uses a smooth approximation of the max-min problem.
    """
    n = len(centres)
    if n != N:
        return 1e10
    
    # Boundary penalty: push points into square
    boundary_pen = 0.0
    for i in range(n):
        for k in range(2):
            if centres[i, k] < -0.5:
                boundary_pen += (-0.5 - centres[i, k]) ** 2
            elif centres[i, k] > 0.5:
                boundary_pen += (centres[i, k] - 0.5) ** 2
    
    if boundary_pen > 1e-6:
        return boundary_pen + 1e10
    
    # Pairwise penalty: smooth max(0, 2r - d)^2
    dists = pdist(centres, metric='euclidean')
    if target_r is None:
        # Estimate r from current configuration
        target_r = np.min(dists) / 2.0
    
    # Soft penalty for pairs that are too close
    excess = np.maximum(0, 2 * target_r - dists)
    pair_pen = np.sum(excess ** 2)
    
    return pair_pen


def _nelder_mead_step(x0, bounds=(-0.5, 0.5), maxiter=200, xatol=1e-6, fatol=1e-8):
    """Run Nelder-Mead optimization from x0."""
    res = minimize(_energy, x0, method='Nelder-Mead', 
                   options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol})
    return res.x, res.fun


def _force_directed(centres, n_iters=200, learning_rate=0.01, time_limit=None, start_time=None):
    """
    Force-directed relaxation: push overlapping circles apart, keep within bounds.
    This is a physics-inspired approach that works well for circle packing.
    """
    if time_limit is None:
        time_limit = 30
    if start_time is None:
        start_time = time.time()
    
    c = centres.copy().astype(float)
    best_c = c.copy()
    best_r = compute_radius(best_c)
    
    # Compute initial radius estimate
    dists = pdist(c, metric='euclidean')
    min_d = np.min(dists)
    r_est = min_d / 2.0
    
    for it in range(n_iters):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        # Repulsion force: proportional to 1/d^2 for close pairs, stronger for overlapping
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                if d < 1e-10:
                    # Random push for coincident points
                    forces[i, 0] += random.uniform(-0.01, 0.01)
                    forces[i, 1] += random.uniform(-0.01, 0.01)
                    forces[j, 0] -= random.uniform(-0.01, 0.01)
                    forces[j, 1] -= random.uniform(-0.01, 0.01)
                    continue
                
                # Strong repulsion when overlapping (d < 2*r_est)
                if d < 2 * r_est:
                    # Force magnitude: stronger when more overlapping
                    overlap = 2 * r_est - d
                    f_mag = 10.0 * overlap / d
                    unit_vec = diff[i, j] / d
                    forces[i] += f_mag * unit_vec
                    forces[j] -= f_mag * unit_vec
                else:
                    # Mild repulsion to spread out
                    f_mag = 0.1 / (d * d)
                    unit_vec = diff[i, j] / d
                    forces[i] += f_mag * unit_vec
                    forces[j] -= f_mag * unit_vec
        
        # Boundary forces: push points away from walls
        for i in range(N):
            for k in range(2):
                if c[i, k] > 0.5 - r_est:
                    forces[i, k] -= 5.0 * (c[i, k] - (0.5 - r_est))
                elif c[i, k] < -0.5 + r_est:
                    forces[i, k] += 5.0 * ((-0.5 + r_est) - c[i, k])
        
        # Update positions with adaptive learning rate
        lr = learning_rate * (0.95 ** (it // 10))
        c += lr * forces
        
        # Clamp to square
        np.clip(c, -0.5, 0.5, out=c)
        
        # Evaluate
        r = compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Update radius estimate
        dists = pdist(c, metric='euclidean')
        min_d = np.min(dists)
        r_est = max(min_d / 2.0, 0.01)
    
    return best_c, best_r


def _gradient_refine(centres, max_iters=50, time_limit=10, start_time=None):
    """
    Gradient-based refinement using L-BFGS-B with bounds.
    """
    if start_time is None:
        start_time = time.time()
    
    best_c = centres.copy()
    best_r = compute_radius(best_c)
    
    def objective(x):
        c = x.reshape(N, 2)
        # Smooth objective: minimize negative log-sum-exp of distances
        dists = pdist(c, metric='euclidean')
        # Wall distances
        wall_dists = np.concatenate([
            0.5 - c[:, 0],
            0.5 + c[:, 0],
            0.5 - c[:, 1],
            0.5 + c[:, 1]
        ])
        all_dists = np.concatenate([dists, wall_dists])
        # Use log-sum-exp for smooth min approximation
        max_d = np.max(all_dists)
        if max_d > 1e-10:
            logsumexp = max_d + np.log(np.sum(np.exp(all_dists - max_d)))
            return -logsumexp
        return -1e-10
    
    def gradient(x):
        c = x.reshape(N, 2)
        grad = np.zeros_like(x)
        
        dists = pdist(c, metric='euclidean')
        max_d = np.max(dists)
        exp_dists = np.exp(dists - max_d)
        sum_exp = np.sum(exp_dists)
        
        # Gradient of log-sum-exp w.r.t. each distance
        # d/dx log(sum(exp(d_i))) = sum(exp(d_i) * d(d_i)/dx) / sum(exp(d_i))
        
        # Pairwise gradients
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                if d < 1e-10:
                    continue
                idx = int(i * (N - 1) - i * (i + 1) // 2 + j - i - 1)
                if idx < len(dists):
                    weight = exp_dists[idx] / sum_exp
                    unit_vec = diff[i, j] / d
                    grad[2*i] += weight * unit_vec[0]
                    grad[2*i+1] += weight * unit_vec[1]
                    grad[2*j] -= weight * unit_vec[0]
                    grad[2*j+1] -= weight * unit_vec[1]
        
        # Wall gradients
        for i in range(N):
            for k in range(2):
                wall_idx = i * 4 + k
                if wall_idx < len(exp_dists):
                    weight = exp_dists[wall_idx] / sum_exp
                    # d(0.5 - c_ik)/dc_ik = -1
                    grad[2*i+k] += weight * (-1)
        
        return grad
    
    flat = best_c.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = minimize(objective, flat, jac=gradient, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': max_iters, 'ftol': 1e-8})
        new_c = res.x.reshape(N, 2)
        np.clip(new_c, -0.5, 0.5, out=new_c)
        new_r = compute_radius(new_c)
        if new_r > best_r:
            best_r = new_r
            best_c = new_c
    except Exception:
        pass
    
    return best_c, best_r


def _grid_init(n_cols=8):
    """Generate a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_init():
    """Generate a random initialization within the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))


def _structured_init(strategy='spiral'):
    """Generate structured initializations."""
    if strategy == 'spiral':
        # Archimedean spiral
        t = np.linspace(0, 4 * np.pi, N)
        r = 0.5 * t / (4 * np.pi)
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.column_stack([x, y])
    elif strategy == 'fibonacci':
        # Fibonacci lattice
        n = N
        i = np.arange(n)
        golden = (1 + np.sqrt(5)) / 2
        x = (i / golden) % 1.0 - 0.5
        y = i / n - 0.5
        return np.column_stack([x, y])
    elif strategy == 'lattice_mismatch':
        # Slightly perturbed lattice
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])[:N].copy()
        # Add small random perturbation
        grid += np.random.normal(0, 0.01, size=grid.shape)
        np.clip(grid, -0.5, 0.5, out=grid)
        return grid
    else:
        return _random_init()


def _local_search(centres, max_iters=100, step_scale=0.05):
    """
    Simple local search: try small random perturbations and keep improvements.
    """
    best = centres.copy()
    best_r = compute_radius(best)
    for it in range(max_iters):
        # Try a random perturbation
        delta = np.random.normal(0, step_scale, size=(N, 2))
        candidate = best + delta
        np.clip(candidate, -0.5, 0.5, out=candidate)
        r = compute_radius(candidate)
        if r > best_r:
            best = candidate
            best_r = r
    return best, best_r


def _simulated_annealing(centres, start_temp=0.1, end_temp=0.001, n_iters=500, time_limit=30, start_time=None):
    """
    Simulated annealing for circle packing.
    """
    if start_time is None:
        start_time = time.time()
    
    best = centres.copy()
    best_r = compute_radius(best)
    current = centres.copy()
    current_r = compute_radius(current)
    current_temp = start_temp
    
    for it in range(n_iters):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Anneal temperature
        current_temp = start_temp * (end_temp / start_temp) ** (it / n_iters)
        
        # Generate candidate
        delta = np.random.normal(0, current_temp, size=(N, 2))
        candidate = current + delta
        np.clip(candidate, -0.5, 0.5, out=candidate)
        r = compute_radius(candidate)
        
        # Accept if better or with probability exp((r - current_r) / current_temp)
        if r > current_r:
            current = candidate
            current_r = r
        else:
            accept_prob = np.exp((r - current_r) / max(current_temp, 1e-10))
            if random.random() < accept_prob:
                current = candidate
                current_r = r
        
        if current_r > best_r:
            best_r = current_r
            best = current.copy()
    
    return best, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin for 150s limit
    
    best_centres = _grid_init()
    best_r = compute_radius(best_centres)
    
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Force-directed relaxation from multiple initializations
    n_attempts = 0
    while time.time() < deadline - 30:
        n_attempts += 1
        strategies = ['grid', 'spiral', 'fibonacci', 'lattice_mismatch', 'random']
        strat = strategies[(n_attempts - 1) % len(strategies)]
        
        if strat == 'grid':
            centres = _grid_init()
        elif strat == 'spiral':
            centres = _structured_init('spiral')
        elif strat == 'fibonacci':
            centres = _structured_init('fibonacci')
        elif strat == 'lattice_mismatch':
            centres = _structured_init('lattice_mismatch')
        else:
            centres = _random_init()
        
        # Force-directed relaxation
        remaining = deadline - time.time()
        fd_time = min(20, remaining * 0.3)
        if fd_time < 5:
            break
        centres, r = _force_directed(centres, n_iters=300, learning_rate=0.005, 
                                      time_limit=fd_time, start_time=time.time())
        
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
    
    # Phase 2: Simulated annealing refinement on best result
    if time.time() < deadline - 20:
        remaining = deadline - time.time()
        sa_time = min(25, remaining * 0.4)
        if sa_time > 5:
            centres, r = _simulated_annealing(best_centres, start_temp=0.05, 
                                               end_temp=0.001, n_iters=1000,
                                               time_limit=sa_time, start_time=time.time())
            if r > best_r:
                best_r = r
                best_centres = centres.copy()
    
    # Phase 3: Gradient refinement
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        grad_time = min(15, remaining * 0.5)
        if grad_time > 3:
            centres, r = _gradient_refine(best_centres, max_iters=30, 
                                           time_limit=grad_time, start_time=time.time())
            if r > best_r:
                best_r = r
                best_centres = centres.copy()
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time()
        ls_iters = int(min(200, remaining * 20))
        if ls_iters > 10:
            centres, r = _local_search(best_centres, max_iters=ls_iters, step_scale=0.01)
            if r > best_r:
                best_r = r
                best_centres = centres.copy()
    
    # Final validation
    best_centres = np.clip(best_centres, -0.5, 0.5)
    if np.any(~np.isfinite(best_centres)):
        best_centres = _grid_init()
    
    return best_centres
