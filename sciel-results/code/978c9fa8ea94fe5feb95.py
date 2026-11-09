"""Optimized packing of 50 equal circles in a unit square."""
import numpy as np
import time
import random
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _score(centres):
    """Compute the minimum radius: min of half-pair-distances and wall distances."""
    # Wall distances
    wall = 0.5 - np.abs(centres)
    min_wall = np.min(wall)
    
    # Pair distances using cdist for speed
    dist = cdist(centres, centres)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_wall, min_pair)


def _score_fast(centres):
    """Faster scoring for annealing - uses squared distances where possible."""
    # Wall distances
    wall = 0.5 - np.abs(centres)
    min_wall = np.min(wall)
    
    # Approximate pair check: only check if any pair is too close
    # Use cdist but early exit logic isn't easy, so just compute
    dist = cdist(centres, centres)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_wall, min_pair)


def _construct_grid():
    """Create initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _construct_honeycomb():
    """Create a honeycomb-like initial configuration for better packing."""
    # Use a hexagonal lattice pattern
    centres = []
    for row in range(8):
        for col in range(8):
            if len(centres) >= N:
                break
            x = (col - 3.5) * 0.25
            y = (row - 3.5) * 0.2165
            if col % 2 == 1:
                y += 0.10825
            if -0.45 <= x <= 0.45 and -0.45 <= y <= 0.45:
                centres.append([x, y])
    
    if len(centres) < N:
        while len(centres) < N:
            centres.append([random.uniform(-0.45, 0.45), random.uniform(-0.45, 0.45)])
    
    return np.array(centres[:N])


def _anneal(centres, max_time=100.0, seed=42):
    """Simulated annealing optimization with adaptive step size."""
    np.random.seed(seed)
    random.seed(seed)
    
    centres = centres.copy()
    current_score = _score(centres)
    best_centres = centres.copy()
    best_score = current_score
    
    start_time = time.time()
    deadline = start_time + max_time
    
    T = 0.01
    T_min = 1e-8
    cooling = 0.9995
    
    n = len(centres)
    step_scale = 0.05
    
    while time.time() < deadline:
        i = np.random.randint(n)
        
        dx = np.random.normal(0, step_scale)
        dy = np.random.normal(0, step_scale)
        
        new_centre = centres[i].copy()
        new_centre[0] += dx
        new_centre[1] += dy
        
        new_centre[0] = np.clip(new_centre[0], -0.499, 0.499)
        new_centre[1] = np.clip(new_centre[1], -0.499, 0.499)
        
        old_centre = centres[i].copy()
        centres[i] = new_centre
        
        new_score = _score(centres)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centres = centres.copy()
            step_scale = max(step_scale * 1.02, 0.001)
        else:
            centres[i] = old_centre
            step_scale = max(step_scale * 0.98, 0.001)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_centres, best_score


def _polish_slsqp(centres, max_time=30.0):
    """Polish using SLSQP to maximize minimum radius directly."""
    start_time = time.time()
    deadline = start_time + max_time
    
    centres = centres.copy()
    current_score = _score(centres)
    best_centres = centres.copy()
    best_score = current_score
    
    def neg_min_radius(x):
        c = x.reshape(N, 2)
        # Wall constraint
        wall = 0.5 - np.abs(c)
        min_wall = np.min(wall)
        # Pair constraint
        dist = cdist(c, c)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        return -min(min_wall, min_pair)
    
    bounds = [(-0.499, 0.499)] * (2 * N)
    
    # Run SLSQP with time limit
    x0 = centres.flatten()
    
    # Use multiple restarts if time permits
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        try:
            result = minimize(
                neg_min_radius,
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            x_opt = result.x.reshape(N, 2)
            # Ensure strictly inside
            x_opt = np.clip(x_opt, -0.4999, 0.4999)
            score = _score(x_opt)
            
            if score > best_score:
                best_score = score
                best_centres = x_opt.copy()
                x0 = x_opt.flatten()
        except Exception:
            break
        
        if time.time() > deadline - 10.0:
            break
        
        # Perturb for next attempt
        if attempts < 3:
            x0 = x0 + np.random.normal(0, 0.01, size=2*N)
            x0 = np.clip(x0, -0.499, 0.499)
    
    return best_centres, best_score


def _polish_nm(centres, max_time=30.0):
    """Polish using Nelder-Mead for non-smooth objective."""
    start_time = time.time()
    deadline = start_time + max_time
    
    centres = centres.copy()
    current_score = _score(centres)
    best_centres = centres.copy()
    best_score = current_score
    
    def neg_min_radius(x):
        c = x.reshape(N, 2)
        wall = 0.5 - np.abs(c)
        min_wall = np.min(wall)
        dist = cdist(c, c)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        return -min(min_wall, min_pair)
    
    x0 = centres.flatten()
    
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        try:
            result = minimize(
                neg_min_radius,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            x_opt = result.x.reshape(N, 2)
            x_opt = np.clip(x_opt, -0.4999, 0.4999)
            score = _score(x_opt)
            
            if score > best_score:
                best_score = score
                best_centres = x_opt.copy()
                x0 = x_opt.flatten()
        except Exception:
            break
        
        if time.time() > deadline - 10.0:
            break
        
        if attempts < 3:
            x0 = x0 + np.random.normal(0, 0.005, size=2*N)
            x0 = np.clip(x0, -0.499, 0.499)
    
    return best_centres, best_score


def construct():
    """Construct the optimal packing of 50 equal circles in a unit square."""
    start_time = time.time()
    total_deadline = start_time + 145.0
    
    np.random.seed(42)
    random.seed(42)
    
    initial_configs = [
        _construct_grid(),
        _construct_honeycomb(),
        np.random.uniform(-0.45, 0.45, size=(N, 2)),
        np.random.uniform(-0.45, 0.45, size=(N, 2)),
    ]
    
    best_centres = None
    best_score = -np.inf
    
    # Phase 1: Annealing from different starts
    anneal_configs = initial_configs[:3]
    for config in anneal_configs:
        if time.time() > total_deadline - 60.0:
            break
        
        remaining = total_deadline - time.time() - 60.0
        anneal_time = min(25.0, remaining)
        if anneal_time <= 0:
            break
        
        centres, score = _anneal(config, max_time=anneal_time, seed=np.random.randint(100000))
        
        if score > best_score:
            best_score = score
            best_centres = centres
    
    if best_centres is None:
        best_centres = initial_configs[0]
        best_score = _score(best_centres)
    
    # Phase 2: SLSQP polish
    polish_time = min(40.0, total_deadline - time.time() - 20.0)
    if polish_time > 5.0:
        polished_centres, polished_score = _polish_slsqp(best_centres, max_time=polish_time)
        if polished_score > best_score:
            best_score = polished_score
            best_centres = polished_centres
    
    # Phase 3: Nelder-Mead polish
    polish_time = min(30.0, total_deadline - time.time() - 10.0)
    if polish_time > 5.0:
        polished_centres, polished_score = _polish_nm(best_centres, max_time=polish_time)
        if polished_score > best_score:
            best_score = polished_score
            best_centres = polished_centres
    
    # Final safety clip
    best_centres = np.clip(best_centres, -0.4999, 0.4999)
    
    return best_centres

# EVOLVE-BLOCK-END
