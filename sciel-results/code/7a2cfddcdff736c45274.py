"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_MARGIN = 5.0  # seconds before the 150s limit

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside cube
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Minimum distance to boundary
    dist_boundary = np.min(0.5 - np.abs(centers), axis=1)
    min_boundary = np.min(dist_boundary)
    # Minimum pairwise distance
    n = len(centers)
    min_pair = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    if min_pair == 0.0:
        return 0.0
    return min(min_pair / 2.0, min_boundary)

def _construct_cubic_grid():
    """Simple cubic grid as baseline."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _construct_fibonacci():
    """Fibonacci sphere distribution scaled to fit in cube."""
    n = N
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
    z = np.linspace(1, -1, n)
    z *= 0.9  # Shrink slightly to avoid corners initially
    r = np.sqrt(1.0 - z**2)
    theta = phi * np.arange(n)
    
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    pts = np.column_stack((x, y, z))
    
    # Scale to fit inside [-0.5, 0.5]^3
    scale = 0.4
    pts = pts * scale
    return pts

def _random_initialization():
    """Random initialization inside the cube with some margin."""
    margin = 0.1
    centers = np.random.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))
    return centers

def _optimize_from_start(centers_start, time_limit=30.0):
    """Run local optimization from a starting point."""
    start_time = time.time()
    params = centers_start.flatten()
    
    # Use L-BFGS-B for constrained optimization
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_params = params.copy()
    best_r = _compute_radius(centers_start)
    
    # Try a few iterations with different step sizes
    try:
        result = minimize(
            lambda p: -_compute_radius(p.reshape(N, 3)), 
            params, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or -result.fun > best_r:
            new_centers = result.x.reshape(N, 3)
            new_r = _compute_radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_params = result.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    return best_params.reshape(N, 3), best_r, elapsed

def _simulated_annealing(centers_start, time_limit=20.0):
    """Simple simulated annealing to escape local minima."""
    start_time = time.time()
    centers = centers_start.copy()
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_r = current_r
    
    T = 0.05  # Initial temperature (larger for better exploration)
    cooling = 0.999
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        if T < 1e-6:
            break
        
        # Random perturbation
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, size=3)
        new_centers = centers.copy()
        new_centers[idx] += delta
        new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
        
        new_r = _compute_radius(new_centers)
        delta_r = new_r - current_r
        
        if delta_r > 0 or np.random.random() < np.exp(delta_r / T):
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        
        T *= cooling
        # Adaptive step size
        step_size = 0.05 * (T / 0.05)
        if step_size < 1e-4:
            step_size = 1e-4
    
    return best_centers, best_r, time.time() - start_time

def _local_refine(centers_start, time_limit=10.0):
    """Coordinate descent local refinement."""
    start_time = time.time()
    pos = centers_start.copy()
    score = _compute_radius(pos)
    
    for _ in range(200):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            for axis in range(3):
                for direction in [1, -1]:
                    for step in [0.01, 0.005, 0.001]:
                        trial = pos.copy()
                        trial[i, axis] += direction * step
                        trial = np.clip(trial, -0.5, 0.5)
                        trial_score = _compute_radius(trial)
                        if trial_score > score:
                            pos = trial
                            score = trial_score
                            improved = True
        if not improved:
            break
    
    return pos, score

def construct():
    """Construct the best packing found within the time budget."""
    global_start = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    best_centers = None
    best_r = 0.0
    
    # Start with cubic grid
    centers0 = _construct_cubic_grid()
    r0 = _compute_radius(centers0)
    best_centers = centers0.copy()
    best_r = r0
    
    # Try Fibonacci initialization
    if time.time() - global_start < deadline:
        centers_fib = _construct_fibonacci()
        r_fib = _compute_radius(centers_fib)
        if r_fib > best_r:
            best_r = r_fib
            best_centers = centers_fib.copy()
    
    # Try optimization from grid
    if time.time() - global_start < deadline:
        opt_centers, opt_r, elapsed = _optimize_from_start(centers0, time_limit=min(25.0, deadline - (time.time() - global_start)))
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()
    
    # Try optimization from Fibonacci
    if time.time() - global_start < deadline:
        opt_centers, opt_r, elapsed = _optimize_from_start(centers_fib, time_limit=min(25.0, deadline - (time.time() - global_start)))
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()
    
    # Try random restarts with optimization
    n_random_tries = 3
    for i in range(n_random_tries):
        if time.time() - global_start > deadline:
            break
        
        centers_rand = _random_initialization()
        r_rand = _compute_radius(centers_rand)
        
        # Quick optimization
        time_remaining = deadline - (time.time() - global_start)
        if time_remaining < 10.0:
            break
        
        opt_centers, opt_r, elapsed = _optimize_from_start(centers_rand, time_limit=min(12.0, time_remaining))
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()
    
    # Try simulated annealing on the best found
    if time.time() - global_start < deadline - 15.0:
        sa_centers, sa_r, elapsed = _simulated_annealing(best_centers, time_limit=min(20.0, deadline - (time.time() - global_start) - 5.0))
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()
    
    # Local refinement
    if time.time() - global_start < deadline - 5.0:
        time_remaining = deadline - (time.time() - global_start)
        refine_centers, refine_r = _local_refine(best_centers, time_limit=min(10.0, time_remaining))
        if refine_r > best_r:
            best_r = refine_r
            best_centers = refine_centers.copy()
    
    # Final optimization pass on the best
    if time.time() - global_start < deadline - 3.0:
        time_remaining = deadline - (time.time() - global_start)
        opt_centers, opt_r, elapsed = _optimize_from_start(best_centers, time_limit=min(8.0, time_remaining))
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()
    
    # Ensure validity
    final_centers = np.clip(best_centers, -0.5, 0.5)
    final_r = _compute_radius(final_centers)
    
    if final_r <= 0.0:
        # Fallback to cubic grid
        final_centers = _construct_cubic_grid()
    
    return final_centers
# EVOLVE-BLOCK-END
