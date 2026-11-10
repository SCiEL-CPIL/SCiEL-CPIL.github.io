import numpy as np
import time

N = 50
CUBE_HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -CUBE_HALF) or np.any(centers > CUBE_HALF):
        return 0.0
    # Min distance to walls
    wall_dists = np.min(np.abs(centers), axis=1)
    wall_min = np.min(wall_dists)
    # Min pairwise distance / 2
    if N > 1:
        from scipy.spatial.distance import pdist
        pair_dists = pdist(centers)
        pair_min = np.min(pair_dists) / 2.0
    else:
        pair_min = np.inf
    return min(wall_min, pair_min)

def _repulsion_energy(centers):
    """
    Energy function: we want to maximize the minimum distance.
    We minimize a potential that penalizes close pairs and wall proximity.
    Using a smooth approximation: sum of 1/d^2 for pairs + wall penalties.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(N, 3)
    
    # Wall penalty: quadratic penalty for being too close to walls
    # We want centers to stay at least some distance from walls
    # Use a soft boundary at 0.5 - epsilon
    eps = 0.01
    wall_viol = np.maximum(0.0, np.abs(centers) - (CUBE_HALF - eps))
    wall_energy = np.sum(wall_viol ** 2) * 1000.0
    
    # Pairwise repulsion: sum of 1/d^2
    from scipy.spatial.distance import pdist
    d = pdist(centers)
    # Avoid division by zero
    d_safe = np.maximum(d, 1e-8)
    pair_energy = np.sum(1.0 / (d_safe ** 2))
    
    return pair_energy + wall_energy

def _simulated_annealing(centers, max_iters=50000, initial_temp=0.05, 
                         cooling_rate=0.9999, max_time=60.0):
    """
    Simulated annealing to maximize minimum distance.
    """
    t_start = time.time()
    centers = np.array(centers, dtype=np.float64).copy()
    current_energy = _repulsion_energy(centers)
    best_centers = centers.copy()
    best_energy = current_energy
    best_radius = _compute_radius(centers)
    
    n_params = N * 3
    temp = initial_temp
    
    for it in range(max_iters):
        if time.time() - t_start > max_time:
            break
        
        # Generate perturbation
        # Random perturbation in a small neighborhood
        perturbation = np.random.normal(0, temp * 0.1, size=n_params)
        new_centers = centers + perturbation
        
        # Clip to cube
        new_centers = np.clip(new_centers, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        new_energy = _repulsion_energy(new_centers)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = new_energy - current_energy
        if delta < 0 or np.random.random() < np.exp(-delta / max(temp, 1e-10)):
            centers = new_centers
            current_energy = new_energy
            
            # Update best
            if current_energy < best_energy:
                best_energy = current_energy
                best_centers = centers.copy()
                best_radius = _compute_radius(centers)
        
        # Cool down
        temp *= cooling_rate
        if temp < 1e-8:
            temp = 1e-8
    
    return best_centers, best_radius

def _local_optimization(centers, max_time=30.0):
    """
    Local optimization using gradient descent on the repulsion energy.
    """
    t_start = time.time()
    centers = np.array(centers, dtype=np.float64).copy()
    
    # Simple gradient descent with adaptive step size
    step_size = 0.001
    lr = 0.01
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    for it in range(10000):
        if time.time() - t_start > max_time:
            break
        
        # Compute gradient numerically (central difference)
        grad = np.zeros_like(centers)
        h = 1e-5
        for i in range(N):
            for j in range(3):
                orig = centers[i, j]
                centers[i, j] = orig + h
                f_plus = _repulsion_energy(centers)
                centers[i, j] = orig - h
                f_minus = _repulsion_energy(centers)
                centers[i, j] = orig
                grad[i, j] = (f_plus - f_minus) / (2 * h)
        
        # Gradient descent step
        centers -= lr * grad
        centers = np.clip(centers, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)
        
        # Update best
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.95
            if lr < 1e-6:
                break
    
    return best_centers, best_radius

def _grid_initialization():
    """Create a structured grid initial configuration."""
    # Use a 4x4x4 grid = 64 points, take 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_initialization(seed=None):
    """Create a random initial configuration."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _bcc_initialization():
    """Create a BCC-like initial configuration."""
    # BCC lattice: corners + body centers of sub-cubes
    # Divide cube into 3x3x3 sub-cubes, place at corners and centers
    m = 3
    sub_size = 1.0 / m
    centers = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x0 = -0.5 + i * sub_size
                y0 = -0.5 + j * sub_size
                z0 = -0.5 + k * sub_size
                # Add corner (one corner of each sub-cube)
                centers.append([x0, y0, z0])
                # Add body center
                centers.append([x0 + sub_size/2, y0 + sub_size/2, z0 + sub_size/2])
    
    centers = np.array(centers)
    # We have 27*2 = 54 points, take 50
    return centers[:N].copy()

def construct():
    t_start = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_centers = None
    best_score = 0.0
    
    # Try multiple initializations
    initializations = [
        _grid_initialization(),
        _bcc_initialization(),
        _random_initialization(seed=42),
        _random_initialization(seed=123),
        _random_initialization(seed=456),
    ]
    
    for idx, init in enumerate(initializations):
        elapsed = time.time() - t_start
        if elapsed > deadline - 20:
            break
        
        # Phase 1: Simulated annealing
        sa_time = min(30.0, deadline - elapsed - 15)
        if sa_time < 5:
            break
        
        sa_centers, sa_score = _simulated_annealing(init, max_time=sa_time)
        elapsed = time.time() - t_start
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers.copy()
        
        # Phase 2: Local optimization
        elapsed = time.time() - t_start
        if elapsed > deadline - 10:
            break
        
        lo_time = min(20.0, deadline - elapsed - 5)
        if lo_time < 3:
            break
        
        lo_centers, lo_score = _local_optimization(sa_centers, max_time=lo_time)
        elapsed = time.time() - t_start
        if lo_score > best_score:
            best_score = lo_score
            best_centers = lo_centers.copy()
    
    # Final validation
    if best_centers is None:
        best_centers = _grid_initialization()
    
    best_centers = np.asarray(best_centers, dtype=np.float64)
    if best_centers.shape != (N, 3) or not np.all(np.isfinite(best_centers)):
        best_centers = _grid_initialization()
    
    # Clip to cube
    best_centers = np.clip(best_centers, -CUBE_HALF, CUBE_HALF)
    
    return best_centers
