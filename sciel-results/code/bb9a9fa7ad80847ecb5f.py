"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 200
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds before deadline to stop


def _score(c):
    """Compute the packing score (largest common radius)."""
    # Check validity
    if not np.all(np.isfinite(c)):
        return 0.0
    if c.shape != (N, 2):
        return 0.0
    if np.any(c < -0.5 - 1e-12) or np.any(c > 0.5 + 1e-12):
        return 0.0
    # Pairwise distances
    d = pdist(c)
    if np.any(d < 1e-15):
        return 0.0
    min_pair = d.min() / 2.0
    # Distance to walls
    wall_dists = np.min(np.stack([
        0.5 - np.abs(c[:, 0]),
        0.5 - np.abs(c[:, 1])
    ], axis=1), axis=1)
    min_wall = wall_dists.min()
    return min(min_pair, min_wall)


def _init_hexagonal():
    """Generate a hexagonal packing initial configuration."""
    # For 200 circles, use a hexagonal lattice pattern
    # Try different row counts to fit 200 circles
    best_c = None
    best_score = 0.0
    
    for rows in range(12, 20):
        # Calculate columns needed
        cols = int(np.ceil(N / rows))
        if rows * cols < N:
            cols += 1
        
        # Hexagonal spacing
        # For a unit square [-0.5, 0.5]^2
        # x spacing = 1.0 / (cols - 0.5) approximately for hex
        # Let's use a simpler approach: distribute in a grid with hex offset
        
        # Base spacing
        x_span = 1.0
        y_span = 1.0
        
        # For hex packing, horizontal spacing is 2*r, vertical is sqrt(3)*r
        # We want to fit cols in x and rows in y
        # x positions: -0.5 + (i + 0.5) * x_step
        # y positions: -0.5 + (j + 0.5) * y_step
        # With hex offset: even rows shifted by x_step/2
        
        x_step = x_span / cols
        y_step = y_span / rows
        
        # Hexagonal: vertical spacing should be x_step * sqrt(3)/2 for optimal
        # But we're constrained by the square, so let's just do a hex-like pattern
        
        c = np.zeros((N, 2))
        idx = 0
        for j in range(rows):
            row_len = cols
            if idx + row_len > N:
                row_len = N - idx
            for i in range(row_len):
                # Hex offset: shift odd rows
                x_offset = 0.0
                if j % 2 == 1:
                    x_offset = x_step / 2.0
                c[idx, 0] = -0.5 + (i + 0.5) * x_step + x_offset
                c[idx, 1] = -0.5 + (j + 0.5) * y_step
                idx += 1
            if idx >= N:
                break
        
        if idx < N:
            # Fill remaining with random positions in the square
            for k in range(idx, N):
                c[k, 0] = np.random.uniform(-0.4, 0.4)
                c[k, 1] = np.random.uniform(-0.4, 0.4)
        
        s = _score(c)
        if s > best_score:
            best_score = s
            best_c = c.copy()
    
    return best_c


def _init_grid():
    """Generate a square grid initial configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _simulated_annealing(c, time_limit=30.0):
    """Run simulated annealing to improve the packing."""
    start_time = time.time()
    c = c.copy()
    best_c = c.copy()
    best_score = _score(c)
    current_score = best_score
    
    # Initial temperature
    T = 0.001
    T_min = 0.00001
    cooling_rate = 0.9995
    
    # Number of circles to move per iteration
    n_movable = 5
    
    while time.time() - start_time < time_limit:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Pick random circles to move
        idxs = np.random.choice(N, size=min(n_movable, N), replace=False)
        
        # Generate random perturbations
        step_size = T * 10
        perturbations = np.random.normal(0, step_size, size=(len(idxs), 2))
        
        # Clamp to stay within bounds
        new_c = c.copy()
        new_c[idxs] += perturbations
        new_c = np.clip(new_c, -0.49, 0.49)
        
        new_score = _score(new_c)
        
        # Accept or reject
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / T):
            c = new_c
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best_c


def _gradient_ascent(c, time_limit=30.0):
    """Run gradient ascent on the smooth surrogate objective."""
    start_time = time.time()
    c = c.copy().flatten()
    n = len(c)
    
    # Smooth surrogate: use soft-min via log-sum-exp
    beta = 100.0  # Larger beta = sharper min
    
    def objective(x):
        x = x.reshape(-1, 2)
        # Pairwise distances
        d = pdist(x)
        # Wall distances
        wall = np.min(np.stack([
            0.5 - np.abs(x[:, 0]),
            0.5 - np.abs(x[:, 1])
        ], axis=1), axis=1)
        
        # Combine all distances
        all_d = np.concatenate([d, wall])
        all_d = np.clip(all_d, 1e-10, None)
        
        # Soft min: -log(sum(exp(-beta*d))) / beta
        min_val = -np.log(np.sum(np.exp(-beta * all_d))) / beta
        return -min_val  # Negate for minimization
    
    def gradient(x):
        x = x.reshape(-1, 2)
        d = pdist(x)
        wall = np.min(np.stack([
            0.5 - np.abs(x[:, 0]),
            0.5 - np.abs(x[:, 1])
        ], axis=1), axis=1)
        
        all_d = np.concatenate([d, wall])
        all_d = np.clip(all_d, 1e-10, None)
        
        # Soft min gradient
        weights = np.exp(-beta * all_d)
        weights /= np.sum(weights)
        
        # Gradient of soft min w.r.t. each distance is -weights
        # Then chain rule to get gradient w.r.t. positions
        
        grad = np.zeros_like(x)
        
        # Pairwise part
        idx = np.triu_indices(N, k=1)
        for k in range(len(idx[0])):
            i, j = idx[0][k], idx[1][k]
            diff = x[i] - x[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                grad[i] -= weights[k] * diff / dist
                grad[j] += weights[k] * diff / dist
        
        # Wall part
        for i in range(N):
            dx = 0.5 - np.abs(x[i, 0])
            dy = 0.5 - np.abs(x[i, 1])
            if dx < dy:
                # x wall
                if x[i, 0] > 0:
                    grad[i, 0] += weights[N*(N-1)//2 + i]
                else:
                    grad[i, 0] -= weights[N*(N-1)//2 + i]
            else:
                # y wall
                if x[i, 1] > 0:
                    grad[i, 1] += weights[N*(N-1)//2 + i]
                else:
                    grad[i, 1] -= weights[N*(N-1)//2 + i]
        
        return -grad.flatten()  # Negate because we're minimizing -softmin
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.49, 0.49)] * n
    
    result = minimize(objective, c, method='L-BFGS-B', jac=gradient,
                      bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
    
    if result.success and _score(result.x.reshape(-1, 2)) > _score(c.reshape(-1, 2)):
        return result.x.reshape(-1, 2)
    return c.reshape(-1, 2)


def _polish(c, time_limit=20.0):
    """Final high-precision polish pass."""
    start_time = time.time()
    c = c.copy()
    
    # Multiple rounds of very tight optimization
    for _ in range(3):
        if time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        c = _gradient_ascent(c, time_limit=5.0)
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
    
    return c


def construct():
    """Construct the optimal packing of 200 equal circles in a unit square."""
    start_time = time.time()
    
    # Try multiple initializations and pick the best
    inits = []
    
    # Grid initialization
    inits.append(_init_grid())
    
    # Hexagonal initialization
    inits.append(_init_hexagonal())
    
    # Random initialization with some structure
    np.random.seed(42)
    c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
    inits.append(c_rand)
    
    # Pick the best initial configuration
    best_init = None
    best_init_score = 0.0
    for init in inits:
        s = _score(init)
        if s > best_init_score:
            best_init_score = s
            best_init = init.copy()
    
    c = best_init
    best_c = c.copy()
    best_score = best_init_score
    
    # Phase 1: Simulated annealing
    if time.time() - start_time < DEADLINE - 60:
        c = _simulated_annealing(c, time_limit=min(40.0, DEADLINE - 60 - (time.time() - start_time)))
        if _score(c) > best_score:
            best_score = _score(c)
            best_c = c.copy()
    
    # Phase 2: Gradient ascent
    if time.time() - start_time < DEADLINE - 30:
        remaining = min(30.0, DEADLINE - 30 - (time.time() - start_time))
        if remaining > 5:
            c = _gradient_ascent(c, time_limit=remaining)
            if _score(c) > best_score:
                best_score = _score(c)
                best_c = c.copy()
    
    # Phase 3: Polish
    if time.time() - start_time < DEADLINE - 15:
        remaining = min(15.0, DEADLINE - 15 - (time.time() - start_time))
        if remaining > 3:
            c = _polish(c, time_limit=remaining)
            if _score(c) > best_score:
                best_score = _score(c)
                best_c = c.copy()
    
    # Final safety check
    if _score(best_c) <= 0:
        return _init_grid()
    
    return best_c
# EVOLVE-BLOCK-END
