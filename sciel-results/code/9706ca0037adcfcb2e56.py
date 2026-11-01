"""Pack 50 equal circles in a unit square; maximize common radius."""
import numpy as np
import time

N = 50
SIDE = 1.0
HALF = 0.5


def _radius(pts):
    """Compute the common radius for a set of centres."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # distance to boundary
    dist_boundary = np.min(np.abs(pts), axis=1)
    min_boundary = HALF - np.max(dist_boundary)
    if min_boundary < -1e-12:
        return 0.0
    # pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    r = min(min_boundary, min_pair / 2.0)
    if r < 0:
        return 0.0
    return r


def _soft_radius(pts, beta=200.0):
    """Smooth surrogate of the radius using soft-min."""
    if pts.shape[0] != N:
        return 0.0
    # boundary term
    dist_boundary = np.min(np.abs(pts), axis=1)
    boundary_term = HALF - np.max(dist_boundary)
    
    # pairwise soft-min
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # soft min of pairwise distances / 2
    d_half = d / 2.0
    # log-sum-exp for soft min: -log(sum(exp(-beta*x))) / beta
    # for numerical stability
    d_min = np.min(d_half)
    log_sum = np.log(np.sum(np.exp(-beta * (d_half - d_min))))
    soft_pair = d_min - np.log(np.sum(np.exp(-beta * (d_half - d_min)))) / beta
    
    # soft min of boundary and pair
    a = boundary_term
    b = soft_pair
    m = np.minimum(a, b)
    log_sum2 = np.log(np.exp(beta * (a - m)) + np.exp(beta * (b - m)))
    soft_min = m - np.log(log_sum2) / beta
    
    return max(soft_min, 0.0)


def _hex_init(seed=42):
    """Initialize with a hexagonal-like packing."""
    rng = np.random.RandomState(seed)
    # Start with a grid and add small jitter
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    # Add small random jitter
    pts += rng.uniform(-0.01, 0.01, size=pts.shape)
    # Clip to boundary
    pts = np.clip(pts, -HALF + 0.001, HALF - 0.001)
    return pts


def _simulated_annealing(pts, time_budget=140.0, start_time=None):
    """Simulated annealing to maximize radius."""
    if start_time is None:
        start_time = time.time()
    
    current = pts.copy()
    current_r = _radius(current)
    best = current.copy()
    best_r = current_r
    
    # Temperature schedule
    T_start = 0.02
    T_end = 0.0001
    # We'll do iterations until time runs out
    
    rng = np.random.RandomState(12345)
    
    iter_count = 0
    no_improve_count = 0
    patience = 50000
    
    while time.time() - start_time < time_budget:
        # Adaptive temperature
        progress = min(1.0, iter_count / 500000.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random point to move
        idx = rng.randint(N)
        
        # Step size scales with temperature
        step_size = T * 0.5
        delta = rng.normal(0, step_size, size=2)
        
        new_pts = current.copy()
        new_pts[idx] += delta
        
        # Keep within bounds
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_r = _radius(new_pts)
        
        # Accept if better, or with probability exp((new_r - current_r) / T)
        if new_r >= current_r:
            current = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            accept_prob = np.exp((new_r - current_r) / T)
            if rng.random() < accept_prob:
                current = new_pts
                current_r = new_r
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        iter_count += 1
        
        # Restart if stuck
        if no_improve_count > patience:
            # Perturb the best solution
            current = best.copy()
            current += rng.uniform(-0.05, 0.05, size=current.shape)
            current = np.clip(current, -HALF, HALF)
            current_r = _radius(current)
            no_improve_count = 0
    
    return best, best_r


def _local_optimization(pts, time_budget=20.0, start_time=None):
    """Local optimization using gradient-free search (Nelder-Mead style)."""
    if start_time is None:
        start_time = time.time()
    
    current = pts.copy()
    current_r = _radius(current)
    best = current.copy()
    best_r = current_r
    
    # Simple coordinate descent with adaptive step
    rng = np.random.RandomState(999)
    step = 0.01
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try small moves
                for sign in [1, -1]:
                    new_pts = current.copy()
                    new_pts[i, coord] += sign * step
                    new_pts = np.clip(new_pts, -HALF, HALF)
                    new_r = _radius(new_pts)
                    if new_r > current_r:
                        current = new_pts
                        current_r = new_r
                        improved = True
                        if current_r > best_r:
                            best_r = current_r
                            best = current.copy()
        
        if not improved:
            # Reduce step size
            step *= 0.8
            if step < 1e-8:
                break
        else:
            # Reset step size slightly
            step = max(step * 1.1, 0.005)
    
    return best, best_r


def construct():
    start_time = time.time()
    
    # Multiple restarts with different initializations
    best_pts = None
    best_r = 0.0
    
    # Try several different seeds for initialization
    for seed in range(10):
        remaining = 150.0 - (time.time() - start_time)
        if remaining < 15.0:
            break
        
        # Initialize
        if seed == 0:
            # Grid init
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            pts = grid[:N].copy()
        elif seed == 1:
            # Hex-like init
            pts = _hex_init(seed=seed + 100)
        else:
            # Random init with some structure
            rng = np.random.RandomState(seed + 200)
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            pts = grid[:N].copy()
            pts += rng.uniform(-0.02, 0.02, size=pts.shape)
            pts = np.clip(pts, -HALF + 0.001, HALF - 0.001)
        
        # SA phase
        sa_time = min(10.0, remaining * 0.6)
        sa_pts, sa_r = _simulated_annealing(pts, time_budget=sa_time, start_time=start_time)
        
        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts.copy()
        
        # Local optimization phase
        remaining = 150.0 - (time.time() - start_time)
        if remaining < 5.0:
            break
        local_time = min(5.0, remaining * 0.3)
        loc_pts, loc_r = _local_optimization(sa_pts, time_budget=local_time, start_time=start_time)
        
        if loc_r > best_r:
            best_r = loc_r
            best_pts = loc_pts.copy()
    
    # Final check
    if best_pts is None:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:N]
    
    # Ensure valid output
    best_pts = np.clip(best_pts, -HALF, HALF)
    
    return best_pts
