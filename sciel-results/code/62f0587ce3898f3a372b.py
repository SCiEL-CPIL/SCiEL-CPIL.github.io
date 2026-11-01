import numpy as np
from scipy.optimize import minimize
import time as _time

_DEADLINE = _time.time() + 140.0  # 150s limit, keep 10s margin
_N = 100

def _compute_radius(centers):
    """Compute the exact largest common radius for given centers inside [-0.5, 0.5]^3."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (_N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    half = 0.5
    wall_dists = np.min(half - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq)
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_objective(x, beta):
    """
    Minimize -soft_min of radius-limit distances.
    Radius limits: d_ij/2 for all i<j, and w_i = min_k(0.5 - |c_ik|) for each i.
    """
    centers = x.reshape(_N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    half = 0.5
    wall_dists = np.min(half - np.abs(centers), axis=1)
    
    # Combine all distance candidates
    iu = np.triu_indices(_N, k=1)
    pair_dists = dists[iu] / 2.0
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Stable soft-min
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = min_val - (1.0 / beta) * log_sum
    
    return -soft_min_val

def _build_initial_grid(jitter=0.0, seed=42):
    """Build a dense grid and pick N points with good separation."""
    rng = np.random.RandomState(seed)
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    if jitter > 0:
        noise = rng.randn(*grid.shape) * jitter
        grid = grid + noise
        np.clip(grid, -0.49, 0.49, out=grid)
        
    return grid[:_N]

def _build_bcc_init(seed=42):
    """Build a BCC-like initial configuration."""
    rng = np.random.RandomState(seed)
    # Start with a 5x5x4 grid, then add BCC offset points
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
                # Add BCC center if there's room
                if len(points) < _N:
                    # BCC center is at the center of the cell
                    cx = x + (xs[1] - xs[0]) / 2 if x != xs[-1] else x
                    cy = y + (ys[1] - ys[0]) / 2 if y != ys[-1] else y
                    cz = z + (zs[1] - zs[0]) / 2 if z != zs[-1] else z
                    # Only add if it doesn't go out of bounds
                    if -0.49 <= cx <= 0.49 and -0.49 <= cy <= 0.49 and -0.49 <= cz <= 0.49:
                        points.append([cx, cy, cz])
    
    # Take first N points
    points = np.array(points[:_N])
    
    # Add small jitter
    noise = rng.randn(*points.shape) * 0.005
    points = points + noise
    np.clip(points, -0.49, 0.49, out=points)
    
    return points

def _build_halton_init(seed=42):
    """Build a Halton sequence based initial configuration."""
    rng = np.random.RandomState(seed)
    
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, _N + 1):
        x = halton(i, 2) - 0.5
        y = halton(i, 3) - 0.5
        z = halton(i, 5) - 0.5
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Add small jitter
    noise = rng.randn(*points.shape) * 0.01
    points = points + noise
    np.clip(points, -0.49, 0.49, out=points)
    
    return points

def _anneal(centers, T_init=0.01, T_min=0.0001, cooling=0.995, n_iter=5000, seed=42):
    """Simulated annealing to escape local minima."""
    rng = np.random.RandomState(seed)
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    for i in range(n_iter):
        if _time.time() > _DEADLINE - 15:
            break
            
        T = T_init * (cooling ** i)
        if T < T_min:
            break
            
        # Random perturbation
        idx = rng.randint(_N)
        delta = rng.randn(3) * T
        new = current.copy()
        new[idx] += delta
        np.clip(new, -0.49, 0.49, out=new)
        
        new_r = _compute_radius(new)
        
        # Accept if better or with probability exp((new_r - current_r) / T)
        if new_r >= current_r or rng.rand() < np.exp((new_r - current_r) / max(T, 1e-10)):
            current = new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best = current.copy()
    
    return best, best_r

def _optimize_from_initial(x0, betas=None):
    """Run multi-stage soft-min optimization from initial point."""
    if betas is None:
        betas = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
    
    bounds = [(-0.5, 0.5)] * (_N * 3)
    current_x = x0.copy()
    best_r = _compute_radius(current_x.reshape(_N, 3))
    best_centers = current_x.reshape(_N, 3).copy()
    
    for beta in betas:
        if _time.time() > _DEADLINE - 20:
            break
            
        try:
            res = minimize(
                _soft_min_objective,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            
            if res.success or res.status == 0:
                new_centers = res.x.reshape(_N, 3)
                new_r = _compute_radius(new_centers)
                
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
                    current_x = res.x.copy()
                else:
                    current_x = res.x.copy()
        except Exception:
            pass
    
    return best_centers, best_r

def _polish(centers):
    """High-precision polish using Nelder-Mead on the exact radius objective."""
    def neg_radius(c):
        c = np.reshape(c, (-1, 3))
        return -_compute_radius(c)
    
    try:
        res = minimize(
            neg_radius,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        
        if res.success:
            new_centers = res.x.reshape(_N, 3)
            new_r = _compute_radius(new_centers)
            if new_r > _compute_radius(centers):
                return new_centers
    except Exception:
        pass
    
    return centers

def construct():
    start_time = _time.time()
    
    best_centers = None
    best_r = -np.inf
    
    # Try multiple initializations
    inits = [
        ("grid", _build_initial_grid(jitter=0.0, seed=42)),
        ("jittered_grid", _build_initial_grid(jitter=0.01, seed=42)),
        ("bcc", _build_bcc_init(seed=42)),
        ("halton", _build_halton_init(seed=42)),
    ]
    
    for name, init in inits:
        if _time.time() > _DEADLINE - 30:
            break
            
        r_init = _compute_radius(init)
        if r_init > best_r:
            best_r = r_init
            best_centers = init.copy()
    
    if best_centers is None:
        best_centers = _build_initial_grid(jitter=0.0, seed=42)
        best_r = _compute_radius(best_centers)
    
    # Run optimization from best initial
    opt_centers, opt_r = _optimize_from_initial(best_centers.flatten())
    if opt_r > best_r:
        best_r = opt_r
        best_centers = opt_centers
    
    # Simulated annealing to escape local minima
    if _time.time() < _DEADLINE - 30:
        anneal_centers, anneal_r = _anneal(best_centers, seed=42)
        if anneal_r > best_r:
            best_r = anneal_r
            best_centers = anneal_centers
            
            # Re-optimize after annealing
            opt_centers2, opt_r2 = _optimize_from_initial(best_centers.flatten())
            if opt_r2 > best_r:
                best_r = opt_r2
                best_centers = opt_centers2
    
    # Final high-precision polish
    if _time.time() < _DEADLINE - 10:
        polished = _polish(best_centers)
        polished_r = _compute_radius(polished)
        if polished_r > best_r:
            best_r = polished_r
            best_centers = polished
    
    # Final safety check
    if best_centers is None:
        best_centers = _build_initial_grid(jitter=0.0, seed=42)
    
    return best_centers
