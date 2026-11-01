"""Improved packing: smooth surrogate objective with soft-min for nonsmooth constraints."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _score(centers):
    """Compute the maximum common radius for given centers (exact, for evaluation)."""
    centers = np.asarray(centers)
    wall_min = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_min)
    n = len(centers)
    if n < 2:
        return r_wall
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _soft_min(values, beta=50.0):
    """Smooth approximation to min via log-sum-exp (negative log-sum-exp)."""
    values = np.asarray(values)
    m = np.min(values)
    # Numerically stable: m - (1/beta)*log(sum(exp(-beta*(v-m))))
    exponent = -beta * (values - m)
    exponent = np.clip(exponent, -700, 700)
    log_sum = np.log(np.sum(np.exp(exponent)))
    return m - log_sum / beta

def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: minimize -soft_min(all constraints).
    Constraints: wall distances (0.5 - |c_ik|) and half pairwise distances.
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: 0.5 - |c_ik| for each coordinate
    wall_constraints = 0.5 - np.abs(c).ravel()
    
    # Pairwise half-distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constraints = dists / 2.0
    pair_constraints = pair_constraints[np.triu_indices_from(pair_constraints, k=1)]
    
    # Combine all constraints
    all_constraints = np.concatenate([wall_constraints, pair_constraints])
    
    # Soft-min of all constraints (smooth approximation to the true min)
    s = _soft_min(all_constraints, beta)
    
    # Penalty for going outside bounds (soft)
    out_of_bounds = np.maximum(0, np.abs(c) - 0.4999)
    penalty = 10.0 * np.sum(out_of_bounds ** 2)
    
    return -s + penalty

def _smooth_gradient(x, beta=50.0):
    """Numerical gradient of the smooth objective."""
    eps = 1e-7
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (_smooth_objective(xp, beta) - _smooth_objective(xm, beta)) / (2 * eps)
    return grad

def _initial_grid():
    """FCC-inspired grid initialization."""
    # Use a 4x4x4 grid = 64 positions, pick 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle deterministically and pick first N
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(grid))
    return grid[perm[:N]].copy()

def _initial_fcc_like():
    """Initialize with FCC-like structure: layers of hexagonal packing."""
    centers = np.zeros((N, 3))
    # Create layers in z-direction
    n_layers = 5
    per_layer = N // n_layers  # 10 per layer
    remaining = N - per_layer * n_layers
    
    for layer in range(n_layers):
        z = (layer + 0.5) / n_layers - 0.5
        n_in_layer = per_layer + (1 if layer < remaining else 0)
        
        # Hexagonal arrangement within layer
        for i in range(n_in_layer):
            row = i // 4
            col = i % 4
            x = (col + 0.5) / 4.0 - 0.5
            y = (row + 0.5) / 3.0 - 0.5
            if row % 2 == 1:
                x += 0.25 / 4.0  # offset alternate rows
            centers[layer * per_layer + i] = [x, y, z]
    
    # Fill remaining
    idx = per_layer * n_layers
    while idx < N:
        centers[idx] = [0, 0, 0]
        idx += 1
    
    return centers

def construct():
    t_start = time.time()
    deadline = t_start + 130  # Safety margin
    
    # Try multiple initializations
    inits = [
        ("grid", _initial_grid()),
        ("fcc", _initial_fcc_like()),
    ]
    
    best_centers = None
    best_score = -np.inf
    
    for name, init in inits:
        if time.time() > deadline - 20:
            break
        
        cur = init.copy()
        cur_score = _score(cur)
        
        if best_centers is None or cur_score > best_score:
            best_centers = cur.copy()
            best_score = cur_score
        
        # Optimize from this initialization with smooth objective
        # Start with coarse beta, then refine
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    _smooth_objective,
                    cur.ravel(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * 3 * N,
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_score = _score(new_centers)
                
                if new_score > best_score:
                    best_centers = new_centers.copy()
                    best_score = new_score
                    cur = new_centers.copy()
            except Exception:
                pass
        
        # Final polish with exact objective via coordinate descent
        if time.time() > deadline - 15:
            break
        
        cur = best_centers.copy()
        improved = True
        step = 0.002
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                for coord in range(3):
                    for delta in [step, -step, step*0.5, -step*0.5, step*2, -step*2]:
                        new_c = cur.copy()
                        new_c[i, coord] += delta
                        if np.all(np.abs(new_c) <= 0.5):
                            s = _score(new_c)
                            if s > best_score + 1e-12:
                                cur = new_c
                                best_centers = new_c.copy()
                                best_score = s
                                improved = True
            # Reduce step size if no improvement
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
    
    return best_centers
