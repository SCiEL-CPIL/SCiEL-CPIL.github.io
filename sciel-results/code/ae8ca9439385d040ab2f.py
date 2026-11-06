"""Solve 100 equal spheres in unit cube: direct construction with optimization."""

import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100
CUBE_HALF = 0.5
DEADLINE = 115  # seconds, safety margin before 120s limit


def compute_radius(centres: np.ndarray) -> float:
    """Compute the common radius r for given centres."""
    # Wall distance
    max_c = np.abs(centres).max(axis=1)
    wall_min = CUBE_HALF - max_c
    r_wall = wall_min.min()
    
    # Pairwise distances
    n = len(centres)
    if n < 2:
        return r_wall
    
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(r_wall, min_pair)


def initial_grid(n: int = N) -> np.ndarray:
    """Create initial grid arrangement."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - CUBE_HALF
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n].astype(float)


def fea_optimize(centres: np.ndarray, target_r: float, 
                  iters: int = 200, lr: float = 0.001) -> tuple:
    """
    Force-exclusion algorithm to push spheres apart while staying in cube.
    Returns (new_centres, min_pair_distance/2, min_wall).
    """
    c = centres.copy().astype(float)
    n = len(c)
    
    # Compute current state
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    wall_min = (CUBE_HALF - np.abs(c).max(axis=1)).min()
    current_r = min(min_pair, wall_min)
    
    if current_r >= target_r:
        return c, min_pair, (CUBE_HALF - np.abs(c).max(axis=1)).min()
    
    # FEA: iteratively push apart overlapping spheres
    for it in range(iters):
        if time.time() > DEADLINE:
            break
            
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        
        # Check if we're already at target
        min_pair = dists.min() / 2.0
        wall_min = (CUBE_HALF - np.abs(c).max(axis=1)).min()
        if min(min_pair, wall_min) >= target_r:
            break
        
        # Build force vectors
        forces = np.zeros_like(c)
        
        # Pairwise repulsion (only for pairs closer than 2*target_r)
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < 2.0 * target_r:
                    diff = c[i] - c[j]
                    d = np.linalg.norm(diff)
                    if d > 1e-10:
                        # Force magnitude: push apart to reach 2*target_r
                        force_mag = (2.0 * target_r - d) / 2.0
                        force_dir = diff / d
                        forces[i] += force_mag * force_dir
                        forces[j] -= force_mag * force_dir
        
        # Wall repulsion
        for i in range(n):
            for k in range(3):
                # Positive wall
                if c[i, k] + target_r > CUBE_HALF:
                    forces[i, k] -= (c[i, k] + target_r - CUBE_HALF)
                # Negative wall
                if c[i, k] - target_r < -CUBE_HALF:
                    forces[i, k] += (-CUBE_HALF - (c[i, k] - target_r))
        
        c += lr * forces
        
        # Clip to cube (with small margin)
        c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    # Final metrics
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    wall_min = (CUBE_HALF - np.abs(c).max(axis=1)).min()
    
    return c, min_pair, wall_min


def soft_min_surrogate(centres: np.ndarray, t: float, beta: float = 50.0) -> float:
    """
    Smooth surrogate for the minimum radius using log-sum-exp.
    Minimizing this surrogate is equivalent to maximizing the min radius.
    """
    # Wall distances
    max_c = np.abs(centres).max(axis=1)
    wall_dists = CUBE_HALF - max_c
    
    # Pairwise half-distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists / 2.0
    
    # Collect all constraints: wall distances and pair distances
    # We want to maximize the minimum of all these
    # Equivalently, minimize -min = max(-x)
    # Soft max via log-sum-exp: log(sum(exp(beta * x))) / beta approximates max(x)
    
    # For numerical stability, use the soft-min formulation:
    # soft_min(x) = -log(sum(exp(-beta*x)))/beta
    # This approximates min(x) as beta -> infinity
    
    all_constraints = np.concatenate([wall_dists, pair_dists])
    
    # Soft min
    min_val = all_constraints.min()
    # Subtract min for numerical stability
    exp_vals = np.exp(-beta * (all_constraints - min_val))
    soft_min_val = min_val - np.log(exp_vals.sum()) / beta
    
    return soft_min_val


def gradient_descent_optimize(centres: np.ndarray, iters: int = 500, 
                               lr_init: float = 0.005, beta_init: float = 20.0) -> np.ndarray:
    """
    Gradient descent on the soft-min surrogate to maximize the minimum radius.
    """
    c = centres.copy().astype(float)
    n = len(c)
    
    # Adaptive parameters
    beta = beta_init
    lr = lr_init
    
    best_c = c.copy()
    best_r = compute_radius(c)
    
    for it in range(iters):
        if time.time() > DEADLINE:
            break
        
        # Compute gradient of the soft-min surrogate
        # soft_min = -log(sum(exp(-beta*x_i)))/beta
        # d(soft_min)/dx_i = exp(-beta*x_i) / sum(exp(-beta*x_j))
        
        # Wall distances
        max_c = np.abs(c).max(axis=1)
        wall_dists = CUBE_HALF - max_c
        
        # Pairwise half-distances
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Build gradient
        grad = np.zeros_like(c)
        
        # Wall gradient contribution
        # For each sphere i, the wall distance is min over axes of (CUBE_HALF - |c_ik|)
        # The gradient with respect to c_ik is -sign(c_ik) for the active axis
        
        # Soft weights for wall constraints
        min_wall = wall_dists.min()
        wall_weights = np.exp(-beta * (wall_dists - min_wall))
        wall_weights /= wall_weights.sum()
        
        for i in range(n):
            # Find which axis is most constraining
            abs_c = np.abs(c[i])
            min_axis = np.argmin(CUBE_HALF - abs_c)
            sign_val = np.sign(c[i, min_axis])
            if sign_val == 0:
                sign_val = 1
            grad[i, min_axis] += wall_weights[i] * (-sign_val)
        
        # Pairwise gradient contribution
        min_pair = pair_dists.min()
        pair_weights = np.exp(-beta * (pair_dists - min_pair))
        pair_weights /= pair_weights.sum()
        
        for i in range(n):
            for j in range(i + 1, n):
                if pair_weights[i, j] > 1e-10:
                    diff = c[i] - c[j]
                    d = np.linalg.norm(diff)
                    if d > 1e-10:
                        # d(pair_dist)/dc_i = diff/d
                        grad[i] += pair_weights[i, j] * (diff / d)
                        grad[j] -= pair_weights[i, j] * (diff / d)
        
        # Update: we want to maximize soft_min, so move in direction of +grad
        c += lr * grad
        
        # Clip to cube
        c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
        
        # Check improvement
        current_r = compute_radius(c)
        if current_r > best_r:
            best_r = current_r
            best_c = c.copy()
        
        # Adaptive learning rate and beta
        if it % 50 == 0 and it > 0:
            lr *= 0.95
            beta *= 1.2
        
        # Early stopping if no improvement for many iterations
        if it > 100 and current_r <= best_r * (1 + 1e-8):
            if it > 200:
                break
    
    return best_c


def local_search(centres: np.ndarray, iters: int = 50, step_size: float = 0.02) -> np.ndarray:
    """
    Simple local search: try small random perturbations and keep if radius improves.
    """
    c = centres.copy().astype(float)
    n = len(c)
    
    def radius_of(arr):
        dists = cdist(arr, arr)
        np.fill_diagonal(dists, np.inf)
        min_pair = dists.min() / 2.0
        wall_min = (CUBE_HALF - np.abs(arr).max(axis=1)).min()
        return min(min_pair, wall_min)
    
    best_r = radius_of(c)
    
    for it in range(iters):
        if time.time() > DEADLINE:
            break
            
        # Pick a random sphere and try a small move
        i = np.random.randint(n)
        original = c[i].copy()
        
        # Try a few random directions
        for _ in range(10):
            delta = np.random.randn(3) * step_size
            c[i] = original + delta
            
            # Keep in cube
            c[i] = np.clip(c[i], -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
            
            r = radius_of(c)
            if r > best_r:
                best_r = r
            else:
                c[i] = original
                break
    
    return c


def construct() -> np.ndarray:
    """
    Construct centres for 100 equal spheres in unit cube.
    Uses grid initialization + FEA optimization + gradient descent on soft-min surrogate + local search.
    """
    t_start = time.time()
    deadline = DEADLINE
    
    # Start with grid
    c = initial_grid(N)
    r_init = compute_radius(c)
    
    # Target radius: slightly above current best known
    # Best known: ~0.11138, try to beat that
    target_r = 0.112
    
    # Phase 1: FEA to expand spacing
    c, _, _ = fea_optimize(c, target_r, iters=300, lr=0.002)
    
    if time.time() > deadline:
        # Return whatever we have
        return np.clip(c, -CUBE_HALF, CUBE_HALF)
    
    r_after_fea = compute_radius(c)
    
    # Phase 2: Gradient descent on soft-min surrogate
    if time.time() < deadline - 30:
        c = gradient_descent_optimize(c, iters=300, lr_init=0.003, beta_init=30.0)
        r_after_gd = compute_radius(c)
    
    # Phase 3: Local search to improve
    if time.time() < deadline - 10:
        c = local_search(c, iters=100, step_size=0.01)
    
    # Final validation
    c = np.clip(c, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return c
