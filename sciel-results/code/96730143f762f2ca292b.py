"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START

def _fcc_lattice(n=100):
    """Generate points on FCC lattice, then select the best n by local density."""
    # FCC lattice: points at (i, j, k) where i+j+k is even, scaled to fit in [-0.5, 0.5]^3
    # Generate a grid of FCC points
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
    # Let's generate a dense set of candidate points on FCC lattice
    # Scale: we want spacing ~ 1/5 or so initially, then refine
    # Actually, let's just generate a fine grid and pick FCC-like subset
    
    # Better approach: generate FCC lattice points in a larger region, then select
    # FCC lattice constant a: nearest neighbor distance = a/sqrt(2)
    # For 100 spheres in unit cube, typical spacing ~ 1/4.64 ~ 0.215
    # So a ~ 0.215 * sqrt(2) ~ 0.304
    
    a = 0.25  # lattice constant, will adjust
    points = []
    # FCC basis vectors
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * a
    
    # Generate lattice points in range [-0.5, 0.5]
    n_max = int(np.ceil(1.0 / a)) + 2
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            for k in range(-n_max, n_max + 1):
                origin = np.array([i, j, k]) * a
                for b in basis:
                    p = origin + b
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    
    points = np.array(points)
    
    # If we have more than n points, select the best n
    if len(points) > n:
        # Use a greedy selection: pick point that maximizes min distance to already selected
        # But this is O(n^2 * m) which might be slow for large m
        # Alternative: random subset, then optimize
        # Let's just take a random subset for speed, the optimizer will fix it
        idx = np.random.choice(len(points), n, replace=False)
        points = points[idx]
    elif len(points) < n:
        # Pad with random points
        extra = np.random.uniform(-0.5, 0.5, (n - len(points), 3))
        points = np.vstack([points, extra])
    
    return points


def _score(points):
    """Compute the minimum radius for the given configuration."""
    n = len(points)
    if n < 2:
        return 0.5
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)


def _soft_min(values, beta):
    """Soft minimum using log-sum-exp trick."""
    values = np.asarray(values, dtype=np.float64)
    m = np.min(values)
    return m - np.log(np.sum(np.exp(beta * (values - m)))) / beta


def _soft_score(points, beta=50.0):
    """Smooth surrogate for the min radius score."""
    n = len(points)
    if n < 2:
        return 0.5
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_soft = _soft_min(dists / 2.0, beta)
    
    wall_dists = 0.5 - np.abs(points)
    min_wall_soft = _soft_min(wall_dists, beta)
    
    return _soft_min(np.array([min_pair_soft, min_wall_soft]), beta)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize with FCC lattice
    rng = np.random.RandomState(42)
    points = _fcc_lattice(N)
    
    # Ensure validity: clip to cube interior
    points = np.clip(points, -0.49, 0.49)
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Simulated annealing with smooth objective
    T = 0.05  # initial temperature
    T_min = 0.001
    cooling = 0.995
    
    # Precompute for efficiency
    def _score_fast(pts):
        """Vectorized score computation."""
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        wall_dists = 0.5 - np.abs(pts)
        min_wall = np.min(wall_dists)
        return min(min_pair, min_wall)
    
    def _score_fast_delta(pts, old_score):
        """Fast score that can early-terminate if already worse than old_score."""
        n = len(pts)
        # Check walls first (cheap)
        wall_dists = 0.5 - np.abs(pts)
        min_wall = np.min(wall_dists)
        if min_wall < old_score:
            return min_wall
        # Check pairs
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        return min(min_pair, min_wall)
    
    it = 0
    while time.time() < deadline:
        it += 1
        
        # Propose a move: pick random point, perturb
        i = rng.randint(N)
        step_size = 0.01 * T / 0.05  # scale step with temperature
        delta = rng.normal(0, step_size, 3)
        new_points = points.copy()
        new_points[i] += delta
        
        # Keep in bounds
        new_points[i] = np.clip(new_points[i], -0.5, 0.5)
        
        old_s = _score_fast(points)
        new_s = _score_fast_delta(new_points, old_s)
        
        if new_s >= old_s or rng.random() < np.exp((new_s - old_s) / T):
            points = new_points
            if new_s > best_score:
                best_score = new_s
                best_points = points.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Every 5000 iterations, do a local polish
        if it % 5000 == 0 and time.time() < deadline - 10.0:
            # Gradient-free local search: try small moves on all points
            for _ in range(20):
                for j in range(N):
                    for axis in range(3):
                        for direction in [-1, 1]:
                            trial = points.copy()
                            trial[j, axis] += direction * 0.001
                            trial[j, axis] = np.clip(trial[j, axis], -0.5, 0.5)
                            ts = _score_fast_delta(trial, _score_fast(points))
                            if ts > _score_fast(points):
                                points = trial
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            
            if _score_fast(points) > best_score:
                best_score = _score_fast(points)
                best_points = points.copy()
    
    # Final polish: try to improve with very small steps
    if time.time() < deadline - 2.0:
        for _ in range(100):
            improved = False
            for j in range(N):
                for axis in range(3):
                    for direction in [-1, 1]:
                        trial = best_points.copy()
                        trial[j, axis] += direction * 0.0001
                        trial[j, axis] = np.clip(trial[j, axis], -0.5, 0.5)
                        ts = _score_fast(trial)
                        if ts > best_score:
                            best_points = trial
                            best_score = ts
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    # Ensure final result is valid
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points

# EVOLVE-BLOCK-END
