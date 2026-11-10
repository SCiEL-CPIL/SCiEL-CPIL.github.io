# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _lj_energy(points, k_rep=12, k_att=6):
    """
    Compute Lennard-Jones-like energy.
    We want to maximize the ratio of min to max distance.
    A common heuristic for maximizing minimum distance (Tammes problem) is minimizing
    a potential that strongly penalizes close distances and weakly penalizes far distances.
    However, we also want to minimize the max distance to keep the cluster tight.
    
    A better potential for "equidistant" packing is:
    E = sum (1/d^12 - 1/d^6) is standard LJ, but it doesn't bound the max distance well.
    
    Alternative: Minimize variance of distances? No, we want max/min ratio.
    
    Let's use a potential that pushes points apart if too close, and pulls them together if too far.
    Target distance d_target = 1.0 (scale invariant).
    E = sum ( (d - 1)^2 )? This minimizes variance around 1.
    But we specifically want to maximize d_min and minimize d_max.
    
    Let's try: E = sum( 1/d^2 ) + sum( d^2 )?
    Or simply use the negative of the score? Gradient is hard.
    
    Let's use a Simulated Annealing approach directly on the score R.
    But SA on a non-smooth function (min/max) is tricky.
    
    Let's use a smoother proxy: Minimize the ratio of max to min distance.
    Objective: f = d_max / d_min. Minimize f.
    Gradient of f is non-trivial.
    
    Alternative Strategy: 
    Use a force-based model.
    Force on point i:
    If d_ij < d_target: Repulsive force F ~ 1/d^12
    If d_ij > d_target: Attractive force F ~ 1/d^2 (weak) or linear?
    
    Actually, for maximizing d_min/d_max, a very effective heuristic is to minimize the 
    "spread" of distances. 
    Let's try minimizing: Sum( (d_ij - d_mean)^2 ) / (d_mean^2) ?
    Or simply: Minimize (d_max - d_min).
    
    Let's stick to Simulated Annealing on the raw score R, but with a smarter move generation.
    Moves: 
    1. Random perturbation of one point.
    2. Perturbation of a pair of points (closest or farthest).
    
    Since R is (d_min/d_max)^2, it is 0 if any points coincide.
    We need to avoid coincident points.
    
    Let's implement a robust SA:
    1. Start with a good initial guess (e.g., 4x4 grid, circle, etc.)
    2. Define a cost function C = 1/R (minimize C). If R=0, C=inf.
    3. Move: Pick random point, add Gaussian noise.
    4. Accept if C_new < C_old or with probability exp(-(C_new-C_old)/T).
    5. Cool T.
    
    To speed up, we can use a "fast" score evaluation.
    pdist is fast for 16 points (120 distances).
    
    Let's also try a specific geometric optimization:
    The optimal configuration for 16 points in 2D to maximize min distance is likely related to a 
    slightly distorted square grid or a hexagonal packing fragment.
    
    Let's combine:
    1. Generate diverse seeds (Grid, Circle, Hex, Random).
    2. Run SA on each seed.
    3. Track best.
    """
    pass

def _simulated_annealing(points_init, iters=5000, T_start=1.0, T_end=0.01, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    n = len(points_init)
    points = points_init.copy()
    score = _compute_score(points)
    if score == 0:
        return points, 0.0
        
    cost = 1.0 / score
    best_points = points.copy()
    best_cost = cost
    best_score = score
    
    # Precompute pair indices for faster access? pdist is C-optimized, probably fine.
    
    T = T_start
    cooling_rate = (T_end / T_start) ** (1.0 / iters)
    
    for i in range(iters):
        # Generate a move
        # Randomly pick a point to move
        idx = rng.integers(0, n)
        # Random direction and magnitude
        delta = rng.normal(0, 0.1, size=2)
        
        new_points = points.copy()
        new_points[idx] += delta
        
        new_score = _compute_score(new_points)
        
        if new_score == 0:
            new_cost = np.inf
        else:
            new_cost = 1.0 / new_score
            
        # Acceptance criterion
        if new_cost < cost:
            cost = new_cost
            points = new_points
            if new_cost < best_cost:
                best_cost = new_cost
                best_points = points.copy()
                best_score = new_score
        else:
            # Accept with probability exp(-(new_cost - cost) / T)
            # Avoid overflow/underflow
            delta_cost = new_cost - cost
            if delta_cost < 0:
                delta_cost = 0
            # If T is very small, exp(-large) is 0
            if T > 1e-10:
                prob = np.exp(-delta_cost / T)
                if rng.random() < prob:
                    cost = new_cost
                    points = new_points
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_points = points.copy()
                        best_score = new_score
        
        T *= cooling_rate
        
    return best_points, best_score

def _local_refine(points, iters=1000, lr=0.05, rng=None):
    """
    Gradient-free local refinement using coordinate descent on the score.
    Try to increase d_min and decrease d_max.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n = len(points)
    points = points.copy()
    best_score = _compute_score(points)
    best_points = points.copy()
    
    if best_score == 0:
        return best_points, 0.0
        
    for i in range(iters):
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        
        # Identify critical pairs
        min_idx = np.argmin(d)
        max_idx = np.argmax(d)
        
        # Convert flat index to (i, j)
        # For n=16, we can hardcode or use a lookup. 
        # pdist index for (i, j) is i*n - i*(i+1)/2 + (j-i-1)
        # Inverse:
        def idx_to_pair(idx, n):
            # Approximate i
            # idx = i*n - i*(i+1)/2 + k
            # i*(2n-i-1)/2 < idx <= (i+1)(2n-i-2)/2
            # For n=16:
            # 0: (0,1)
            # 1: (0,2) ... 14: (0,15)
            # 15: (1,2) ... 28: (1,15)
            # ...
            i = 0
            count = 0
            while count + (n - i - 1) <= idx:
                count += (n - i - 1)
                i += 1
            j = i + 1 + (idx - count)
            return i, j
        
        i_min, j_min = idx_to_pair(min_idx, n)
        i_max, j_max = idx_to_pair(max_idx, n)
        
        # Move min pair apart
        vec_min = points[j_min] - points[i_min]
        norm_min = np.linalg.norm(vec_min)
        if norm_min > 1e-8:
            dir_min = vec_min / norm_min
            step = lr * 0.5
            points[i_min] -= step * dir_min
            points[j_min] += step * dir_min
            
        # Move max pair together
        vec_max = points[j_max] - points[i_max]
        norm_max = np.linalg.norm(vec_max)
        if norm_max > 1e-8:
            dir_max = vec_max / norm_max
            step = lr * 0.5
            points[i_max] += step * dir_max
            points[j_max] -= step * dir_max
            
        # Small random jitter to avoid getting stuck
        points += rng.normal(0, lr * 0.01, size=points.shape)
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    return best_points, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin
    
    n_points = 16
    n_dim = 2
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = 0.0
    
    # 1. Generate diverse initial configurations
    configs = []
    
    # Regular polygon
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    configs.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 4x4 Grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    configs.append(np.column_stack([xx.ravel(), yy.ravel()]))
    
    # Hexagonal lattice (approx)
    pts_hex = []
    for row in range(4):
        for col in range(4):
            x = col * 1.0
            y = row * np.sqrt(3)/2
            if col % 2 == 1:
                y += np.sqrt(3)/4
            pts_hex.append([x, y])
    pts_hex = np.array(pts_hex)
    pts_hex -= pts_hex.mean(axis=0)
    configs.append(pts_hex)
    
    # Two concentric circles
    r1, r2 = 0.5, 1.5
    a1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    a2 = np.linspace(np.pi/8, 2*np.pi + np.pi/8, 8, endpoint=False)
    p1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
    configs.append(np.vstack([p1, p2]))
    
    # Random configs
    for _ in range(10):
        pts = rng.standard_normal((n_points, n_dim))
        pts /= np.max(np.abs(pts))
        configs.append(pts)
        
    # 2. Optimize each configuration
    for i, cfg in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Local Refinement first (fast)
        opt_pts, score = _local_refine(cfg, iters=200, lr=0.1, rng=rng)
        if score > best_score:
            best_score = score
            best_points = opt_pts.copy()
            
        # Simulated Annealing (slower, global)
        if time.time() < deadline - 50:
            sa_pts, sa_score = _simulated_annealing(opt_pts, iters=2000, T_start=0.5, T_end=0.01, rng=rng)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_pts.copy()
                
            # Final local refinement on SA result
            final_pts, final_score = _local_refine(sa_pts, iters=500, lr=0.05, rng=rng)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
                
    # 3. If we have time, do more random restarts
    while time.time() < deadline - 10:
        pts = rng.standard_normal((n_points, n_dim))
        pts /= np.max(np.abs(pts))
        
        # Quick local refine
        opt_pts, score = _local_refine(pts, iters=100, lr=0.1, rng=rng)
        if score > best_score:
            best_score = score
            best_points = opt_pts.copy()
            
        # SA if time permits
        if time.time() < deadline - 20:
            sa_pts, sa_score = _simulated_annealing(opt_pts, iters=1000, T_start=0.5, T_end=0.01, rng=rng)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_pts.copy()
                
    if best_points is None:
        # Fallback
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END
