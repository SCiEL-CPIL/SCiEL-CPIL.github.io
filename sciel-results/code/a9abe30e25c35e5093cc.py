import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import warnings

N = 50
DEADLINE = 140.0  # seconds, leave safety margin before 150s kill


def compute_radius(pts):
    """Compute the largest common radius for given centers."""
    # Minimum distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Minimum pairwise distance
    from scipy.spatial.distance import pdist
    pair_dists = pdist(pts)
    min_pair_dist = np.min(pair_dists) if len(pair_dists) > 0 else np.inf
    
    return min(min_pair_dist / 2.0, np.min(dist_boundary))


def ring_init(n_rings):
    """Initialize points in concentric rings."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    
    ring_sizes = [1, 6, 12, 18, 24, 30, 36]  # Typical ring sizes for hexagonal packing
    idx = 1
    
    for r_idx in range(1, n_rings + 1):
        if idx >= N:
            break
        
        # Radius of this ring
        ring_radius = 0.2 * r_idx  # Adjust spacing
        
        # Number of points on this ring
        n_pts = min(ring_sizes[r_idx - 1] if r_idx < len(ring_sizes) else 36, N - len(pts))
        if n_pts <= 0:
            continue
        
        # Place points on the ring
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        # Add some jitter
        angles += np.random.uniform(-0.1, 0.1, n_pts)
        angles = np.sort(angles)
        
        for ang in angles:
            x = ring_radius * np.cos(ang)
            y = ring_radius * np.sin(ang)
            # Ensure within unit circle
            if np.hypot(x, y) < 0.95:
                pts.append([x, y])
            else:
                # Scale down to fit
                scale = 0.95 / np.hypot(x, y)
                pts.append([x * scale, y * scale])
        
        if len(pts) >= N:
            break
    
    # If we don't have enough points, add more randomly
    while len(pts) < N:
        x = np.random.uniform(-0.8, 0.8)
        y = np.random.uniform(-0.8, 0.8)
        if np.hypot(x, y) < 0.9:
            pts.append([x, y])
    
    return np.array(pts[:N])


def hex_init():
    """Initialize points in hexagonal lattice pattern."""
    spacing = 0.18
    pts = []
    
    for j in range(-5, 6):
        for i in range(-5, 6):
            x = i * spacing + (0.5 * spacing if j % 2 else 0)
            y = j * spacing * np.sqrt(3) / 2
            if np.hypot(x, y) < 0.85:
                pts.append([x, y])
    
    pts = np.array(pts)
    
    # If we have too many, select the closest to center
    if len(pts) > N:
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
    elif len(pts) < N:
        # Add random points
        while len(pts) < N:
            x = np.random.uniform(-0.8, 0.8)
            y = np.random.uniform(-0.8, 0.8)
            if np.hypot(x, y) < 0.9:
                pts = np.vstack([pts, [[x, y]]])
    
    return pts


def smooth_objective(x, beta=100.0):
    """Smooth surrogate for the min radius objective using soft-min."""
    pts = x.reshape(-1, 2)
    
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Pairwise distances
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair_dists = np.min(D, axis=1)
    
    # Soft-min of all constraints
    all_dists = np.concatenate([dist_boundary, min_pair_dists])
    
    # Soft-min using log-sum-exp
    m = np.min(all_dists)
    if m < -100:
        return -1.0
    
    exp_vals = np.exp(beta * (all_dists - m))
    soft_min = m - np.log(np.sum(exp_vals)) / beta
    
    # We want to maximize the soft-min, so minimize its negative
    return -soft_min


def optimize_from_start(pts0, time_left):
    """Optimize starting from initial points."""
    start_time = time.time()
    current_best = pts0.copy()
    current_radius = compute_radius(pts0)
    
    # Stage 1: Differential evolution for global exploration
    if time_left > 30:
        def de_objective(x):
            pts = x.reshape(-1, 2)
            # Penalty if points are outside or too close
            dist_boundary = 1.0 - np.linalg.norm(pts, axis=1)
            if np.any(dist_boundary < 0):
                return 1e10
            from scipy.spatial.distance import pdist
            pair_dists = pdist(pts)
            if np.any(pair_dists < 0.01):
                return 1e10
            r = min(np.min(pair_dists) / 2.0, np.min(dist_boundary))
            return -r
        
        bounds = [(-1, 1)] * (2 * N)
        
        try:
            result = differential_evolution(
                de_objective, bounds,
                maxiter=50, popsize=20,
                seed=42, tol=1e-6,
                polish=True
            )
            elapsed = time.time() - start_time
            if result.success and -result.fun > current_radius:
                current_best = result.x.reshape(-1, 2)
                current_radius = -result.fun
        except:
            pass
    
    # Stage 2: Local refinement using L-BFGS-B
    if time_left > 10:
        def local_objective(x):
            return smooth_objective(x, beta=200.0)
        
        constraints = []
        # Add soft constraints via penalty
        try:
            result = minimize(
                local_objective,
                current_best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if result.fun < -current_radius * 0.99:  # Better
                new_pts = result.x.reshape(-1, 2)
                new_r = compute_radius(new_pts)
                if new_r > current_radius:
                    current_best = new_pts
                    current_radius = new_r
        except:
            pass
    
    # Stage 3: Binary search polishing
    if time_left > 5:
        # Try to increase radius by moving points outward
        for _ in range(20):
            if time.time() - start_time > time_left - 2:
                break
            new_pts = current_best.copy()
            # Nudge points slightly outward
            for i in range(N):
                norm = np.linalg.norm(new_pts[i])
                if norm > 1e-6:
                    direction = new_pts[i] / norm
                    new_pts[i] += direction * 0.001
            # Check if still valid
            r_new = compute_radius(new_pts)
            if r_new > current_radius:
                current_best = new_pts
                current_radius = r_new
            else:
                break
    
    return current_best, current_radius


def construct():
    """Main entry point."""
    start_time = time.time()
    
    # Try multiple initialization strategies
    best_pts = None
    best_radius = -np.inf
    
    # 1. Ring initialization
    for trial in range(3):
        if time.time() - start_time > DEADLINE:
            break
        np.random.seed(42 + trial)
        pts0 = ring_init(5)
        time_left = DEADLINE - (time.time() - start_time)
        if time_left <= 0:
            break
        pts, r = optimize_from_start(pts0, time_left)
        if r > best_radius:
            best_radius = r
            best_pts = pts
    
    # 2. Hexagonal initialization
    if time.time() - start_time < DEADLINE:
        np.random.seed(99)
        pts0 = hex_init()
        time_left = DEADLINE - (time.time() - start_time)
        if time_left > 0:
            pts, r = optimize_from_start(pts0, time_left)
            if r > best_radius:
                best_radius = r
                best_pts = pts
    
    # 3. Random initialization
    if time.time() - start_time < DEADLINE:
        for trial in range(2):
            if time.time() - start_time > DEADLINE:
                break
            np.random.seed(123 + trial)
            pts0 = np.random.uniform(-0.8, 0.8, (N, 2))
            # Ensure within unit circle
            norms = np.linalg.norm(pts0, axis=1)
            mask = norms > 0.9
            pts0[mask] = pts0[mask] * (0.9 / norms[mask, None])
            
            time_left = DEADLINE - (time.time() - start_time)
            if time_left <= 0:
                break
            pts, r = optimize_from_start(pts0, time_left)
            if r > best_radius:
                best_radius = r
                best_pts = pts
    
    # Final safety check
    if best_pts is None:
        # Fallback: simple grid
        best_pts = np.zeros((N, 2))
        best_radius = 0.0
    
    # Ensure valid output
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.zeros((N, 2))
    
    return best_pts
