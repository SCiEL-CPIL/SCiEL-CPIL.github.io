import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree

N = 300

def construct():
    """
    Constructs the centers of 300 circles inside a unit circle.
    Uses simulated annealing with momentum, a good hexagonal initialisation,
    and a final local optimisation pass.
    """
    def score(pos):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=0.15, output_type='ndarray')
        if len(pairs) == 0:
            min_pair = 2.0
        else:
            diff = pos[pairs[:, 0]] - pos[pairs[:, 1]]
            min_pair = np.min(np.linalg.norm(diff, axis=1))
        radii = np.linalg.norm(pos, axis=1)
        min_boundary = np.min(1.0 - radii)
        r = min(min_pair / 2.0, min_boundary)
        return r

    def generate_hex_grid(n_points):
        # Estimate spacing
        s = np.sqrt(2 * np.pi / (3 * np.sqrt(3) * n_points))
        max_i = int(np.ceil(1.5 / s)) + 3
        max_j = int(np.ceil(1.5 / s)) + 3
        points = []
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = s * (i + 0.5 * j)
                y = s * (np.sqrt(3) / 2) * j
                if x * x + y * y <= 1.0:
                    points.append((x, y))
        points = np.array(points)
        
        if len(points) < n_points:
            s *= 0.92
            max_i = int(np.ceil(1.5 / s)) + 3
            max_j = int(np.ceil(1.5 / s)) + 3
            points = []
            for j in range(-max_j, max_j + 1):
                for i in range(-max_i, max_i + 1):
                    x = s * (i + 0.5 * j)
                    y = s * (np.sqrt(3) / 2) * j
                    if x * x + y * y <= 1.0:
                        points.append((x, y))
            points = np.array(points)

        if len(points) > n_points:
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:n_points]
            points = points[indices]
        elif len(points) < n_points:
            extra = np.random.uniform(-0.8, 0.8, (n_points - len(points), 2))
            points = np.vstack([points, extra])
            
        return points

    # Initial position: hex grid
    pos = generate_hex_grid(N)
    # Add a tiny random perturbation to break symmetry
    pos += np.random.normal(0, 1e-4, pos.shape)
    
    # Simulated Annealing
    current_score = score(pos)
    best_pos = pos.copy()
    best_score = current_score
    
    T_start = 0.05
    T_end = 1e-5
    n_iter = 30000
    alpha = (T_end / T_start) ** (1.0 / n_iter)
    momentum = 0.5
    velocity = np.zeros_like(pos)
    base_step = 0.01
    
    for it in range(n_iter):
        T = T_start * (alpha ** it)
        step = base_step * (T / T_start) + 0.001
        
        # Generate candidate: move all points randomly with momentum
        delta = np.random.normal(0, step, pos.shape)
        velocity = momentum * velocity + delta
        new_pos = pos + velocity
        
        # Project outside points inward
        radii = np.linalg.norm(new_pos, axis=1)
        outside = radii > 1.0
        if np.any(outside):
            new_pos[outside] = new_pos[outside] / radii[outside, np.newaxis] * 0.9999
        
        new_score = score(new_pos)
        
        # Accept or reject
        if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
            pos = new_pos
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
    
    # Final local refinement: gradient-free local search with smaller steps
    pos = best_pos.copy()
    current_score = best_score
    for _ in range(2000):
        # Find the worst point (min distance to any other or boundary)
        radii = np.linalg.norm(pos, axis=1)
        boundary_dists = 1.0 - radii
        
        tree = cKDTree(pos)
        dists, _ = tree.query(pos, k=2)
        nearest_dists = dists[:, 1]
        
        # Score per point: min(nearest_dist/2, boundary_dist)
        point_scores = np.minimum(nearest_dists / 2.0, boundary_dists)
        worst_idx = np.argmin(point_scores)
        
        # Try to improve the worst point
        worst_score = point_scores[worst_idx]
        improved = False
        for _ in range(20):
            trial = pos[worst_idx].copy()
            trial += np.random.normal(0, 0.005, 2)
            
            # Check boundary
            r_trial = np.linalg.norm(trial)
            if r_trial > 1.0:
                trial = trial / r_trial * 0.9999
                r_trial = 1.0 * 0.9999
            
            # Check nearest neighbor
            d_nearest = np.min(np.linalg.norm(pos - trial, axis=1))
            # exclude self
            if d_nearest < 1e-9:
                d_nearest = np.inf
            
            trial_score = min(d_nearest / 2.0, 1.0 - r_trial)
            
            if trial_score > worst_score:
                pos[worst_idx] = trial
                worst_score = trial_score
                improved = True
                break
        
        if not improved:
            # Try a small random kick
            pos[worst_idx] += np.random.normal(0, 0.001, 2)
            r = np.linalg.norm(pos[worst_idx])
            if r > 1.0:
                pos[worst_idx] = pos[worst_idx] / r * 0.9999
        
        new_score = score(pos)
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
    
    return best_pos
