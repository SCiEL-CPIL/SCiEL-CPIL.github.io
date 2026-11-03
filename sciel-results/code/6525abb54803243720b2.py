"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def construct():
    """
    Place 50 equal non-overlapping circles in the unit square [-0.5, 0.5]^2
    to maximize the common radius.
    
    Uses multi-restart Simulated Annealing with optimized scoring and diverse initializations.
    """
    
    def score(points):
        """Calculate the maximum radius for a given configuration - optimized with cKDTree."""
        # Use cKDTree for efficient nearest neighbor queries
        tree = cKDTree(points)
        # Query for 2 nearest neighbors (self + 1 other)
        dists, _ = tree.query(points, k=2)
        min_pair_dist = np.min(dists[:, 1])
        
        # Distance to boundaries
        dist_x = 0.5 - np.abs(points[:, 0])
        dist_y = 0.5 - np.abs(points[:, 1])
        min_boundary_dist = np.min(np.minimum(dist_x, dist_y))
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    def make_hex_grid(n_rows, n_cols, jitter=0.0, seed=42):
        """Create a hexagonal grid pattern for better initial packing."""
        rng = np.random.RandomState(seed)
        # Hexagonal spacing
        dx = 1.0 / (n_cols + 0.5)
        dy = dx * np.sqrt(3) / 2
        
        points = []
        for i in range(n_rows):
            for j in range(n_cols):
                x = -0.5 + dx/2 + j * dx + (dx/2 if i % 2 == 1 else 0)
                y = -0.5 + dy/2 + i * dy
                points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            # Add more points to reach N
            while len(points) < N:
                x = rng.uniform(-0.45, 0.45)
                y = rng.uniform(-0.45, 0.45)
                points = np.vstack([points, [x, y]])
        elif len(points) > N:
            points = points[:N]
        
        if jitter > 0:
            points += rng.normal(0, jitter, size=points.shape)
            points = np.clip(points, -0.5, 0.5)
        return points

    def make_square_grid(rows, cols, jitter=0.0, seed=42):
        """Create a square grid pattern."""
        rng = np.random.RandomState(seed)
        xs = np.linspace(-0.45, 0.45, cols)
        ys = np.linspace(-0.45, 0.45, rows)
        X, Y = np.meshgrid(xs, ys)
        points = np.column_stack((X.ravel(), Y.ravel()))
        
        if len(points) < N:
            while len(points) < N:
                x = rng.uniform(-0.45, 0.45)
                y = rng.uniform(-0.45, 0.45)
                points = np.vstack([points, [x, y]])
        elif len(points) > N:
            points = points[:N]
        
        if jitter > 0:
            points += rng.normal(0, jitter, size=points.shape)
            points = np.clip(points, -0.5, 0.5)
        return points

    def anneal(points, T_start, T_end, alpha, step_size, time_budget, rng, verbose=False):
        """Run simulated annealing from given initial configuration."""
        current_score = score(points)
        best_points = points.copy()
        best_score = current_score
        
        T = T_start
        ss = step_size
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 5000
        
        while T > T_end and time.time() - start_time < time_budget:
            idx = rng.randint(N)
            delta = rng.normal(0, ss, size=2)
            
            new_points = points.copy()
            new_points[idx] += delta
            new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
            
            new_score = score(new_points)
            
            if new_score > current_score:
                points = new_points
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                no_improve_count += 1
                prob = np.exp((new_score - current_score) / T)
                if rng.random() < prob:
                    points = new_points
                    current_score = new_score
                    
            T *= alpha
            ss *= 0.9998
            # Adaptive step size: increase if stuck
            if no_improve_count > max_no_improve:
                ss = min(ss * 1.5, 0.05)
                no_improve_count = 0
                T = max(T * 1.2, T_end * 10)
                
        return best_points, best_score

    def hill_climb(points, time_budget, rng):
        """Refine configuration with local search using adaptive step size."""
        current_score = score(points)
        best_points = points.copy()
        best_score = current_score
        step_size = 0.003
        no_improve_count = 0
        
        start_time = time.time()
        
        while time.time() - start_time < time_budget:
            idx = rng.randint(N)
            delta = rng.normal(0, step_size, size=2)
            
            new_points = points.copy()
            new_points[idx] += delta
            new_points[idx] = np.clip(new_points[idx], -0.5, 0.5)
            
            new_score = score(new_points)
            
            if new_score > current_score:
                points = new_points
                current_score = new_score
                no_improve_count = 0
                step_size = min(step_size * 1.05, 0.01)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                no_improve_count += 1
                step_size *= 0.97
                step_size = max(step_size, 0.0002)
                
                if no_improve_count > 1000:
                    # Try larger jump to escape local minimum
                    if rng.random() < 0.1:
                        step_size = 0.01
                        no_improve_count = 0
                        
        return best_points, best_score

    # Multi-restart strategy with diverse initial configurations
    np.random.seed(42)
    overall_best_points = None
    overall_best_score = -1
    
    start_time = time.time()
    time_limit = 115.0
    
    # Define diverse initial configurations
    initializers = [
        ("hex_5x10", lambda: make_hex_grid(5, 10, jitter=0.0, seed=1)),
        ("hex_7x8", lambda: make_hex_grid(7, 8, jitter=0.01, seed=2)),
        ("square_10x5", lambda: make_square_grid(10, 5, jitter=0.0, seed=3)),
        ("square_5x10", lambda: make_square_grid(5, 10, jitter=0.0, seed=4)),
        ("hex_8x7", lambda: make_hex_grid(8, 7, jitter=0.015, seed=5)),
        ("square_7x8", lambda: make_square_grid(7, 8, jitter=0.02, seed=6)),
        ("hex_6x9", lambda: make_hex_grid(6, 9, jitter=0.02, seed=7)),
        ("random", lambda: np.random.uniform(-0.45, 0.45, size=(N, 2))),
    ]
    
    for name, init_func in initializers:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15:
            break
            
        try:
            points = init_func()
            points = np.clip(points, -0.5, 0.5)
            if points.shape != (N, 2) or not np.all(np.isfinite(points)):
                continue
                
            rng = np.random.RandomState(hash(name) % (2**32))
            
            # Annealing phase
            anneal_time = min(30.0, time_limit - 10 - elapsed)
            if anneal_time <= 5:
                continue
                
            best_pts, best_sc = anneal(points, 
                                        T_start=0.04, 
                                        T_end=0.0003, 
                                        alpha=0.9997, 
                                        step_size=0.05, 
                                        time_budget=anneal_time, 
                                        rng=rng)
            
            # Hill climbing phase
            elapsed2 = time.time() - start_time
            hc_time = min(12.0, time_limit - 3 - elapsed2)
            if hc_time > 1:
                best_pts, best_sc = hill_climb(best_pts, hc_time, rng)
            
            if best_sc > overall_best_score:
                overall_best_score = best_sc
                overall_best_points = best_pts.copy()
                
        except Exception:
            continue
    
    # Final refinement on best solution
    if overall_best_points is not None:
        remaining = time_limit - 1 - (time.time() - start_time)
        if remaining > 3:
            rng = np.random.RandomState(12345)
            overall_best_points, overall_best_score = hill_climb(overall_best_points, remaining, rng)
    
    if overall_best_points is None:
        # Fallback
        overall_best_points = make_square_grid(10, 5, jitter=0.0, seed=42)
    
    return overall_best_points

# EVOLVE-BLOCK-END
