"""Packing 50 equal circles in a unit circle using simulated annealing and L-BFGS."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # safety margin before 120s kill


def _compute_radius(points):
    """Compute the maximum radius for a set of points."""
    # Distance to boundary
    dists_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_dist_boundary = np.min(dists_boundary)
    
    # Pairwise distances
    # Efficiently compute min pairwise distance
    # N=50 is small, so O(N^2) is fine, but we can vectorize slightly
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_dist_pair = np.min(dists)
    
    r = min(min_dist_boundary, min_dist_pair / 2.0)
    return r


def _score(points):
    """Score function for minimization (we minimize negative radius)."""
    return -_compute_radius(points)


def _hexagonal_init():
    """Generate initial points based on a hexagonal lattice."""
    # Start with a spacing that fits roughly N points
    # Area per point ~ pi*r^2. Total area pi.
    # r ~ 0.125. Area ~ 0.05. N=50. Total area 2.5. Unit circle area 3.14.
    # Spacing s = 2*r ~ 0.25.
    s = 0.24
    points = []
    # Generate a grid of points
    k_max = int(1.0 / s) + 2
    for i in range(-k_max, k_max + 1):
        for j in range(-k_max, k_max + 1):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * np.sqrt(3) / 2
            if x*x + y*y < 0.8: # Keep well inside initially
                points.append([x, y])
    
    points = np.array(points)
    # If we have more than N, keep the N closest to center? Or random?
    # Random selection from valid lattice points is usually better for diversity
    if len(points) > N:
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # Pad with random points if necessary (should not happen with s=0.24)
        while len(points) < N:
            p = np.random.uniform(-0.8, 0.8, 2)
            if np.linalg.norm(p) < 0.8:
                points = np.vstack([points, p])
            else:
                p = np.random.uniform(-0.5, 0.5, 2)
                points = np.vstack([points, p])
    return points


def _local_optimize(points):
    """Refine points using L-BFGS-B."""
    # Flatten points
    x0 = points.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 2)
        # Penalty for going outside
        max_r = np.max(np.linalg.norm(pts, axis=1))
        if max_r > 1.0:
            return 1e6 + max_r
        return _score(pts)
    
    # Use L-BFGS-B
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'gtol': 1e-6})
    best_points = res.x.reshape(-1, 2)
    
    # Ensure no points are outside
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(len(best_points)):
        if norms[i] > 1.0:
            best_points[i] = best_points[i] / norms[i] * 0.999
            
    return best_points


def construct():
    global start_time
    start_time = time.time()
    
    best_points = None
    best_score = -np.inf
    
    # Initial configuration
    points = _hexagonal_init()
    # Initial local optimization
    points = _local_optimize(points)
    score = _compute_radius(points)
    if score > best_score:
        best_score = score
        best_points = points.copy()
    
    # Simulated Annealing
    T = 0.05
    T_min = 0.001
    cooling = 0.995
    
    # We will run SA until deadline
    while time.time() - start_time < DEADLINE:
        # Generate perturbation
        # Move a random subset of points by a random amount
        n_move = np.random.randint(1, 5)
        idx = np.random.choice(N, n_move, replace=False)
        
        delta = np.random.normal(0, T, size=(n_move, 2))
        
        new_points = points.copy()
        new_points[idx] += delta
        
        # Ensure new points are inside container (hard constraint for SA step)
        # If a point goes out, pull it back or reject? 
        # Better to reject if it violates boundary significantly, or project.
        # Let's just calculate score. If invalid, score is very low.
        
        new_score = _compute_radius(new_points)
        old_score = _compute_radius(points)
        
        # Acceptance probability
        if new_score > old_score:
            points = new_points
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        else:
            # Metropolis criterion
            prob = np.exp((new_score - old_score) / T)
            if np.random.random() < prob:
                points = new_points
        
        # Cool down
        T *= cooling
        
        # Periodic local optimization to escape shallow local minima
        # Every ~500 SA steps or if we have time
        if time.time() - start_time < DEADLINE - 5:
             # Check if we should do a local optimize
             # Do it if we've made some progress or just randomly to refine
             if np.random.random() < 0.05: # 5% chance
                 points = _local_optimize(points)
                 score = _compute_radius(points)
                 if score > best_score:
                     best_score = score
                     best_points = points.copy()
                 # Reheat slightly?
                 T = max(T, 0.01)

    # Final polish
    if time.time() - start_time < DEADLINE - 2:
        best_points = _local_optimize(best_points)
        final_score = _compute_radius(best_points)
        if final_score > best_score:
            best_points = best_points # already updated
            best_score = final_score

    return best_points
