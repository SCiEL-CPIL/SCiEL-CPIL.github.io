"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _score(points):
    """Compute the raw radius score for a set of centers."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check inside square
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0
    # Compute min distance between pairs / 2
    # and min distance to boundary
    # Boundary distance: min(0.5 - |x|, 0.5 - |y|) for each point
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    boundary = np.minimum(dx, dy)
    min_boundary = np.min(boundary)
    
    # Pairwise distances
    # Use broadcasting for N=50 (2500 pairs)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    
    return min(min_pair, min_boundary)


def _soft_min(values, beta):
    """Soft min approximation using log-sum-exp."""
    # soft min = -1/beta * log(sum(exp(-beta * x)))
    # For numerical stability
    m = np.min(values)
    return m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (values - m))))


def _soft_score(points, beta):
    """Smooth surrogate for the score."""
    if points.shape != (N, 2):
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
    # Boundary terms
    dx = 0.5 - np.abs(points[:, 0])
    dy = 0.5 - np.abs(points[:, 1])
    boundary = np.minimum(dx, dy)
    min_boundary = _soft_min(boundary, beta)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # We want min over i<j of dist_ij / 2
    # Take upper triangle
    iu = np.triu_indices(N, k=1)
    pair_dists = dist[iu] / 2.0
    min_pair = _soft_min(pair_dists, beta)
    
    return min(min_pair, min_boundary)


def _project_to_square(points):
    """Project points back into the unit square [-0.5, 0.5]^2."""
    points = np.clip(points, -0.5, 0.5)
    return points


def _initialize_grid():
    """Create initial grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _initialize_random():
    """Create random initialization within the square."""
    return np.random.uniform(-0.45, 0.45, (N, 2))


def _initialize_hexagonal():
    """Create a hexagonal-like initialization."""
    # Try to fit 50 circles in a hexagonal pattern
    # Approximate: use rows with offset
    points = []
    # Try different row counts
    best_points = None
    best_score = -1
    
    for rows in range(5, 11):
        cols = int(math.ceil(N / rows))
        # Create hexagonal grid
        for i in range(rows):
            for j in range(cols):
                if len(points) >= N:
                    break
                x = (j + (i % 2) * 0.5) / (cols - 0.5) - 0.5
                y = (i + 0.5) / rows - 0.5
                # Scale to fit
                x = x * 0.9
                y = y * 0.9
                points.append((x, y))
        points = np.array(points[:N])
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    return best_points if best_points is not None else _initialize_grid()


def _lbfgs_optimize(points, max_iter=2000, time_limit=100):
    """Optimize using L-BFGS-B with smooth surrogate."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Start with moderate beta and decrease
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit:
            break
        
        def objective(x):
            pts = x.reshape(N, 2)
            return -_soft_score(pts, beta)
        
        def gradient(x):
            pts = x.reshape(N, 2)
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros_like(x)
            f0 = objective(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                f_plus = objective(x_plus)
                grad[i] = (f_plus - f0) / eps
            return grad
        
        # Use L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                objective,
                points.flatten(),
                method='L-BFGS-B',
                jac=gradient,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < objective(points.flatten()):
                new_points = result.x.reshape(N, 2)
                new_points = _project_to_square(new_points)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
    
    return best_points, best_score


def _local_search(points, n_iterations=500, time_limit=30):
    """Local search with random perturbations."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Get current pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Find the closest pair
        iu = np.triu_indices(N, k=1)
        min_idx = np.argmin(dist[iu])
        i, j = iu[0][min_idx], iu[1][min_idx]
        
        # Separate the closest pair
        d = points[i] - points[j]
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-10:
            d = np.random.normal(0, 1e-3, 2)
            d_norm = np.linalg.norm(d)
        
        direction = d / d_norm
        sep = (dist[i, j] - 2 * best_score) / 2 + 0.001
        if sep < 0:
            sep = 0.001
        
        new_points = points.copy()
        new_points[i] += direction * sep
        new_points[j] -= direction * sep
        
        # Project to square
        new_points = _project_to_square(new_points)
        
        new_score = _score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            # Update distances
            diff = new_points[:, np.newaxis, :] - new_points[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
    
    return best_points, best_score


def _simulated_annealing(points, time_limit=60):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(points)
    current_points = points.copy()
    current_score = _score(points)
    
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling = 0.999
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Propose a move: perturb a random point
        idx = np.random.randint(N)
        delta = np.random.normal(0, T * 0.5, 2)
        new_points = current_points.copy()
        new_points[idx] += delta
        new_points = _project_to_square(new_points)
        
        new_score = _score(new_points)
        delta_score = new_score - current_score
        
        if delta_score > 0 or np.random.random() < math.exp(delta_score / T):
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling
    
    return best_points, best_score


def construct():
    """Construct the optimal packing of 50 circles in a unit square."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Multiple initializations and optimization
    best_points = None
    best_score = -1
    
    # Try different initializations
    initializations = [
        _initialize_grid(),
        _initialize_hexagonal(),
        _initialize_random(),
        _initialize_random(),
        _initialize_random(),
    ]
    
    for init in initializations:
        if time.time() > deadline:
            break
        
        # Local search first
        points, score = _local_search(init, n_iterations=200, time_limit=10)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # L-BFGS optimization
        if time.time() < deadline - 20:
            points, score = _lbfgs_optimize(points, max_iter=1000, time_limit=20)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        
        # More local search
        if time.time() < deadline - 10:
            points, score = _local_search(points, n_iterations=300, time_limit=15)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # Simulated annealing on the best solution
    if time.time() < deadline - 10 and best_points is not None:
        points, score = _simulated_annealing(best_points, time_limit=40)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Final polish with L-BFGS
    if time.time() < deadline - 5 and best_points is not None:
        points, score = _lbfgs_optimize(best_points, max_iter=500, time_limit=10)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Final local search
    if time.time() < deadline - 1 and best_points is not None:
        points, score = _local_search(best_points, n_iterations=200, time_limit=5)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    if best_points is None:
        best_points = _initialize_grid()
    
    # Ensure validity
    best_points = _project_to_square(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
