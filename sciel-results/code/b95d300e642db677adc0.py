# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Uses a multi-start local optimization approach starting from
    structured initial configurations (Fibonacci sphere, random, etc.)
    and optimizing with gradient-free methods.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 10.0  # seconds
    
    best_points = None
    best_score = -1.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    # Objective function to maximize: (d_min/d_max)^2
    # We'll minimize the negative of this, or equivalently maximize it.
    # For optimization, let's define a function that we maximize.
    
    def objective(points):
        """Returns (d_min/d_max)^2, higher is better."""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0:
            return 0.0
        if not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    
    # Try several initialization strategies
    init_strategies = []
    
    # Strategy 1: Fibonacci sphere
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    # Strategy 2: Random uniform on sphere
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    # Strategy 3: Tetrahedron-like symmetric start
    def symmetric_start(n_points):
        # Start with regular polytope approximation
        # For 14 points, use a modified icosahedron + center perturbation
        # Icosahedron has 12 vertices, add 2 more
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(icosahedron, axis=1, keepdims=True)
        icosahedron /= norms
        
        # Add 2 points: try opposite of existing or perturb
        # Add points at (0,0,0) shifted out, or use symmetry
        p13 = np.array([1.0, 1.0, 1.0])
        p14 = np.array([-1.0, -1.0, -1.0])
        p13 /= np.linalg.norm(p13)
        p14 /= np.linalg.norm(p14)
        
        return np.vstack([icosahedron, [p13, p14]])
    
    init_strategies.append(("fib_sphere", fibonacci_sphere(n)))
    init_strategies.append(("symmetric", symmetric_start(n)))
    
    # Add several random sphere starts
    for seed in range(10):
        init_strategies.append((f"rand_sphere_{seed}", random_sphere(n, seed)))
    
    # Add random Gaussian starts
    for seed in range(10):
        rng = np.random.default_rng(seed + 100)
        pts = rng.standard_normal((n, 3))
        init_strategies.append((f"rand_gauss_{seed}", pts))
    
    # For each initialization, run basin hopping optimization
    # Basin hopping is better for escaping local minima in non-smooth landscapes
    
    from scipy.optimize import basinhopping
    
    def flat_objective(x):
        """Negative objective for minimization."""
        points = x.reshape(n, d)
        return -objective(points)
    
    for name, init_points in init_strategies:
        if check_deadline():
            break
        
        x0 = init_points.flatten()
        
        try:
            # Use Powell as the local minimizer within basin hopping
            # Powell is robust for non-smooth functions
            result = basinhopping(
                flat_objective,
                x0,
                minimizer_kwargs={
                    'method': 'Powell',
                    'options': {
                        'maxiter': 1000,
                        'xatol': 1e-8,
                        'fatol': 1e-10
                    }
                },
                niter=50,  # Number of basin hops
                stepsize=1.0, # Initial step size for random jumps
                T=100.0,      # Temperature for acceptance
                accept_test=None,
                seed=42
            )
        except Exception:
            continue
        
        optimized_points = result.x.reshape(n, d)
        score = objective(optimized_points)
        
        if score > best_score:
            best_score = score
            best_points = optimized_points.copy()
        
        if check_deadline():
            break
    
    # If no valid result, use Fibonacci sphere as fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    # Ensure shape is correct
    if best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
