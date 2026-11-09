"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _hexagonal_lattice():
    """Generate a high-quality hexagonal lattice packing."""
    best_r = 0.0
    best_points = None
    
    # Spacing a relates to radius r by r = a/2 in ideal hex packing.
    for a in np.linspace(0.105, 0.115, 20):
        h = a * np.sqrt(3) / 2
        
        points = []
        # Generate enough points to fill the disk
        for j in range(-30, 31):
            for i in range(-30, 31):
                x = (i + 0.5 * (j % 2)) * a
                y = j * h
                # Ensure the circle fits inside the unit circle
                if x*x + y*y + a <= 1.0: # approximation for quick filter, exact check later
                    points.append([x, y])
        
        if len(points) < N:
            continue
            
        points = np.array(points)
        
        # Exact filter: ensure 1 - norm >= a/2
        norms = np.linalg.norm(points, axis=1)
        mask = (1.0 - norms) >= (a/2) - 1e-9
        points = points[mask]
        
        if len(points) < N:
            continue
        
        # Select N points closest to center to maximize symmetry and minimize boundary issues
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        selected = points[idx]
        
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_points = selected.copy()
            
    return best_points, best_r

def _repulsion_energy(points, k=10.0):
    """
    Compute a repulsive energy that is high when circles overlap or are near boundary.
    We want to minimize this energy.
    E = sum_{i<j} max(0, (2r - d_ij))^2 + sum_i max(0, (r + ||p_i|| - 1))^2
    However, r is variable. Let's use a potential that pushes points apart and inward.
    A simpler approach: minimize the sum of inverse powers of distances? No, that diverges.
    
    Let's use a soft constraint approach.
    Energy = sum_{i<j} (1 / d_ij^2) + sum_i (||p_i||^2)
    This minimizes distances and pushes to center? No, we want to maximize min distance.
    
    Better: Simulated Annealing on the objective:
    Maximize min( min_ij d_ij/2, min_i (1-||p_i||) )
    
    Let's use a potential energy function:
    U = sum_{i<j} (1/d_ij)^p + sum_i (||p_i||/R)^q
    where p and q are large. Minimizing U pushes points apart and to the center.
    But we also need to ensure they stay inside.
    
    Let's use a force-based approach.
    Force on i from j: if d_ij < 2r, push apart.
    Force from boundary: if ||p_i|| + r > 1, push in.
    
    Since r is not fixed, we can use a "virtual" r that is the current best estimate,
    or simply use a repulsive potential that doesn't depend on r explicitly,
    and then compute the resulting r.
    
    Let's use: U = sum_{i<j} (sigma / d_ij)^12 - (sigma / d_ij)^6  (Lennard-Jones like)
    This has a minimum at some distance. We want them to settle at a distance.
    But we don't know the target distance.
    
    Alternative:
    Use the objective directly in a stochastic local search or annealing.
    Objective: f(P) = min( min_ij d_ij/2, min_i (1-||p_i||) )
    
    Let's implement a simple Simulated Annealing that perturbs points and accepts moves
    if they improve the objective, or with some probability if they don't (to escape local minima).
    """
    pass # Placeholder, we'll implement the main loop in construct or a separate function.

def _anneal(points, time_limit):
    """
    Simulated Annealing to improve the packing.
    """
    start_time = time.time()
    current_points = points.copy()
    current_r = _compute_radius(current_points)
    best_points = current_points.copy()
    best_r = current_r
    
    T = 0.01  # Initial temperature
    cooling = 0.999
    
    # Precompute indices for pairwise distances? No, just compute on fly or vectorize.
    
    while time.time() < time_limit:
        # Perturb a few random points
        n_perturb = np.random.randint(1, 5)
        idx = np.random.choice(N, n_perturb, replace=False)
        
        new_points = current_points.copy()
        # Add random noise
        noise = np.random.normal(0, T, size=(n_perturb, 2))
        new_points[idx] += noise
        
        # Keep points inside unit circle (hard constraint for stability, soft in objective)
        # If a point goes outside, pull it back to the boundary?
        # Or just let the objective penalize it.
        # Let's clamp to unit circle to avoid divergence.
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] /= norms[mask]
        
        new_r = _compute_radius(new_points)
        
        # Acceptance probability
        if new_r > current_r:
            # Always accept improvement
            current_points = new_points
            current_r = new_r
            if current_r > best_r:
                best_points = current_points.copy()
                best_r = current_r
        else:
            # Metropolis criterion
            delta = new_r - current_r
            if delta < 0:
                # Accept with probability exp(delta / T)
                if np.random.rand() < np.exp(delta / T):
                    current_points = new_points
                    current_r = new_r
        
        # Cool down
        T *= cooling
        if T < 1e-6:
            # Reheat if stuck?
            T = 0.005
            # Or just stop?
            # Let's reheat to escape local minima.
            T = max(T, 0.001)

    return best_points, best_r

def construct():
    """Construct the optimal packing."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # 1. Generate initial hexagonal lattice
    points, init_r = _hexagonal_lattice()
    
    if points is None:
        # Fallback: simple grid
        points = np.zeros((N, 2))
        init_r = 0.0
    
    # 2. Run Simulated Annealing
    # We need to be careful with the time budget.
    # Annealing can be slow. Let's run it for a fixed amount of time or iterations.
    
    # Let's run annealing for most of the available time.
    anneal_deadline = deadline - 5.0
    if time.time() < anneal_deadline:
        opt_points, opt_r = _anneal(points, anneal_deadline)
        
        if opt_r > init_r:
            points = opt_points
            init_r = opt_r
    
    return points
# EVOLVE-BLOCK-END
