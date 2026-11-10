"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place N=50 equal circles inside the unit circle to maximize the common radius.
    
    Strategy:
    1. Initialize centers using a well-distributed pattern (Fibonacci sphere projected to disk, or hexagonal).
    2. Use a grid-based collision map for fast neighbor lookups.
    3. Run a physics-inspired relaxation (force-directed) to maximize the minimum distance.
    4. Polish with scipy.optimize to fine-tune.
    
    The score is r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|)).
    We maximize this by maximizing the minimum pairwise distance and keeping points inside.
    """
    # Step 1: Initial placement using a hexagonal-like pattern or random with good spread
    # Use a spiral pattern for good initial spread
    def fibonacci_disk(n, radius=0.95):
        """Generate n points on a disk using a spiral pattern."""
        if n == 1:
            return np.array([[0.0, 0.0]])
        points = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            r = radius * np.sqrt((i + 0.5) / n)
            theta = i * golden_angle
            points.append([r * np.cos(theta), r * np.sin(theta)])
        return np.array(points)
    
    # Try multiple initial configurations and pick the best
    best_centers = None
    best_score = 0.0
    
    def compute_score(centers):
        """Compute the packing radius for given centers."""
        if len(centers) != N:
            return 0.0
        # Check all points are inside unit circle
        dists = np.linalg.norm(centers, axis=1)
        if np.any(dists >= 1.0):
            return 0.0
        # Compute minimum pairwise distance
        min_pair_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_pair_dist:
                    min_pair_dist = d
        r = min(min_pair_dist / 2.0, 1.0 - np.max(dists))
        return max(0.0, r)
    
    # Generate several initial configurations
    configs = []
    
    # Config 1: Fibonacci disk
    configs.append(fibonacci_disk(N, 0.95))
    
    # Config 2: Hexagonal lattice
    def hex_lattice(n):
        """Generate hexagonal lattice points inside unit disk."""
        points = []
        spacing = 0.3
        while len(points) < n:
            k = int(1.0 / spacing) + 2
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= (1.0 - spacing/2.0)**2:
                        points.append([x, y])
            if len(points) >= n:
                break
            spacing *= 0.98
        points = np.array(points[:n])
        return points
    
    configs.append(hex_lattice(N))
    
    # Config 3: Random with rejection
    np.random.seed(42)
    for _ in range(3):
        pts = []
        while len(pts) < N:
            angle = np.random.uniform(0, 2*np.pi)
            r = np.sqrt(np.random.uniform(0, 0.9))
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        configs.append(np.array(pts))
    
    # For each config, run relaxation and keep the best
    for init_centers in configs:
        centers = _relax(init_centers)
        score = compute_score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers
    
    if best_centers is None:
        best_centers = fibonacci_disk(N, 0.95)
    
    # Final polish with scipy
    best_centers = _polish(best_centers)
    
    # Ensure validity: clip to unit disk
    dists = np.linalg.norm(best_centers, axis=1)
    mask = dists > 0.999
    best_centers[mask] *= (0.999 / dists[mask])[:, None]
    
    return best_centers


def _relax(centers, iterations=2000, learning_rate=0.01):
    """
    Physics-inspired relaxation using grid-based neighbor lookup.
    Maximizes the minimum pairwise distance via repulsive forces.
    """
    centers = centers.copy()
    n = len(centers)
    
    # Grid parameters
    grid_size = 0.1  # cell size
    grid_range = 1.0  # grid covers [-1, 1] in both dims
    
    def build_grid(centers):
        """Build a dictionary mapping grid cell -> list of point indices."""
        grid = {}
        for i, (x, y) in enumerate(centers):
            gx = int((x + grid_range) / grid_size)
            gy = int((y + grid_range) / grid_size)
            key = (gx, gy)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)
        return grid
    
    def get_neighbors(grid, i, centers, search_radius=0.5):
        """Get indices of points within search_radius of point i using grid."""
        x, y = centers[i]
        gx = int((x + grid_range) / grid_size)
        gy = int((y + grid_range) / grid_size)
        r_cells = int(np.ceil(search_radius / grid_size))
        neighbors = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                key = (gx + dx, gy + dy)
                if key in grid:
                    for j in grid[key]:
                        if j != i:
                            neighbors.append(j)
        return neighbors
    
    for it in range(iterations):
        grid = build_grid(centers)
        forces = np.zeros_like(centers)
        
        # Compute repulsive forces between nearby points
        # Use a soft repulsion that's stronger when points are close
        min_dist = float('inf')
        for i in range(n):
            neighbors = get_neighbors(grid, i, centers, 0.5)
            for j in neighbors:
                if j <= i:
                    continue
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d < 1e-10:
                    # Jitter if coincident
                    diff = np.array([np.random.uniform(-0.01, 0.01), np.random.uniform(-0.01, 0.01)])
                    d = np.linalg.norm(diff)
                if d < min_dist:
                    min_dist = d
                # Repulsive force: stronger when closer
                # Use 1/d^2 for strong repulsion at close range
                if d < 0.5:
                    force_mag = 1.0 / (d * d + 1e-6)
                    force = force_mag * diff / d
                    forces[i] += force
                    forces[j] -= force
        
        # Add constraint force to keep points inside unit disk
        for i in range(n):
            d = np.linalg.norm(centers[i])
            if d > 0.9:
                # Push inward
                push = (d - 0.9) * 0.5
                forces[i] -= (centers[i] / d) * push
        
        # Update positions with adaptive learning rate
        lr = learning_rate * (1.0 - it / iterations) + 0.001
        centers += lr * forces
        
        # Project back to unit disk
        dists = np.linalg.norm(centers, axis=1)
        mask = dists > 0.999
        centers[mask] *= (0.999 / dists[mask])[:, None]
    
    return centers


def _polish(centers, max_iter=500):
    """
    Fine-tune using scipy.optimize to maximize the minimum distance.
    """
    def objective(x):
        """Negative of the packing radius (we minimize this)."""
        c = x.reshape(-1, 2)
        dists = np.linalg.norm(c, axis=1)
        if np.any(dists >= 1.0):
            return 1.0  # Large penalty
        # Compute min pairwise distance
        min_pair = float('inf')
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_pair:
                    min_pair = d
        r = min(min_pair / 2.0, 1.0 - np.max(dists))
        return -r
    
    x0 = centers.flatten()
    
    # Use Nelder-Mead for derivative-free optimization
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
    
    result = res.x.reshape(-1, 2)
    
    # Ensure validity
    dists = np.linalg.norm(result, axis=1)
    mask = dists > 0.999
    result[mask] *= (0.999 / dists[mask])[:, None]
    
    # Check if polish improved the score
    if compute_score(result) > compute_score(centers):
        return result
    return centers


# EVOLVE-BLOCK-END
