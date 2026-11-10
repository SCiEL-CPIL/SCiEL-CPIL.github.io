"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START

def _score(centers):
    """Compute the packing radius for a set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    # Wall distance
    wall_dist = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    # Pairwise distances
    tree = cKDTree(centers)
    # Query all points to get pairwise distances
    # For N=100, we can just compute all pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    return min(r_wall, r_pair)


def _random_initial():
    """Generate a random initial configuration."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))


def _improve(centers, max_iters=5000, time_limit=None):
    """
    Local improvement via iterative relaxation.
    At each step, find the most constrained sphere and try to move it.
    """
    centers = centers.copy()
    best_score = _score(centers)
    
    # Precompute some helpers
    def get_constrained_spheres(centers):
        """Return list of (sphere_idx, constraint_type, other_idx_or_-1, distance) sorted by tightness."""
        tree = cKDTree(centers)
        constraints = []
        for i in range(N):
            # Wall constraints
            for k in range(3):
                d = 0.5 - abs(centers[i, k])
                constraints.append((d, i, 'wall', k, -1))
            # Pair constraints
            # Only check neighbors within some radius to save time
            # Actually for N=100, just check all
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j]) / 2.0
                    constraints.append((d, i, 'pair', j, j))
        constraints.sort(key=lambda x: x[0])
        return constraints
    
    start_time = time.time()
    
    for it in range(max_iters):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
            
        # Find the most constrained sphere
        # We'll use a simpler approach: pick a random sphere that has a tight constraint
        # and try to move it in a direction that increases its minimum distance
        
        # Compute current score
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists, axis=1) / 2.0
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        r_current = np.min(np.minimum(min_pair, wall_dists))
        
        # Find spheres that are at the minimum
        tight_spheres = np.where(np.abs(np.minimum(min_pair, wall_dists) - r_current) < 1e-10)[0]
        if len(tight_spheres) == 0:
            tight_spheres = np.random.choice(N, 1, replace=False)
        
        i = tight_spheres[np.random.randint(len(tight_spheres))]
        
        # Try to move sphere i in a random direction
        # The direction should be away from its closest constraints
        # Compute gradient-like direction: sum of repulsion vectors from close neighbors
        
        # Find close neighbors
        close_neighbors = []
        for j in range(N):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2 * r_current * 1.5:  # within 1.5x current radius
                    close_neighbors.append((j, d))
        
        # Also consider wall constraints
        wall_dirs = []
        for k in range(3):
            if centers[i, k] > 0:
                wall_dirs.append((-1.0) * np.eye(3)[k])  # push toward -x
            else:
                wall_dirs.append(np.eye(3)[k])  # push toward +x
        
        # Compute repulsion direction
        repulsion = np.zeros(3)
        for j, d in close_neighbors:
            if d > 1e-12:
                dir_vec = (centers[i] - centers[j]) / d
                repulsion += dir_vec / max(d, 1e-6)
        
        # Add wall repulsion
        for k in range(3):
            if centers[i, k] > 0.5 - r_current * 1.5:
                repulsion[k] -= 1.0
            elif centers[i, k] < -0.5 + r_current * 1.5:
                repulsion[k] += 1.0
        
        norm = np.linalg.norm(repulsion)
        if norm < 1e-12:
            continue
        repulsion /= norm
        
        # Try several step sizes
        best_new_centers = None
        best_new_score = r_current
        
        for step in [0.01, 0.005, 0.002, 0.001, 0.02, 0.05]:
            new_centers = centers.copy()
            new_centers[i] += repulsion * step
            # Clamp to cube
            new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
            
            # Quick score check
            diff = new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists, axis=1) / 2.0
            wall_dists = 0.5 - np.max(np.abs(new_centers), axis=1)
            r_new = np.min(np.minimum(min_pair, wall_dists))
            
            if r_new > best_new_score + 1e-12:
                best_new_score = r_new
                best_new_centers = new_centers
        
        if best_new_centers is not None:
            centers = best_new_centers
            if best_new_score > best_score:
                best_score = best_new_score
    
    return centers, best_score


def _anneal(centers, time_limit=None):
    """
    Simulated annealing-style improvement.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    start_time = time.time()
    T = 0.01  # initial temperature
    cooling = 0.999
    
    def get_score(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists, axis=1) / 2.0
        wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
        return np.min(np.minimum(min_pair, wall_dists))
    
    while (time_limit is None or (time.time() - start_time) < time_limit) and T > 1e-6:
        # Pick a random sphere to move
        i = np.random.randint(N)
        
        # Move in a random direction
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        # Step size proportional to temperature
        step = T * np.random.uniform(0.5, 2.0)
        
        new_centers = centers.copy()
        new_centers[i] += direction * step
        new_centers[i] = np.clip(new_centers[i], -0.5, 0.5)
        
        score_new = get_score(new_centers)
        score_old = get_score(centers)
        
        if score_new >= score_old or np.random.rand() < np.exp((score_new - score_old) / T):
            centers = new_centers
            if score_new > best_score:
                best_score = score_new
                best_centers = centers.copy()
        
        T *= cooling
    
    return best_centers, best_score


def construct():
    """
    Construct centers for N equal spheres in a unit cube.
    Uses multiple restarts with local improvement and simulated annealing.
    """
    start_time = time.time()
    time_budget = 140.0  # Leave 10s safety margin
    
    best_centers = None
    best_score = 0.0
    
    # Start with a grid-based initial configuration
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial = grid[:N].copy()
    
    # Add small random perturbations to break symmetry
    np.random.seed(42)
    initial += np.random.randn(N, 3) * 0.01
    
    # Improve the initial configuration
    improved, score = _improve(initial, max_iters=3000, time_limit=20.0)
    if score > best_score:
        best_score = score
        best_centers = improved.copy()
    
    # Try simulated annealing from the improved configuration
    if time.time() - start_time < time_budget:
        annealed, score = _anneal(improved, time_limit=min(40.0, time_budget - (time.time() - start_time)))
        if score > best_score:
            best_score = score
            best_centers = annealed.copy()
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() - start_time > time_budget:
            break
        
        random_init = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Quick local improvement
        improved, score = _improve(random_init, max_iters=1000, time_limit=10.0)
        if score > best_score:
            best_score = score
            best_centers = improved.copy()
        
        # Annealing
        if time.time() - start_time < time_budget:
            annealed, score = _anneal(improved, time_limit=min(20.0, time_budget - (time.time() - start_time)))
            if score > best_score:
                best_score = score
                best_centers = annealed.copy()
    
    # Final polish
    if time.time() - start_time < time_budget:
        polished, score = _improve(best_centers, max_iters=2000, time_limit=min(30.0, time_budget - (time.time() - start_time)))
        if score > best_score:
            best_score = score
            best_centers = polished.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
