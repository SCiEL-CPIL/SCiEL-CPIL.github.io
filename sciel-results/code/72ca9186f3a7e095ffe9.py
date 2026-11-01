"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(positions):
    """Compute the radius score for given positions."""
    if positions.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    # Check bounds
    if np.any(positions[:, 0] < -0.5) or np.any(positions[:, 0] > 0.5) or \
       np.any(positions[:, 1] < -0.5) or np.any(positions[:, 1] > 0.5):
        return 0.0
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Distance to boundary
    dist_boundary = np.min([
        np.min(positions[:, 0] + 0.5),
        np.min(0.5 - positions[:, 0]),
        np.min(positions[:, 1] + 0.5),
        np.min(0.5 - positions[:, 1])
    ])
    return min(min_pair / 2.0, dist_boundary)

def _hex_init():
    """Create a hexagonal-like initial packing."""
    # Use a hexagonal lattice arrangement
    # For 200 circles, we can use rows with alternating offsets
    # Target radius ~ 0.0366, so spacing ~ 0.0732
    # Let's create a hex grid and select 200 points
    
    # Start with a grid of points in hex arrangement
    # Row spacing: r * sqrt(3)
    # Column spacing: 2 * r
    # Let's use a base spacing and adjust
    
    # Create a hex lattice with enough points, then select 200
    # Use spacing s such that we can fit ~200 points
    s = 0.075  # initial spacing
    
    # Generate hex lattice points
    points = []
    # Number of rows and columns needed
    # Area = 1, each hex cell area = s^2 * sqrt(3)/2
    # Need ~200 cells
    # Let's just generate a grid and filter
    
    # Use a rectangular grid with hex offset
    n_rows = 15
    n_cols = 14  # 15*14 = 210, enough
    
    for i in range(n_rows):
        for j in range(n_cols):
            x = -0.5 + s * (j + 0.5 * (i % 2))
            y = -0.5 + s * i * np.sqrt(3) / 2
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N, randomly select N
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Add random points to fill
        while len(points) < N:
            p = np.random.uniform(-0.5, 0.5, 2)
            points = np.vstack([points, p.reshape(1, 2)])
    
    return points[:N]

def _random_init():
    """Random initial positions."""
    return np.random.uniform(-0.5, 0.5, (N, 2))

def _optimize_from_init(init_pos, time_budget=10.0):
    """Optimize from a given initial position using gradient-based method."""
    start_time = time.time()
    
    # Use L-BFGS-B to minimize negative of the score
    # But score is non-smooth (min), so use a soft approximation
    
    def neg_score_soft(pos):
        """Negative of a soft version of the score for optimization."""
        pos = pos.reshape(N, 2)
        if np.any(pos[:, 0] < -0.5) or np.any(pos[:, 0] > 0.5) or \
           np.any(pos[:, 1] < -0.5) or np.any(pos[:, 1] > 0.5):
            return 1e10
        
        # Pairwise distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Soft min of pairwise distances
        # Use log-sum-exp trick for soft min
        # soft_min = -log(sum(exp(-d^2 / T))) * T
        # But this is for squared distances. Let's use a different approach.
        
        # Actually, let's just use the raw min but with a penalty approach
        # Minimize: -min(dists) + penalty for boundary violations
        
        min_pair = np.min(np.sqrt(dists_sq))
        
        # Boundary distances
        db = np.min([
            np.min(pos[:, 0] + 0.5),
            np.min(0.5 - pos[:, 0]),
            np.min(pos[:, 1] + 0.5),
            np.min(0.5 - pos[:, 1])
        ])
        
        score = min(min_pair / 2.0, db)
        return -score
    
    # Use a custom force-directed approach instead
    # This is more robust for this type of problem
    
    pos = init_pos.copy()
    best_pos = pos.copy()
    best_score = _score(pos)
    
    # Force-directed relaxation
    # Each pair of circles repels with force ~ 1/d^2
    # Also repel from boundaries
    
    lr = 0.01  # learning rate
    decay = 0.999
    
    for iteration in range(5000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute forces
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, 1e-10)
        dists = np.sqrt(dists_sq)
        
        # Repulsive force: F = k * d_hat / d^2
        # Vector form: F_ij = k * diff_ij / d_ij^3
        k = 0.001
        forces = np.zeros_like(pos)
        
        # Only consider pairs with distance < some threshold for efficiency
        # But N=200, so N^2 = 40000, which is fine
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 0.15:  # Only nearby pairs
                    f = k / (d ** 2 + 1e-10)
                    direction = diff[i, j] / (d + 1e-10)
                    forces[i] += f * direction
                    forces[j] -= f * direction
        
        # Boundary repulsion
        boundary_force = 0.01
        for i in range(N):
            # Left boundary
            if pos[i, 0] < -0.4:
                forces[i, 0] += boundary_force * (-0.5 - pos[i, 0])
            # Right boundary
            if pos[i, 0] > 0.4:
                forces[i, 0] -= boundary_force * (pos[i, 0] - 0.5)
            # Bottom boundary
            if pos[i, 1] < -0.4:
                forces[i, 1] += boundary_force * (-0.5 - pos[i, 1])
            # Top boundary
            if pos[i, 1] > 0.4:
                forces[i, 1] -= boundary_force * (pos[i, 1] - 0.5)
        
        # Update positions
        pos += lr * forces
        lr *= decay
        
        # Clip to valid region
        pos[:, 0] = np.clip(pos[:, 0], -0.5, 0.5)
        pos[:, 1] = np.clip(pos[:, 1], -0.5, 0.5)
        
        # Check score
        current_score = _score(pos)
        if current_score > best_score:
            best_score = current_score
            best_pos = pos.copy()
    
    return best_pos, best_score

def construct():
    """Construct the optimal packing of 200 equal circles in a unit square."""
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_pos = None
    best_score = 0.0
    
    # Try multiple initializations
    np.random.seed(42)
    
    # First, try hexagonal init
    hex_pos = _hex_init()
    opt_pos, opt_score = _optimize_from_init(hex_pos, time_budget=20.0)
    best_pos = opt_pos
    best_score = opt_score
    
    if time.time() > deadline:
        return best_pos
    
    # Try random inits with different seeds
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        rand_pos = _random_init()
        opt_pos, opt_score = _optimize_from_init(rand_pos, time_budget=10.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pos = opt_pos
    
    # Final polish: run more iterations on the best solution
    if time.time() < deadline - 5.0:
        final_pos, final_score = _optimize_from_init(best_pos, time_budget=20.0)
        if final_score > best_score:
            best_score = final_score
            best_pos = final_pos
    
    return best_pos
# EVOLVE-BLOCK-END
