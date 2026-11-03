import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 300 non-overlapping circles inside a unit circle.
    Uses physics-based simulated annealing to maximize the minimum distance.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal lattice
    # Estimate spacing for N points in area pi
    # Area per point ~ pi / N
    # Side of hex cell ~ sqrt(area/sqrt(3)) * 2 ... roughly
    # Let's just generate a dense grid and pick the first N that fit, or just generate enough.
    
    def generate_hex_init():
        # Generate points on a hexagonal grid
        # We want roughly N points.
        # Radius of grid ~ sqrt(N)
        R = int(np.sqrt(N)) + 5
        points = []
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                x = i + 0.5 * (j % 2)
                y = j * np.sqrt(3) / 2.0
                points.append([x, y])
        points = np.array(points)
        
        # Normalize/Scale? No, we need to fit in unit circle.
        # The grid above is in "lattice units". We need to scale it to fit in radius 1.
        # Find the max radius of these points
        max_r = np.max(np.hypot(points[:, 0], points[:, 1]))
        # Scale so that max_r becomes slightly less than 1 to allow for optimization
        # Actually, it's better to start with a reasonable spacing.
        # Let's determine spacing such that N points fit.
        # Area of unit circle = pi. Area per point = pi/N.
        # Hex cell area = s^2 * sqrt(3)/2 * 2 = s^2 * sqrt(3) (if s is distance between centers? No, side length a, area a^2 sqrt(3)/2 * 2 for two triangles? 
        # Distance between centers d. Area per point = d^2 * sqrt(3)/2.
        # d = sqrt(2 * pi / (N * sqrt(3)))
        d_init = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        
        # Regenerate grid with this spacing
        points = []
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                x = (i + 0.5 * (j % 2)) * d_init
                y = j * d_init * np.sqrt(3) / 2.0
                points.append([x, y])
        points = np.array(points)
        
        # Filter points that are strictly inside the unit circle (with some margin)
        dists = np.hypot(points[:, 0], points[:, 1])
        mask = dists < 0.99
        points = points[mask]
        
        # If we have too many, pick the first N. If too few, add random points.
        if len(points) >= N:
            # Sort by distance to center to prefer central points? Or just random shuffle?
            # Random shuffle is better for diversity, but central packing is usually denser.
            # Let's just take the first N from the grid generation order (which is roughly concentric rings if we sorted, but here it's linear).
            # Better: Sort by radius and take inner ones? No, boundary points are important.
            # Just take a random subset of size N to avoid bias.
            idx = np.random.choice(len(points), N, replace=False)
            return points[idx]
        else:
            # If not enough, fill remaining with random points in the disk
            remaining = N - len(points)
            angles = np.random.uniform(0, 2*np.pi, remaining)
            radii = np.sqrt(np.random.uniform(0, 1, remaining))
            new_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            return np.vstack([points, new_points])

    # Initial positions
    positions = generate_hex_init()
    
    # Current best score (minimum distance)
    def calc_score(pos):
        # Min distance between any pair
        # For N=300, O(N^2) is 90k ops, fast enough.
        # Also check boundary violation
        # Score is min( min_pair_dist/2, min_boundary_dist )
        # But we want to maximize min_pair_dist while keeping boundary_dist > 0.
        # If boundary is violated, score is negative.
        
        # Boundary check
        r = np.hypot(pos[:, 0], pos[:, 1])
        boundary_dist = 1.0 - r
        min_boundary = np.min(boundary_dist)
        if min_boundary < 0:
            return -min_boundary # Penalize being outside
        
        # Pairwise distances
        # Using broadcasting for distance matrix might be memory heavy (300x300 is fine)
        # dist_matrix[i,j] = |pos_i - pos_j|
        # We only need min of upper triangle
        # To save time, we can use scipy.spatial.distance.pdist, but let's stick to numpy for simplicity unless it's slow.
        # 300x300 is 90,000 floats. Very small.
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # The actual radius of circles is min_pair_dist / 2
        # The constraint is min_pair_dist > 0 and boundary_dist > 0.
        # The score defined in task is: min(min_pair_dist/2, min_boundary_dist)
        # Wait, task says: r = min(min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|))
        # So we want to maximize this 'r'.
        
        return min(min_pair_dist / 2.0, min_boundary)

    current_score = calc_score(positions)
    best_positions = positions.copy()
    best_score = current_score
    
    # Parameters for Simulated Annealing
    # We want to maximize score.
    # Energy function: -score. Minimize Energy.
    
    T = 0.01  # Initial temperature (step size scale)
    cooling_rate = 0.999
    min_T = 1e-6
    
    # We will run for a fixed number of iterations or until time limit.
    # Let's estimate iterations per second.
    # One iteration:
    # 1. Propose move: pick random point, add random noise.
    # 2. Calc score: O(N^2).
    # Let's run as many iterations as possible.
    
    while time.time() < deadline:
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Propose a new position
        # Step size is proportional to T
        step = T * np.random.randn(2)
        new_pos = positions[i].copy()
        new_pos += step
        
        # Check if new position is wildly outside (optimization)
        # If it's way outside, reject immediately?
        # Or just calculate score.
        
        positions[i] = new_pos
        
        new_score = calc_score(positions)
        
        # Acceptance criterion (Metropolis)
        # We want to maximize score.
        # delta_E = new_score - current_score
        # If new_score > current_score, accept.
        # If new_score < current_score, accept with probability exp((new_score - current_score) / T)
        # Note: In standard SA for minimization, E is energy. Here we maximize score.
        # Let's define Energy = -Score.
        # dE = E_new - E_old = -new_score - (-current_score) = current_score - new_score.
        # If dE < 0 (score increased), accept.
        # If dE > 0 (score decreased), accept with prob exp(-dE/T).
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_positions = positions.copy()
        else:
            dE = current_score - new_score
            # If dE is large, probability is tiny.
            if dE < 1e-10:
                # Negligible difference, accept to explore
                current_score = new_score
            else:
                prob = np.exp(-dE / T)
                if np.random.rand() < prob:
                    current_score = new_score
                else:
                    # Reject
                    positions[i] = positions[i] - step # Revert
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            # If we hit min_T, we can break or keep going with small jitter
            # To ensure we use time, we might want to restart or keep jittering.
            # Let's keep jittering at min_T.

    return best_positions
# EVOLVE-BLOCK-END
