"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist


def _check_valid(circles, eps=1e-7):
    """Check if circles are valid: inside unit square, disjoint interiors."""
    n = len(circles)
    if n != N:
        return False
    for i in range(n):
        x, y, r = circles[i]
        if x - r < -eps or x + r > 1 + eps:
            return False
        if y - r < -eps or y + r > 1 + eps:
            return False
        if r < -eps:
            return False
    # Check pairwise disjointness
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if dist < ri + rj - eps:
                return False
    return True


def _calculate_energy_vec(circles, lambda_pen=100.0):
    """
    Vectorized energy calculation.
    Energy = -Sum(Radii) + Penalty(Overlaps + Boundary Violations)
    """
    n = len(circles)
    # Maximize sum of radii -> Minimize negative sum
    e_sum_r = -np.sum(circles[:, 2])
    
    # Boundary penalties
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # dist to boundaries
    dist_left = x - r
    dist_right = 1 - x - r
    dist_bottom = y - r
    dist_top = 1 - y - r
    dist_neg_r = -r # if r < 0
    
    # Penalty is sum of squared violations
    pen_left = np.where(dist_left < 0, dist_left**2, 0)
    pen_right = np.where(dist_right < 0, dist_right**2, 0)
    pen_bottom = np.where(dist_bottom < 0, dist_bottom**2, 0)
    pen_top = np.where(dist_top < 0, dist_top**2, 0)
    pen_r = np.where(r < 0, r**2, 0)
    
    e_bound = np.sum(pen_left + pen_right + pen_bottom + pen_top + pen_r)
    
    # Overlap penalties
    # Compute pairwise distances
    coords = circles[:, :2]
    dists = cdist(coords, coords, 'euclidean')
    radii_sum = circles[:, 2][:, None] + circles[:, 2][None, :]
    
    # Overlap is max(0, radii_sum - dists)
    # We only care about i < j, but cdist gives full matrix.
    # We can sum the upper triangle.
    overlaps = np.maximum(radii_sum - dists, 0)
    # Zero out diagonal
    np.fill_diagonal(overlaps, 0)
    # Sum only upper triangle to avoid double counting
    idx = np.triu_indices(n, k=1)
    e_overlap = np.sum(overlaps[idx]**2)
    
    return e_sum_r + lambda_pen * (e_bound + e_overlap)


def _simulated_annealing(initial_circles, num_iterations=2000, t_start=0.5, t_end=0.001, lambda_pen=100.0):
    """
    Perform Simulated Annealing to optimize the circle packing.
    """
    n = N
    current_circles = initial_circles.copy()
    current_energy = _calculate_energy_vec(current_circles, lambda_pen)
    best_circles = current_circles.copy()
    best_energy = current_energy
    
    # Cooling schedule
    cooling_factor = (t_end / t_start) ** (1.0 / num_iterations)
    
    for i in range(num_iterations):
        t = t_start * (cooling_factor ** i)
        
        # Create a perturbed state
        new_circles = current_circles.copy()
        
        # Choose a random circle to perturb
        idx = np.random.randint(n)
        
        # Perturb position and radius
        # Scale of perturbation
        scale_pos = 0.05
        scale_r = 0.02
        
        new_circles[idx, 0] += np.random.randn() * scale_pos
        new_circles[idx, 1] += np.random.randn() * scale_pos
        new_circles[idx, 2] += np.random.randn() * scale_r
        
        # Clip to valid ranges to keep search space bounded
        new_circles[idx, 0] = np.clip(new_circles[idx, 0], 0, 1)
        new_circles[idx, 1] = np.clip(new_circles[idx, 1], 0, 1)
        new_circles[idx, 2] = np.clip(new_circles[idx, 2], 0, 0.5)
        
        new_energy = _calculate_energy_vec(new_circles, lambda_pen)
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta_e / t)
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / t):
            current_circles = new_circles
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_circles = current_circles.copy()
                
    return best_circles


def construct():
    """Pack 32 circles with disjoint interiors inside [0,1]^2, maximizing sum of radii."""
    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.01
            cells.append([x, y, r])
    while len(cells) < N:
        cells.append([0.5, 0.5, 0.01])
    initial_circles = np.array(cells[:N])
    
    best_circles = None
    best_value = -np.inf
    
    # Run multiple restarts with different seeds for randomness
    num_restarts = 5
    
    for k in range(num_restarts):
        # Perturb initial state for each restart
        if k == 0:
            init = initial_circles.copy()
        else:
            init = initial_circles + np.random.randn(N, 3) * 0.05
            init[:, 0] = np.clip(init[:, 0], 0, 1)
            init[:, 1] = np.clip(init[:, 1], 0, 1)
            init[:, 2] = np.clip(init[:, 2], 0.01, 0.4)
            
        # Run SA
        result = _simulated_annealing(init, num_iterations=2000, t_start=0.5, t_end=0.001)
        
        # Check validity and value
        if _check_valid(result):
            value = np.sum(result[:, 2])
            if value > best_value:
                best_value = value
                best_circles = result.copy()
                
    if best_circles is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
