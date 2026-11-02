"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack 32 circles in the unit square maximizing sum of radii.
    
    Uses a continuous differentiable optimization with a soft contact graph
    derived from nearest-neighbor search. We optimize positions and radii
    simultaneously, using smooth penalty functions for constraint violations.
    """
    # Use a multi-start approach with different initial configurations
    best_result = None
    best_score = -np.inf
    
    # Initial configurations to try
    def grid_init(n_rows, n_cols, n_circles):
        """Generate a grid initialization."""
        positions = []
        radii = []
        total = n_rows * n_cols
        for i in range(n_rows):
            for j in range(n_cols):
                if len(positions) >= n_circles:
                    break
                x = (j + 0.5) / n_cols
                y = (i + 0.5) / n_rows
                positions.append([x, y])
                radii.append(0.5 / max(n_rows, n_cols) * 0.9)
        return np.array(positions[:n_circles]), np.array(radii[:n_circles])
    
    def random_init(n_circles, seed=42):
        """Generate a random initialization."""
        rng = np.random.RandomState(seed)
        positions = rng.uniform(0.1, 0.9, size=(n_circles, 2))
        radii = np.full(n_circles, 0.05)
        return positions, radii
    
    def hex_init(n_circles):
        """Generate a hexagonal packing initialization."""
        positions = []
        radii = []
        r = 0.08
        dx = 2 * r
        dy = r * np.sqrt(3)
        for row in range(10):
            for col in range(10):
                if len(positions) >= n_circles:
                    break
                x = dx * (col + 0.5 * (row % 2))
                y = dy * row
                # Center in unit square
                x = 0.5 + (x - 4.5 * dx) * 0.1
                y = 0.5 + (y - 4.5 * dy) * 0.1
                if 0.05 < x < 0.95 and 0.05 < y < 0.95:
                    positions.append([x, y])
                    radii.append(r * 0.9)
        # Fill remaining with random
        while len(positions) < n_circles:
            positions.append([0.5 + np.random.uniform(-0.3, 0.3), 
                              0.5 + np.random.uniform(-0.3, 0.3)])
            radii.append(0.05)
        return np.array(positions[:n_circles]), np.array(radii[:n_circles])
    
    def objective_and_constraints(x, n_circles, contact_threshold=0.05):
        """
        Compute objective (negative sum of radii) and constraint violations.
        
        x: array of shape (3*n,) containing [x1, y1, r1, x2, y2, r2, ...]
        """
        positions = x[0::3]
        radii = x[2::3]
        
        # Objective: minimize negative sum of radii
        obj = -np.sum(radii)
        
        # Constraints:
        # 1. Circles must be inside unit square: x_i >= r_i, x_i <= 1-r_i, etc.
        # 2. Circles must not overlap: dist(i,j) >= r_i + r_j
        
        # Compute all pairwise distances
        diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Sum of radii pairs
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Constraint violation for overlaps (only for pairs that are "close")
        # Use a soft threshold to focus on near-contact pairs
        overlap_violations = np.maximum(0, r_sums - dists)
        
        # Weight violations more heavily for closer pairs (contact graph)
        # Soft contact: weight = exp(-dists^2 / (2*sigma^2)) where sigma is small
        sigma = contact_threshold
        contact_weights = np.exp(-dists**2 / (2 * sigma**2))
        np.fill_diagonal(contact_weights, 0)
        
        weighted_violation = np.sum(contact_weights * overlap_violations**2)
        
        # Boundary violations
        boundary_viol = 0
        for i in range(n_circles):
            boundary_viol += max(0, radii[i] - positions[i, 0])**2
            boundary_viol += max(0, radii[i] - (1 - positions[i, 0]))**2
            boundary_viol += max(0, radii[i] - positions[i, 1])**2
            boundary_viol += max(0, radii[i] - (1 - positions[i, 1]))**2
        
        # Total penalty
        penalty = 100 * weighted_violation + 50 * boundary_viol
        
        return obj + penalty
    
    def solve_from_init(positions, radii, method='L-BFGS-B', maxiter=500):
        """Solve the optimization problem from a given initialization."""
        n_circles = len(radii)
        
        # Flatten: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * n_circles)
        for i in range(n_circles):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]
        
        # Bounds
        bounds = []
        for i in range(n_circles):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.001, 0.5))  # radius
        
        # Use L-BFGS-B with the combined objective
        result = minimize(
            objective_and_constraints,
            x0,
            args=(n_circles,),
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        return result
    
    # Try multiple initializations
    inits = []
    
    # Grid inits
    inits.append(grid_init(6, 6, N))
    inits.append(grid_init(5, 7, N))
    inits.append(grid_init(7, 5, N))
    
    # Random inits with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        inits.append(random_init(N, seed))
    
    # Hex init
    inits.append(hex_init(N))
    
    # Solve from each initialization
    for positions, radii in inits:
        try:
            result = solve_from_init(positions, radii, method='L-BFGS-B', maxiter=300)
            
            if result.success or result.fun < best_score:
                x_opt = result.x
                pos_opt = x_opt[0::3]
                rad_opt = x_opt[2::3]
                
                # Verify constraints
                valid = True
                for i in range(N):
                    if pos_opt[i, 0] < rad_opt[i] - 1e-6 or pos_opt[i, 0] > 1 - rad_opt[i] + 1e-6:
                        valid = False
                        break
                    if pos_opt[i, 1] < rad_opt[i] - 1e-6 or pos_opt[i, 1] > 1 - rad_opt[i] + 1e-6:
                        valid = False
                        break
                
                if valid:
                    # Check for overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt((pos_opt[i,0]-pos_opt[j,0])**2 + (pos_opt[i,1]-pos_opt[j,1])**2)
                            if dist < rad_opt[i] + rad_opt[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(rad_opt)
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([pos_opt, rad_opt])
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
