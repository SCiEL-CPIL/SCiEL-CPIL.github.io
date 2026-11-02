"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Pack 32 circles in the unit square maximizing sum of radii.
    
    Uses a continuous differentiable optimization with a soft contact graph
    derived from nearest-neighbor search. We optimize positions and radii
    simultaneously, using smooth penalty functions for constraint violations.
    """
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    best_result = None
    best_score = -np.inf
    
    # Initial configurations to try
    def grid_init(n_rows, n_cols, n_circles):
        """Generate a grid initialization."""
        positions = []
        radii = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(positions) >= n_circles:
                    break
                x = (j + 0.5) / n_cols
                y = (i + 0.5) / n_rows
                positions.append([x, y])
                radii.append(0.5 / max(n_rows, n_cols) * 0.95)
        while len(positions) < n_circles:
            positions.append([0.5, 0.5])
            radii.append(0.01)
        return np.array(positions[:n_circles]), np.array(radii[:n_circles])
    
    def random_init(n_circles, seed=42):
        """Generate a random initialization."""
        rng = np.random.RandomState(seed)
        positions = rng.uniform(0.1, 0.9, size=(n_circles, 2))
        radii = np.full(n_circles, 0.04)
        return positions, radii
    
    def hex_init(n_circles):
        """Generate a hexagonal packing initialization."""
        positions = []
        radii = []
        # Try to fit hexagonal pattern
        r = 0.06
        dx = 2 * r
        dy = r * np.sqrt(3)
        for row in range(8):
            for col in range(8):
                if len(positions) >= n_circles:
                    break
                x = dx * (col + 0.5 * (row % 2))
                y = dy * row
                # Scale and center to fit in unit square
                x = 0.5 + (x - 3.5 * dx) * 0.12
                y = 0.5 + (y - 3.5 * dy) * 0.12
                if 0.02 < x < 0.98 and 0.02 < y < 0.98:
                    positions.append([x, y])
                    radii.append(r * 0.95)
        # Fill remaining
        while len(positions) < n_circles:
            positions.append([0.5 + np.random.uniform(-0.3, 0.3), 
                              0.5 + np.random.uniform(-0.3, 0.3)])
            radii.append(0.03)
        return np.array(positions[:n_circles]), np.array(radii[:n_circles])
    
    def objective_and_constraints(x, n_circles, sigma=0.03):
        """
        Compute objective (negative sum of radii) and constraint violations.
        
        x: array of shape (3*n,) containing [x1, y1, r1, x2, y2, r2, ...]
        """
        positions = x[0::3]
        radii = x[2::3]
        
        # Objective: minimize negative sum of radii
        obj = -np.sum(radii)
        
        # Compute all pairwise distances
        diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2) + 1e-12)
        
        # Sum of radii pairs
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap violations
        overlap_violations = np.maximum(0, r_sums - dists)
        
        # Soft contact weights
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
        penalty = 500 * weighted_violation + 200 * boundary_viol
        
        return obj + penalty
    
    def solve_from_init(positions, radii, method='L-BFGS-B', maxiter=200):
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
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        
        return result
    
    def verify_and_score(pos_opt, rad_opt):
        """Verify constraints and return score if valid."""
        valid = True
        for i in range(N):
            if pos_opt[i, 0] < rad_opt[i] - 1e-7 or pos_opt[i, 0] > 1 - rad_opt[i] + 1e-7:
                valid = False
                break
            if pos_opt[i, 1] < rad_opt[i] - 1e-7 or pos_opt[i, 1] > 1 - rad_opt[i] + 1e-7:
                valid = False
                break
        
        if valid:
            # Check for overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((pos_opt[i,0]-pos_opt[j,0])**2 + (pos_opt[i,1]-pos_opt[j,1])**2)
                    if dist < rad_opt[i] + rad_opt[j] - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            return np.sum(rad_opt)
        return None
    
    # Try multiple initializations
    inits = []
    
    # Grid inits
    inits.append(grid_init(6, 6, N))
    inits.append(grid_init(5, 7, N))
    inits.append(grid_init(7, 5, N))
    inits.append(grid_init(4, 8, N))
    inits.append(grid_init(8, 4, N))
    
    # Random inits with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        inits.append(random_init(N, seed))
    
    # Hex init
    inits.append(hex_init(N))
    
    # Solve from each initialization
    for positions, radii in inits:
        if time.time() - start_time > timeout:
            break
        try:
            result = solve_from_init(positions, radii, method='L-BFGS-B', maxiter=150)
            
            x_opt = result.x
            pos_opt = x_opt[0::3]
            rad_opt = x_opt[2::3]
            
            score = verify_and_score(pos_opt, rad_opt)
            
            if score is not None and score > best_score:
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
