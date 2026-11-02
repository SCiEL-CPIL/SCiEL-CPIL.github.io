"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack 32 circles in the unit square maximizing sum of radii.
    
    Uses a continuous differentiable optimization with a soft contact graph
    derived from k-nearest neighbor search. We optimize positions and radii
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
        for i in range(n_rows):
            for j in range(n_cols):
                if len(positions) >= n_circles:
                    break
                x = (j + 0.5) / n_cols
                y = (i + 0.5) / n_rows
                positions.append([x, y])
                radii.append(0.5 / max(n_rows, n_cols) * 0.95)
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
        # Try to fit hexagonal lattice
        r = 0.08
        dx = 2 * r
        dy = r * np.sqrt(3)
        
        # Generate points in a larger area and filter
        for row in range(12):
            for col in range(12):
                if len(positions) >= n_circles:
                    break
                x = dx * (col + 0.5 * (row % 2))
                y = dy * row
                
                # Center and scale to fit in [0,1]
                # Approximate bounding box of this hex cluster
                max_x = dx * 11.5
                max_y = dy * 11
                x = 0.5 + (x - 0.5 * max_x) * (0.9 / max_x)
                y = 0.5 + (y - 0.5 * max_y) * (0.9 / max_y)
                
                if 0.05 < x < 0.95 and 0.05 < y < 0.95:
                    positions.append([x, y])
                    radii.append(r * 0.95)
        
        # Fill remaining with random if needed
        while len(positions) < n_circles:
            positions.append([0.5 + np.random.uniform(-0.3, 0.3), 
                              0.5 + np.random.uniform(-0.3, 0.3)])
            radii.append(0.05)
        return np.array(positions[:n_circles]), np.array(radii[:n_circles])
    
    def get_contact_weights(dists, k=6):
        """
        Compute a differentiable weight matrix for the contact graph.
        Pairs within the k-th nearest neighbor distance get high weight.
        """
        n = dists.shape[0]
        masked_dists = dists.copy()
        np.fill_diagonal(masked_dists, np.inf)
        
        # Get the k-th smallest distance for each row
        kth_indices = np.argsort(masked_dists, axis=1)[:, k-1]
        kth_dists = np.array([masked_dists[i, kth_indices[i]] for i in range(n)])
        
        # Average kth distance for global scale
        avg_kth = np.mean(kth_dists)
        if avg_kth < 1e-6:
            avg_kth = 1e-6
            
        # Soft weights
        weights = np.exp(-dists**2 / (2 * avg_kth**2))
        np.fill_diagonal(weights, 0)
        
        return weights

    def objective_and_constraints(x, n_circles, k=6):
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
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Sum of radii pairs
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Constraint violation for overlaps
        overlap_violations = np.maximum(0, r_sums - dists)
        
        # Dynamic contact weights
        contact_weights = get_contact_weights(dists, k=k)
        
        weighted_violation = np.sum(contact_weights * overlap_violations**2)
        
        # Boundary violations
        # Vectorized boundary check
        # x >= r, x <= 1-r
        # y >= r, y <= 1-r
        left_viol = np.maximum(0, radii - positions[:, 0])
        right_viol = np.maximum(0, radii - (1 - positions[:, 0]))
        bottom_viol = np.maximum(0, radii - positions[:, 1])
        top_viol = np.maximum(0, radii - (1 - positions[:, 1]))
        
        boundary_viol = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                        np.sum(bottom_viol**2) + np.sum(top_viol**2)
        
        # Total penalty
        # High penalty for overlaps to ensure feasibility
        penalty = 100 * weighted_violation + 50 * boundary_viol
        
        return obj + penalty
    
    def repair_solution(pos, rad):
        """
        Attempt to fix small constraint violations by moving circles.
        Returns (pos, rad, is_valid)
        """
        pos = pos.copy()
        rad = rad.copy()
        
        for _ in range(50):
            changed = False
            # Check boundaries
            for i in range(N):
                if pos[i, 0] < rad[i]:
                    pos[i, 0] = rad[i]
                    changed = True
                if pos[i, 0] > 1 - rad[i]:
                    pos[i, 0] = 1 - rad[i]
                    changed = True
                if pos[i, 1] < rad[i]:
                    pos[i, 1] = rad[i]
                    changed = True
                if pos[i, 1] > 1 - rad[i]:
                    pos[i, 1] = 1 - rad[i]
                    changed = True
            
            # Check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = rad[i] + rad[j]
                    if dist < min_dist:
                        # Push apart
                        if dist < 1e-9:
                            # Random push
                            angle = np.random.uniform(0, 2*np.pi)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            dist = 1
                        else:
                            dx /= dist
                            dy /= dist
                        
                        overlap = (min_dist - dist) / 2
                        # Move i back, j forward
                        pos[i, 0] -= dx * overlap
                        pos[i, 1] -= dy * overlap
                        pos[j, 0] += dx * overlap
                        pos[j, 1] += dy * overlap
                        changed = True
            
            if not changed:
                break
        
        # Final validation
        valid = True
        for i in range(N):
            if pos[i, 0] < rad[i] - 1e-9 or pos[i, 0] > 1 - rad[i] + 1e-9:
                valid = False
            if pos[i, 1] < rad[i] - 1e-9 or pos[i, 1] > 1 - rad[i] + 1e-9:
                valid = False
        
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                if dist < rad[i] + rad[j] - 1e-9:
                    valid = False
        
        return pos, rad, valid

    def solve_from_init(positions, radii, method='L-BFGS-B', maxiter=500, k=6):
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
            args=(n_circles, k),
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
    inits.append(grid_init(4, 8, N))
    inits.append(grid_init(8, 4, N))
    
    # Random inits with different seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
        inits.append(random_init(N, seed))
    
    # Hex init
    inits.append(hex_init(N))
    
    # Solve from each initialization
    for positions, radii in inits:
        try:
            result = solve_from_init(positions, radii, method='L-BFGS-B', maxiter=300, k=6)
            
            x_opt = result.x
            pos_opt = x_opt[0::3]
            rad_opt = x_opt[2::3]
            
            # Repair and validate
            pos_repaired, rad_repaired, valid = repair_solution(pos_opt, rad_opt)
            
            if valid:
                score = np.sum(rad_repaired)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([pos_repaired, rad_repaired])
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
