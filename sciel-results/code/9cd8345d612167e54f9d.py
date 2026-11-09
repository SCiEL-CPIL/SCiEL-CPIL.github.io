"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _min_radius(centers):
    """Compute the maximum radius for given centers."""
    # Distance to boundaries
    dist_to_boundary = np.min(np.stack([
        0.5 - centers[:, 0],
        centers[:, 0] + 0.5,
        0.5 - centers[:, 1],
        centers[:, 1] + 0.5
    ], axis=1), axis=1)
    
    # Pairwise distances
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pairwise = np.min(dist_matrix, axis=1)
    
    r = np.min(np.minimum(dist_to_boundary, min_pairwise / 2.0))
    return r

def _negative_radius(centers):
    """Negative radius for minimization."""
    return -_min_radius(centers)

def _generate_hexagonal_packing():
    """Generate a hexagonal lattice packing of N circles."""
    # Hexagonal lattice: rows offset by half spacing
    # We'll try different spacings and pick the best
    best_centers = None
    best_r = 0
    
    for spacing in np.linspace(0.03, 0.05, 50):
        # Hexagonal packing: row y = i * spacing * sqrt(3)/2
        # Even rows: x = j * spacing
        # Odd rows: x = (j + 0.5) * spacing
        centers = []
        i = 0
        while len(centers) < N:
            y = i * spacing * np.sqrt(3) / 2
            if i % 2 == 0:
                j = 0
                while len(centers) < N:
                    x = j * spacing
                    centers.append([x, y])
                    j += 1
            else:
                j = 0
                while len(centers) < N:
                    x = (j + 0.5) * spacing
                    centers.append([x, y])
                    j += 1
            i += 1
        
        centers = np.array(centers[:N])
        
        # Center the configuration
        center = centers.mean(axis=0)
        centers = centers - center
        
        # Scale to fit in the square
        max_coord = np.max(np.abs(centers))
        if max_coord > 0:
            scale = 0.48 / max_coord  # Leave small margin
            centers = centers * scale
        
        r = _min_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers
    
    return best_centers

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Start with hexagonal packing
    centers = _generate_hexagonal_packing()
    best_r = _min_radius(centers)
    best_centers = centers.copy()
    
    # Local optimization using L-BFGS-B
    # We'll optimize a smooth surrogate: sum of exp(-lambda * min_dist)
    # to encourage increasing the minimum distance
    
    def smooth_objective(x):
        c = x.reshape(-1, 2)
        # Keep centers in bounds
        c = np.clip(c, -0.499, 0.499)
        
        # Compute pairwise distances
        n = len(c)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Distance to boundaries
        dist_to_boundary = np.min(np.stack([
            0.5 - c[:, 0],
            c[:, 0] + 0.5,
            0.5 - c[:, 1],
            c[:, 1] + 0.5
        ], axis=1), axis=1)
        
        # Minimum distance for each circle
        min_pairwise = np.min(dist_matrix, axis=1)
        min_dist_each = np.minimum(dist_to_boundary, min_pairwise / 2.0)
        
        # Smooth minimum using log-sum-exp
        lambda_val = 50
        shifted = min_dist_each - np.max(min_dist_each)
        smooth_min = np.max(min_dist_each) + np.log(np.sum(np.exp(lambda_val * shifted))) / lambda_val
        
        return -smooth_min
    
    # Run optimization with time check
    result = None
    try:
        result = minimize(
            smooth_objective,
            centers.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
    except Exception:
        pass
    
    if result is not None and result.x is not None:
        candidate = np.clip(result.x.reshape(-1, 2), -0.499, 0.499)
        cand_r = _min_radius(candidate)
        if cand_r > best_r:
            best_r = cand_r
            best_centers = candidate
    
    # Check time and potentially do another round
    if time.time() < deadline - 20:
        # Try another optimization round from best position
        try:
            result2 = minimize(
                smooth_objective,
                best_centers.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            if result2 is not None and result2.x is not None:
                candidate2 = np.clip(result2.x.reshape(-1, 2), -0.499, 0.499)
                cand_r2 = _min_radius(candidate2)
                if cand_r2 > best_r:
                    best_r = cand_r2
                    best_centers = candidate2
        except Exception:
            pass
    
    # Final polish: try to push circles apart more
    if time.time() < deadline - 5:
        # Simple iterative push-apart
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            c = best_centers.copy()
            n = len(c)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Find closest pairs and push apart
            min_dist = np.min(dist_matrix)
            if min_dist > 0.075:  # Already good enough
                break
            
            # Get all close pairs
            close_pairs = np.argwhere(dist_matrix < 0.076)
            for i, j in close_pairs:
                if i < j:
                    direction = (c[i] - c[j]) / (dist_matrix[i, j] + 1e-12)
                    push = 0.0001
                    c[i] += direction * push
                    c[j] -= direction * push
            
            # Keep in bounds
            c = np.clip(c, -0.499, 0.499)
            
            new_r = _min_radius(c)
            if new_r > best_r:
                best_r = new_r
                best_centers = c
    
    return best_centers
# EVOLVE-BLOCK-END
