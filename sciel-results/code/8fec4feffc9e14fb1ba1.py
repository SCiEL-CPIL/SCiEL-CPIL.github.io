"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Initialization: Ring-based representation
    # We place points on concentric rings.
    # Center point + rings.
    # We need to choose the number of points per ring.
    # Let's start with a heuristic distribution.
    
    def initialize_rings():
        points = []
        # Center
        points.append([0.0, 0.0])
        
        # Rings
        # Estimate ring radii and counts
        # Total points N=300.
        # Area of ring i is ~ 2*pi*r_i*dr. 
        # Let's try to distribute points such that density is roughly uniform.
        # n_k ~ r_k.
        # Sum n_k = 299.
        # Let's try a simple grid-like approximation or just iterative fitting.
        
        # Simple approach: assume ~ sqrt(N) rings? No, too many.
        # For N=300, maybe 15-20 rings?
        # Let's try to optimize the ring radii and counts.
        
        # Start with a guess:
        # Number of rings K.
        # n_k = k * c?
        # Sum k*c = N-1 => c = (N-1)/ (K*(K+1)/2)
        
        # Let's try K=15
        # c = 299 / 120 ~ 2.5.
        # n_1=3, n_2=5, ...
        
        # Let's just generate a hexagonal lattice first, then project to rings?
        # Or generate rings directly.
        
        # Direct ring generation:
        K = 16 # Number of rings
        counts = []
        remaining = N - 1
        # Distribute counts roughly proportional to radius (index)
        # n_k ~ k
        weights = np.arange(1, K+1)
        weights = weights / np.sum(weights)
        counts = np.round(weights * remaining).astype(int)
        # Fix sum
        diff = remaining - np.sum(counts)
        counts[-1] += diff # Adjust last ring
        
        # Initial radii: uniform spacing?
        # r_k ~ k / K
        radii = np.linspace(0.1, 0.95, K)
        
        for k in range(K):
            r = radii[k]
            n = counts[k]
            if n <= 0: continue
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            # Add small random offset to break symmetry initially?
            # Or keep symmetric for now.
            for a in angles:
                x = r * np.cos(a)
                y = r * np.sin(a)
                points.append([x, y])
                
        return np.array(points)

    points = initialize_rings()
    
    # Ensure validity (inside unit circle)
    radii = np.linalg.norm(points, axis=1)
    points[radii > 0.99] *= (0.99 / radii[radii > 0.99])[:, np.newaxis]

    def calc_min_radius(pts):
        if len(pts) < 2:
            return 0.0
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
        min_bound = 1.0 - np.max(np.linalg.norm(pts, axis=1))
        return min(min_pair / 2.0, min_bound)

    r_current = calc_min_radius(points)
    r_target = r_current * 1.01
    
    lr = 0.01
    momentum = 0.9
    v = np.zeros_like(points)
    best_score = r_current
    best_points = points.copy()
    
    # Main Optimization Loop
    # Use force-directed relaxation on the ring structure?
    # The directive suggests "ring-based representation... independent tuning of radial and tangential spacing".
    # This implies we might want to optimize the ring parameters (radii, angular offsets) directly,
    # or just use the force-directed method but initialized with rings.
    # Given the time constraint, force-directed on the points themselves is robust.
    # However, to exploit the "ring" nature, we could constrain points to rings?
    # That's harder. Let's stick to free movement but start from rings.
    # The "independent tuning" might just mean that the initial structure allows better local search.
    
    # Let's also add a small random perturbation to the ring init to help escape local minima.
    points += np.random.normal(0, 0.01, points.shape)
    # Re-check boundary
    radii = np.linalg.norm(points, axis=1)
    outside = radii > 0.99
    if np.any(outside):
        points[outside] *= (0.99 / radii[outside])[:, np.newaxis]
        
    r_current = calc_min_radius(points)
    r_target = r_current * 1.01
    best_score = r_current
    best_points = points.copy()
    v = np.zeros_like(points)

    while time.time() < deadline - 5.0:
        steps_in_batch = 20
        for _ in range(steps_in_batch):
            if time.time() > deadline:
                break
            
            dists = cdist(points, points)
            np.fill_diagonal(dists, np.inf)
            
            mask = dists < 2.0 * r_target
            dists_safe = np.where(dists > 1e-8, dists, 1.0)
            
            viol = np.where(mask, 2.0 * r_target - dists, 0.0)
            weights = viol / dists_safe
            weights[~mask] = 0.0
            np.fill_diagonal(weights, 0.0)
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
            
            radii = np.linalg.norm(points, axis=1)
            b_mask = radii > 1.0 - r_target
            b_viol = np.where(b_mask, radii - (1.0 - r_target), 0.0)
            b_forces = -b_viol[:, np.newaxis] * (points / np.maximum(radii, 1e-8)[:, np.newaxis])
            
            total_forces = forces + b_forces
            
            v = momentum * v + lr * total_forces
            points += v
            
            new_radii = np.linalg.norm(points, axis=1)
            max_r = 1.0 - r_target
            outside = new_radii > max_r
            if np.any(outside):
                points[outside] *= (max_r / new_radii[outside])[:, np.newaxis]
                v[outside] = 0.0

        current_score = calc_min_radius(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            r_current = current_score
            r_target = r_current * 1.005
            lr = max(0.001, lr * 0.98)
        else:
            r_target = r_current * 0.995
            if r_target < best_score * 0.95:
                points = best_points.copy()
                v = np.zeros_like(points)
                r_target = best_score * 1.001
                lr = 0.005
            else:
                pass

    # Final Polish
    if time.time() < deadline - 1.0:
        r_target = best_score * 1.0001
        lr = 0.0005
        momentum = 0.9
        v = np.zeros_like(points)
        
        for _ in range(50):
            if time.time() > deadline:
                break
            dists = cdist(points, points)
            np.fill_diagonal(dists, np.inf)
            mask = dists < 2.0 * r_target
            dists_safe = np.where(dists > 1e-8, dists, 1.0)
            viol = np.where(mask, 2.0 * r_target - dists, 0.0)
            weights = viol / dists_safe
            weights[~mask] = 0.0
            np.fill_diagonal(weights, 0.0)
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
            
            radii = np.linalg.norm(points, axis=1)
            b_mask = radii > 1.0 - r_target
            b_viol = np.where(b_mask, radii - (1.0 - r_target), 0.0)
            b_forces = -b_viol[:, np.newaxis] * (points / np.maximum(radii, 1e-8)[:, np.newaxis])
            
            total_forces = forces + b_forces
            v = momentum * v + lr * total_forces
            points += v
            
            new_radii = np.linalg.norm(points, axis=1)
            max_r = 1.0 - r_target
            outside = new_radii > max_r
            if np.any(outside):
                points[outside] *= (max_r / new_radii[outside])[:, np.newaxis]
                v[outside] = 0.0
                
        final_score = calc_min_radius(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()

    return best_points
# EVOLVE-BLOCK-END
