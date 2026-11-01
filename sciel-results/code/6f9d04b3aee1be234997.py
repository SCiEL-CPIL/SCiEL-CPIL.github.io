"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # 1. Initialization: Hexagonal Lattice
    # We start with a hexagonal lattice which is a good approximation of the optimal packing.
    # We generate a grid large enough and clip to the unit circle.
    
    # Estimate spacing. Area of unit circle is pi. Area per circle is pi*r^2.
    # N * pi * r^2 <= pi  => r <= 1/sqrt(N).
    # Hex lattice density is sqrt(3)/2. 
    # Spacing s ~ r * 2 / (sqrt(3)/2) ? No, distance between centers is 2r.
    # In hex lattice, area per point is (sqrt(3)/2) * s^2.
    # N * (sqrt(3)/2) * s^2 ~ pi  => s ~ sqrt(2*pi / (N*sqrt(3))).
    
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    
    # Generate points
    points = []
    # Determine range
    k = int(2.0 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x**2 + y**2 <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    # If we have too many, randomly select N.
    # If too few, reduce spacing (increase s) and retry? 
    # The hex lattice usually provides more than N points for this N if s is chosen correctly.
    # Let's ensure we have at least N points.
    if len(points) < N:
        # Fallback: Spiral or random
        # Use Vogel spiral as fallback
        n = N
        golden_angle = np.pi * (3 - np.sqrt(5))
        points = np.zeros((n, 2))
        for i in range(n):
            r = np.sqrt(i / n)
            theta = i * golden_angle
            points[i] = [r * np.cos(theta), r * np.sin(theta)]
    else:
        # Select N points closest to the center to maintain density
        radii = np.linalg.norm(points, axis=1)
        indices = np.argsort(radii)[:N]
        points = points[indices]
        
    # 2. Simulated Annealing Optimization
    # Objective: Maximize r = min(min_pair_dist/2, min_boundary_dist)
    
    def calc_score(pts):
        # Calculate pairwise distances
        # dists[i, j] = |pts[i] - pts[j]|
        # Use broadcasting
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal (self distance)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists)
        
        # Boundary distance
        norms = np.linalg.norm(pts, axis=1)
        min_boundary_dist = np.min(1.0 - norms)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    current_pts = points.copy()
    current_score = calc_score(current_pts)
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    # SA Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    max_iters = 50000
    
    # Random state for reproducibility in local tests, but SA is stochastic
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if T < T_min:
            break
            
        # Pick a random circle
        idx = rng.integers(0, N)
        
        # Perturbation
        # Scale perturbation by T and current radius scale
        # A good scale is roughly the current estimated radius
        scale = max(best_score, 1e-4)
        step_size = T * scale * 10
        
        new_pts = current_pts.copy()
        # Gaussian noise
        noise = rng.normal(0, step_size, size=2)
        new_pts[idx] += noise
        
        # Ensure the new point is not wildly outside the container
        # If it goes too far, the score will be 0 or negative, which is bad.
        # We can clamp it to the unit circle to help convergence, 
        # or just let the score penalty handle it.
        # Clamping is safer for stability.
        norm_new = np.linalg.norm(new_pts[idx])
        if norm_new > 0.99:
            new_pts[idx] = new_pts[idx] / norm_new * 0.99
        
        new_score = calc_score(new_pts)
        
        # Acceptance criterion
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Metropolis criterion
            # Delta is negative
            delta = new_score - current_score
            # Avoid log(0) or underflow
            if delta == 0:
                accept_prob = 1.0
            else:
                # T is small, delta is small negative.
                # exp(delta/T)
                try:
                    accept_prob = np.exp(delta / T)
                except OverflowError:
                    accept_prob = 0.0
                    
            if rng.random() < accept_prob:
                current_pts = new_pts
                current_score = new_score
        
        # Cool down
        T *= alpha
        
    return best_pts
# EVOLVE-BLOCK-END
