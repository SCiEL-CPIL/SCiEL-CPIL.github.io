"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random

def _compute_radius(pts):
    """Compute the maximum radius for the given points."""
    n = len(pts)
    if n == 0:
        return 0.0
    
    # Boundary constraint
    max_abs = np.max(np.abs(pts))
    r_boundary = 0.5 - max_abs
    
    # Pairwise distance constraint
    # Efficient calculation of minimum pairwise distance
    # For N=100, O(N^2) is fine
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
    r_pair = min_dist / 2.0
    
    return min(r_boundary, r_pair)

def construct():
    n = N
    random.seed(42)
    np.random.seed(42)
    
    # Initialize on a grid
    m = int(np.ceil(n ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:n].copy()
    
    # Shuffle to break symmetry slightly if needed, but grid is usually a good start
    # pts = np.random.permutation(pts)
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Simulated Annealing
    # Parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.99  # Cooling rate
    max_iters = 20000
    move_scale = 0.05
    
    current_pts = best_pts.copy()
    current_r = best_r
    
    for i in range(max_iters):
        # Generate a new point by perturbing one random point
        idx = random.randint(0, n - 1)
        new_pts = current_pts.copy()
        
        # Random step
        step = np.random.normal(0, move_scale, size=3)
        new_pts[idx] += step
        
        # Clamp to box
        new_pts[idx] = np.clip(new_pts[idx], -0.5, 0.5)
        
        # Calculate new radius
        new_r = _compute_radius(new_pts)
        
        # Acceptance criterion
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = new_pts.copy()
        else:
            # Accept worse solution with probability exp((new_r - current_r) / T)
            # Since we are maximizing, delta = new_r - current_r is negative
            # We use a penalty factor to make the exponent reasonable
            # Standard SA uses energy. Here "energy" is -radius.
            # delta_E = -new_r - (-current_r) = current_r - new_r > 0
            delta_E = current_r - new_r
            if delta_E < 0:
                # This shouldn't happen if new_r < current_r, but just in case
                current_pts = new_pts
                current_r = new_r
            else:
                # Scale delta_E to be comparable to T
                # A difference of 0.01 in radius is significant
                scaled_delta = delta_E / 0.01 
                prob = np.exp(-scaled_delta / T)
                if random.random() < prob:
                    current_pts = new_pts
                    current_r = new_r
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Final Polishing: Deterministic Local Search
    # Iterate through all points and try small moves in all directions
    # to see if we can increase the minimum radius.
    # This helps fix "jitter" and settle into a tighter configuration.
    
    polish_pts = best_pts.copy()
    polish_r = best_r
    
    # Simple hill climbing with small steps
    step_sizes = [0.01, 0.005, 0.001]
    for size in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                orig = polish_pts[i].copy()
                # Try 6 directions + 6 diagonal directions + random?
                # Just 6 axis-aligned is often enough for final polish
                for d in range(3):
                    for sign in [-1, 1]:
                        new_p = orig.copy()
                        new_p[d] += sign * size
                        # Clamp
                        new_p[d] = np.clip(new_p[d], -0.5, 0.5)
                        
                        # Check if this move improves the *local* minimum
                        # To be fast, we only check distances involving point i
                        # and the boundary for point i.
                        # However, moving i might affect the global min if i was part of the min pair.
                        # For simplicity and robustness, compute full radius.
                        # N=100 is small enough for this in a few hundred iterations.
                        test_pts = polish_pts.copy()
                        test_pts[i] = new_p
                        new_r = _compute_radius(test_pts)
                        
                        if new_r > polish_r + 1e-12:
                            polish_pts[i] = new_p
                            polish_r = new_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
    if polish_r > best_r:
        best_r = polish_r
        best_pts = polish_pts
        
    # Ensure points are strictly inside
    eps = 1e-10
    best_pts = np.clip(best_pts, -0.5 + eps, 0.5 - eps)
    
    return best_pts
# EVOLVE-BLOCK-END
