import numpy as np
import time

N = 200

def construct():
    # New approach: Simulated Annealing minimizing sum of inverse squared distances
    # This encourages all points to be as far apart as possible and away from boundaries.
    
    # Initialize with a jittered grid to avoid starting with overlapping circles which can be hard to escape
    # We use a grid slightly smaller than the optimal to ensure initial validity, then let SA expand them.
    # Or simply random initialization. Random is simpler and SA handles it well.
    # Let's use a jittered grid for better start.
    
    # Estimate a good grid size
    # sqrt(200) ~ 14.14. Let's try 14x14 grid.
    side = 14
    if side * side < N:
        side += 1
        
    # Create grid points
    x_coords = np.linspace(-0.45, 0.45, side)
    y_coords = np.linspace(-0.45, 0.45, side)
    
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Shuffle and take first N
    np.random.shuffle(grid_points)
    P = grid_points[:N].copy()
    
    # Add small jitter to break symmetries
    P += np.random.randn(N, 2) * 0.01
    
    # Ensure initial points are inside [-0.5, 0.5]
    P = np.clip(P, -0.49, 0.49)
    
    def energy(P):
        # Calculate sum of inverse squared distances between pairs
        # and inverse squared distances to boundaries
        # Use vectorization for speed
        # Pairwise distances
        # To avoid O(N^2) memory blowup for N=200 (40k pairs is fine), we can do it directly
        # But 200x200x2 array is small.
        
        # Pairwise
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        np.fill_diagonal(dists_sq, np.inf)
        
        # Avoid division by zero if points coincide (should be rare with jitter)
        # If dist is 0, energy is huge.
        pair_energy = np.sum(1.0 / np.maximum(dists_sq, 1e-10))
        
        # Boundary distances
        # Dist to left: 0.5 - x
        # Dist to right: 0.5 + x
        # Dist to bottom: 0.5 - y
        # Dist to top: 0.5 + y
        
        d_left = 0.5 - P[:, 0]
        d_right = 0.5 + P[:, 0]
        d_bottom = 0.5 - P[:, 1]
        d_top = 0.5 + P[:, 1]
        
        # Square distances
        d_left_sq = np.maximum(d_left**2, 1e-10)
        d_right_sq = np.maximum(d_right**2, 1e-10)
        d_bottom_sq = np.maximum(d_bottom**2, 1e-10)
        d_top_sq = np.maximum(d_top**2, 1e-10)
        
        bound_energy = np.sum(1.0 / d_left_sq) + np.sum(1.0 / d_right_sq) + \
                       np.sum(1.0 / d_bottom_sq) + np.sum(1.0 / d_top_sq)
        
        return pair_energy + bound_energy
    
    def min_radius(P):
        # Calculate the actual score for reporting/monitoring
        diffs = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        d_left = 0.5 - P[:, 0]
        d_right = 0.5 + P[:, 0]
        d_bottom = 0.5 - P[:, 1]
        d_top = 0.5 + P[:, 1]
        
        min_bound = np.min([np.min(d_left), np.min(d_right), np.min(d_bottom), np.min(d_top)])
        
        return min(min_pair, min_bound)

    # Simulated Annealing Parameters
    start_time = time.time()
    max_time = 110.0  # Leave 10s buffer
    
    T = 0.1  # Initial temperature
    cooling_rate = 0.9995
    min_T = 1e-6
    
    current_energy = energy(P)
    best_energy = current_energy
    best_P = P.copy()
    current_radius = min_radius(P)
    
    step_scale = 0.05  # Initial step size for random moves
    
    while time.time() - start_time < max_time:
        # Propose a move: pick a random point and move it by a Gaussian step
        idx = np.random.randint(N)
        delta = np.random.randn(2) * step_scale
        P_new = P.copy()
        P_new[idx] += delta
        
        # Keep inside boundaries? 
        # If we allow points to go outside, the boundary energy term will penalize them heavily.
        # However, if they go too far, 1/d^2 might underflow or behave weirdly.
        # Clipping to [-0.5, 0.5] is safer for the "inside" constraint, but the energy function
        # already penalizes going close to boundary.
        # Let's allow free movement but rely on energy. 
        # Actually, if a point goes to x=10, d_left=9.5, 1/9.5^2 is small. It won't push it back strongly.
        # We should ensure points stay roughly in the square. 
        # Let's clip the new position to [-0.6, 0.6] to prevent runaway, but the energy 
        # for boundary is defined for distance to [0.5]. If x > 0.5, d_left > 0, d_right < 0.
        # Wait, d_right = 0.5 + x. If x=0.6, d_right=1.1. d_left = -0.1.
        # The formula `0.5 - x` becomes negative. Squaring it makes it positive.
        # So the energy function as defined handles points outside the box by penalizing them
        # via the "wrong side" boundary distance becoming small (if we use abs) or just 
        # continuing the quadratic.
        # Actually, if x > 0.5, d_right = 0.5+x is large, d_left = 0.5-x is negative.
        # d_left^2 is positive. The penalty for being close to x=0.5 from the right is 
        # captured by d_left being small (negative).
        # So the energy function naturally repels points from the boundary from both sides.
        
        new_energy = energy(P_new)
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            P = P_new
            current_energy = new_energy
            
            # Check if we found a better radius
            # Calculating min_radius is expensive. Do it periodically.
            if np.random.rand() < 0.05: # 5% chance to check
                r = min_radius(P)
                if r > current_radius:
                    current_radius = r
                    best_P = P.copy()
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            T = min_T
            # Optionally reheat or stop if stuck. 
            # For now, just stop if T is very low and we haven't improved in a while.
            # But we have a time limit, so we can just keep going with small T (greedy).
        
        # Adapt step size?
        # If we accept many moves, maybe increase step? If we reject, decrease?
        # Simple heuristic: keep constant or decay slightly.
        # Let's decay step size slightly to converge.
        step_scale *= 0.9999

    # Final local optimization pass? 
    # SA with low T acts like greedy hill climbing. 
    # The result best_P is likely good.
    # Let's do a few iterations of greedy local search on best_P to polish.
    
    P = best_P
    for _ in range(500):
        improved = False
        # Try moving each point slightly towards increasing min radius?
        # This is hard to compute gradient for min radius.
        # Instead, use the energy function for a few greedy steps.
        for idx in range(N):
            P_orig = P[idx].copy()
            best_local_energy = current_energy
            best_local_pos = P_orig
            
            for _ in range(10):
                delta = np.random.randn(2) * 0.005
                P_test = P.copy()
                P_test[idx] = P_orig + delta
                e_test = energy(P_test)
                if e_test < best_local_energy:
                    best_local_energy = e_test
                    best_local_pos = P_test[idx]
            
            if not np.allclose(best_local_pos, P_orig):
                P[idx] = best_local_pos
                improved = True
        
        if not improved:
            break
        current_energy = energy(P)

    return P
