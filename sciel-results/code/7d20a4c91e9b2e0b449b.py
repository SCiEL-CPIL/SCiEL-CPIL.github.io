import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    max_time = 110.0  # Leave buffer for overhead
    
    # Initial setup
    # Use a heuristic initial layout to avoid bad local minima
    # A grid is okay to start, but we will optimize from there.
    # Better: Random initialization with some spread.
    
    np.random.seed(42) # For reproducibility in this context, though random is fine
    # Let's try a few restarts or a single long run.
    
    def run_optimization():
        # Initialize positions
        # Spread them out initially
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        for i in range(N):
            r = i // side
            c = i % side
            # Add jitter
            x = (c + 0.5 + 0.2 * (np.random.rand() - 0.5)) / side
            y = (r + 0.5 + 0.2 * (np.random.rand() - 0.5)) / side
            xs.append(x)
            ys.append(y)
        
        pos = np.array(list(zip(xs, ys)))
        radii = np.full(N, 0.01)
        
        # Optimization parameters
        step_size = 0.01
        growth_rate = 0.005
        iterations = 20000
        tol = 1e-6
        
        best_sum_r = np.sum(radii)
        best_pos = pos.copy()
        best_radii = radii.copy()
        
        for it in range(iterations):
            if time.time() - start_time > max_time:
                break
                
            # 1. Try to grow radii
            radii += growth_rate
            
            # 2. Optimize positions to reduce overlaps
            # We use a simple gradient descent step on the overlap penalty
            # Penalty P = sum_{i<j} max(0, r_i + r_j - dist_ij)^2 + boundary penalties
            
            for _ in range(20): # Inner loop for position optimization
                forces = np.zeros_like(pos)
                
                # Pairwise interactions
                for i in range(N):
                    for j in range(i+1, N):
                        d_vec = pos[i] - pos[j]
                        d = np.linalg.norm(d_vec)
                        min_d = radii[i] + radii[j]
                        if d < min_d:
                            if d < 1e-6:
                                # Random push
                                d_vec = np.random.randn(2)
                                d = np.linalg.norm(d_vec)
                            overlap = min_d - d
                            # Force is proportional to overlap
                            f = overlap * d_vec / d
                            forces[i] += f
                            forces[j] -= f
                
                # Boundary constraints
                for i in range(N):
                    x, y = pos[i]
                    r = radii[i]
                    # Left
                    if x < r:
                        forces[i][0] += (r - x)
                    # Right
                    if x > 1 - r:
                        forces[i][0] -= (x - (1 - r))
                    # Bottom
                    if y < r:
                        forces[i][1] += (r - y)
                    # Top
                    if y > 1 - r:
                        forces[i][1] -= (y - (1 - r))
                
                # Update positions
                pos += step_size * forces
                # Clamp to valid region (hard constraint for stability)
                pos[:, 0] = np.clip(pos[:, 0], radii, 1 - radii)
                pos[:, 1] = np.clip(pos[:, 1], radii, 1 - radii)
            
            # Check validity
            valid = True
            for i in range(N):
                x, y = pos[i]
                r = radii[i]
                if x < r - 1e-8 or x > 1 - r + 1e-8 or y < r - 1e-8 or y > 1 - r + 1e-8:
                    valid = False
                    break
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < radii[i] + radii[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                current_sum = np.sum(radii)
                if current_sum > best_sum_r:
                    best_sum_r = current_sum
                    best_pos = pos.copy()
                    best_radii = radii.copy()
            else:
                # Revert radius growth
                radii -= growth_rate
                # Optionally shrink growth rate
                growth_rate *= 0.95
                
        return best_pos, best_radii
    
    pos, radii = run_optimization()
    
    # Construct output array
    result = np.column_stack((pos, radii))
    
    # Final safety check and clamp
    # Ensure no negative radii or out of bounds
    for i in range(N):
        x, y, r = result[i]
        if r < 0: r = 0
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        result[i] = [x, y, r]
        
    return result
