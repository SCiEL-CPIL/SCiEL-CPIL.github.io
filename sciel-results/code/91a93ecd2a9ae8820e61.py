import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50

def compute_radius(positions):
    """Compute the maximum common radius for given circle centers."""
    # Boundary constraint
    min_bound = 0.5 - np.max(np.abs(positions))
    
    # Pairwise distance constraint
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    return min(min_bound, min_dist / 2.0)

def generate_hexagonal_initial():
    """Generate a hexagonal packing initialization."""
    positions = []
    sqrt3 = np.sqrt(3.0)
    
    # Try different row counts and spacings
    best_pos = None
    best_r = 0
    
    for n_rows in range(5, 10):
        # Calculate spacing to fit in square
        # For hex packing: horizontal spacing = s, vertical spacing = s * sqrt(3)/2
        # Total height = (n_rows - 1) * s * sqrt(3)/2 + s (for the circles themselves)
        # We want this to fit in 1.0 (from -0.5 to 0.5)
        
        # Let's use a different approach: place circles in a grid pattern
        # with alternating rows offset
        
        # Estimate spacing
        # If we have n_rows rows, and roughly N/n_rows columns
        n_cols = int(np.ceil(N / n_rows))
        
        # Spacing in x direction
        # We want n_cols circles to fit in width 1.0
        # With offset rows, the effective width is slightly different
        s_x = 1.0 / (n_cols + 0.5)  # Add half for offset
        s_y = s_x * sqrt3 / 2.0
        
        # Check if it fits in height
        total_height = (n_rows - 1) * s_y + s_x  # Approximate
        if total_height > 1.0:
            # Scale down
            scale = 1.0 / total_height
            s_x *= scale
            s_y *= scale
        
        # Generate positions
        pos_list = []
        for row in range(n_rows):
            y = -0.5 + s_x/2 + row * s_y
            if y > 0.5 - s_x/2:
                continue
            n_in_row = n_cols if row % 2 == 0 else n_cols - 1
            if row % 2 == 1:
                x_start = -0.5 + s_x
            else:
                x_start = -0.5 + s_x/2
            
            for col in range(n_in_row):
                x = x_start + col * s_x
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pos_list.append([x, y])
                if len(pos_list) >= N:
                    break
            if len(pos_list) >= N:
                break
        
        if len(pos_list) >= N:
            pos_arr = np.array(pos_list[:N])
            r = compute_radius(pos_arr)
            if r > best_r:
                best_r = r
                best_pos = pos_arr.copy()
    
    if best_pos is not None:
        return best_pos
    
    # Fallback: simple grid
    n_side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(n_side):
        for j in range(n_side):
            if len(positions) >= N:
                break
            x = -0.5 + (j + 0.5) * (1.0 / n_side)
            y = -0.5 + (i + 0.5) * (1.0 / n_side)
            positions.append([x, y])
    
    return np.array(positions[:N])

def polish(positions, time_limit=5.0):
    """Local optimization using scipy minimize to polish the configuration."""
    start_time = time.time()
    positions = positions.copy()
    current_r = compute_radius(positions)
    
    # Define objective function: maximize radius (minimize negative radius)
    def objective(x):
        if time.time() - start_time > time_limit:
            return 0
        pos = x.reshape(N, 2)
        if np.any(np.abs(pos) > 0.5):
            return 0
        return -compute_radius(pos)
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Try multiple random restarts within the polish phase
    best_pos = positions.copy()
    best_r = current_r
    
    n_restarts = 3
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Add small random perturbation
        if restart > 0:
            noise_scale = 0.01 / restart
            x0 = positions.flatten() + np.random.uniform(-noise_scale, noise_scale, 2*N)
            x0 = np.clip(x0, -0.5, 0.5)
        else:
            x0 = positions.flatten()
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 500, 'maxls': 50})
            if result.success or result.fun < -best_r:
                new_pos = result.x.reshape(N, 2)
                if np.all(np.abs(new_pos) <= 0.5):
                    new_r = compute_radius(new_pos)
                    if new_r > best_r:
                        best_r = new_r
                        best_pos = new_pos.copy()
        except Exception:
            pass
    
    return best_pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Step 1: Generate initial hexagonal packing
    positions = generate_hexagonal_initial()
    current_r = compute_radius(positions)
    best_pos = positions.copy()
    best_r = current_r
    
    # Step 2: Polish with local optimization
    positions = polish(positions, time_limit=20.0)
    current_r = compute_radius(positions)
    if current_r > best_r:
        best_r = current_r
        best_pos = positions.copy()
    
    # Step 3: If time remains, try differential evolution for global search
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        # Use a smaller DE search with tight bounds around current best
        def de_objective(x):
            if time.time() > deadline:
                return 0
            pos = x.reshape(N, 2)
            if np.any(np.abs(pos) > 0.5):
                return 0
            return -compute_radius(pos)
        
        # Bounds: stay within the square
        de_bounds = [(-0.5, 0.5)] * (2 * N)
        
        # Use a population size that's manageable
        popsize = 20
        
        try:
            de_result = differential_evolution(
                de_objective, de_bounds,
                popsize=popsize,
                maxiter=50,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            
            if de_result.fun < -best_r:
                new_pos = de_result.x.reshape(N, 2)
                if np.all(np.abs(new_pos) <= 0.5):
                    new_r = compute_radius(new_pos)
                    if new_r > best_r:
                        best_r = new_r
                        best_pos = new_pos.copy()
        except Exception:
            pass
    
    # Step 4: Final polish
    remaining_time = deadline - time.time()
    if remaining_time > 3.0:
        best_pos = polish(best_pos, time_limit=min(remaining_time - 2.0, 15.0))
        final_r = compute_radius(best_pos)
        if final_r > best_r:
            best_r = final_r
    
    return best_pos
