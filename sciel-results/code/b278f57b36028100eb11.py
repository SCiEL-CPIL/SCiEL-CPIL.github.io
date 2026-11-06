"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Best known packing radius for 50 circles in unit square
    # Boll, Donovan, Graham and Lubachevsky (2000): r ≈ 0.071377103865
    # We aim to find a packing with radius close to this.
    
    # Helper: compute the minimum distance constraint value
    # For circle i and j: distance >= 2r
    # For circle i and boundary: center must be at least r from each wall
    # In [-0.5, 0.5]^2, boundary distance for point (x,y) is min(0.5-|x|, 0.5-|y|)
    
    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        centers = np.asarray(centers).reshape(-1, 2)
        n = centers.shape[0]
        
        # Circle-circle distances
        min_pair_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                min_pair_dist = min(min_pair_dist, d)
        
        # Boundary distances
        min_boundary = np.inf
        for i in range(n):
            x, y = centers[i]
            bd = min(0.5 - abs(x), 0.5 - abs(y))
            min_boundary = min(min_boundary, bd)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    # Objective: minimize negative radius (maximize radius)
    # Use a smooth approximation: soft-min of the constraint values
    # Constraints: d_ij/2 for all i<j, and boundary distances
    # We want to maximize the minimum of these.
    
    def soft_min_logsumexp(values, beta=50.0):
        """Approximate min via -logsumexp(-beta * values) / beta"""
        values = np.asarray(values)
        # Clip to avoid overflow
        v = -beta * values
        v_max = np.max(v)
        if v_max > 700:
            v = v - v_max
        s = np.log(np.sum(np.exp(v))) + v_max
        return -s / beta
    
    def objective(x):
        """Negative soft-min radius. x is flattened centers."""
        if time.time() > deadline:
            return 1e10
        centers = x.reshape(-1, 2)
        
        # Check for non-finite or out of bounds
        if not np.all(np.isfinite(centers)):
            return 1e10
        if np.any(centers[:, 0] < -0.5) or np.any(centers[:, 0] > 0.5):
            return 1e10
        if np.any(centers[:, 1] < -0.5) or np.any(centers[:, 1] > 0.5):
            return 1e10
        
        # Compute all constraint values
        n = centers.shape[0]
        constraints = []
        
        # Circle-circle
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                constraints.append(d / 2.0)
        
        # Boundary
        for i in range(n):
            x, y = centers[i]
            bd = min(0.5 - abs(x), 0.5 - abs(y))
            constraints.append(bd)
        
        constraints = np.array(constraints)
        
        # Use soft-min with moderate beta for smoothness
        beta = 30.0
        r_soft = soft_min_logsumexp(constraints, beta)
        
        return -r_soft
    
    # Initial guess: hexagonal-like lattice with jitter
    def hexagonal_init(rng):
        """Generate a hexagonal packing initial guess with jitter."""
        # For 50 circles, try a hexagonal arrangement
        # Hexagonal packing: rows offset by half spacing
        # Try different row/column configurations
        
        best_centers = None
        best_r = -np.inf
        
        # Try various grid-like hexagonal arrangements
        for rows in range(5, 10):
            cols = int(np.ceil(N / rows))
            if cols * rows < N:
                cols += 1
            
            # Hexagonal spacing
            # For a hexagonal lattice with spacing s:
            # Horizontal distance = s
            # Vertical distance = s * sqrt(3)/2
            # Offset every other row by s/2
            
            # We need to fit in [-0.5, 0.5]^2
            # Try to find good s
            
            # Generate base hexagonal positions
            centers = []
            for row in range(rows):
                for col in range(cols):
                    if len(centers) >= N:
                        break
                    # Hexagonal offset
                    x = (col - (cols - 1) / 2.0) * 1.0  # unit spacing initially
                    y = (row - (rows - 1) / 2.0) * 0.866  # sqrt(3)/2
                    if row % 2 == 1:
                        x += 0.5
                    centers.append([x, y])
            
            centers = np.array(centers[:N])
            
            # Scale to fit in square with some margin
            # Find extent
            x_min, x_max = centers[:, 0].min(), centers[:, 0].max()
            y_min, y_max = centers[:, 1].min(), centers[:, 1].max()
            
            # Scale to fit within [-0.4, 0.4] initially (leave room for radius)
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range > 0 and y_range > 0:
                scale = min(0.8 / x_range, 0.8 / y_range)
                centers = centers * scale
                # Center
                centers[:, 0] -= (x_min + x_max) / 2.0 * scale
                centers[:, 1] -= (y_min + y_max) / 2.0 * scale
            
            # Add small jitter
            centers = centers + rng.normal(0, 0.01, size=centers.shape)
            
            r = compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
        
        return best_centers
    
    # Also try a simple grid initialization
    def grid_init(rng):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N].copy()
        centers = centers + rng.normal(0, 0.005, size=centers.shape)
        return centers
    
    # Run multiple restarts with different initializations
    best_centers = None
    best_radius = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several initializations
    inits = []
    
    # Hexagonal with different jitter levels
    for jitter in [0.005, 0.01, 0.02, 0.03]:
        rng_sub = np.random.default_rng(42 + int(jitter * 1000))
        # Modify hexagonal_init to use custom jitter
        def hex_init_custom(rng_sub, jitter_level):
            best_c = None
            best_r = -np.inf
            for rows in range(5, 10):
                cols = int(np.ceil(N / rows))
                if cols * rows < N:
                    cols += 1
                
                centers = []
                for row in range(rows):
                    for col in range(cols):
                        if len(centers) >= N:
                            break
                        x = (col - (cols - 1) / 2.0) * 1.0
                        y = (row - (rows - 1) / 2.0) * 0.866
                        if row % 2 == 1:
                            x += 0.5
                        centers.append([x, y])
                
                centers = np.array(centers[:N])
                
                x_min, x_max = centers[:, 0].min(), centers[:, 0].max()
                y_min, y_max = centers[:, 1].min(), centers[:, 1].max()
                
                x_range = x_max - x_min
                y_range = y_max - y_min
                if x_range > 0 and y_range > 0:
                    scale = min(0.8 / x_range, 0.8 / y_range)
                    centers = centers * scale
                    centers[:, 0] -= (x_min + x_max) / 2.0 * scale
                    centers[:, 1] -= (y_min + y_max) / 2.0 * scale
                
                centers = centers + rng_sub.normal(0, jitter_level, size=centers.shape)
                
                r = compute_radius(centers)
                if r > best_r:
                    best_r = r
                    best_c = centers.copy()
            return best_c
        
        c = hex_init_custom(rng_sub, jitter)
        if c is not None:
            inits.append(c)
    
    # Grid initialization
    inits.append(grid_init(rng))
    
    # Random initialization
    for _ in range(2):
        c = rng.uniform(-0.4, 0.4, size=(N, 2))
        inits.append(c)
    
    # Optimize each initialization
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        x0 = init_centers.flatten()
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={
                    'maxiter': 200,
                    'maxfun': 500,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < 1e9:
                centers_opt = result.x.reshape(-1, 2)
                r_opt = compute_radius(centers_opt)
                
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = centers_opt.copy()
        except Exception:
            pass
    
    # Final polish: use the best result and do a more refined optimization
    if best_centers is not None and time.time() < deadline - 10.0:
        # Try increasing beta for sharper approximation
        def objective_sharp(x):
            if time.time() > deadline:
                return 1e10
            centers = x.reshape(-1, 2)
            if not np.all(np.isfinite(centers)):
                return 1e10
            if np.any(centers[:, 0] < -0.5) or np.any(centers[:, 0] > 0.5):
                return 1e10
            if np.any(centers[:, 1] < -0.5) or np.any(centers[:, 1] > 0.5):
                return 1e10
            
            n = centers.shape[0]
            constraints = []
            
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    constraints.append(d / 2.0)
            
            for i in range(n):
                x, y = centers[i]
                bd = min(0.5 - abs(x), 0.5 - abs(y))
                constraints.append(bd)
            
            constraints = np.array(constraints)
            
            beta = 100.0
            r_soft = soft_min_logsumexp(constraints, beta)
            
            return -r_soft
        
        try:
            x0 = best_centers.flatten()
            result = minimize(
                objective_sharp,
                x0,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'maxfun': 1000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            
            if result.fun < 1e9:
                centers_opt = result.x.reshape(-1, 2)
                r_opt = compute_radius(centers_opt)
                
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = centers_opt.copy()
        except Exception:
            pass
    
    # Ensure validity: clip to square and handle edge cases
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Clip to ensure centers are within [-0.5, 0.5]
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Final validity check
    if not np.all(np.isfinite(best_centers)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_centers = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
