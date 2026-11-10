import numpy as np
import time
import scipy.optimize as opt

N = 50
DEADLINE = 140.0  # Safety margin from 150s

def _soft_min(z, beta):
    """
    Compute the soft minimum of a vector z.
    soft_min(z, beta) approx min(z) as beta -> infinity.
    """
    z_min = np.min(z)
    # Shift to avoid overflow in exp
    z_shifted = z - z_min
    # Sum of exp(-beta * z_shifted)
    # If z_shifted is 0, exp(0)=1. If z_shifted > 0, exp(-beta*z) < 1.
    terms = np.exp(-beta * z_shifted)
    log_sum = np.log(np.sum(terms))
    return z_min - (1.0 / beta) * log_sum

def _compute_radius(centers):
    """
    Compute the exact minimum radius for a set of centers.
    centers: (N, 2) array.
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check boundaries
    # dist to boundary
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    boundary_dists = np.minimum(dx, dy)
    min_boundary = np.min(boundary_dists)
    
    # Pairwise distances
    # Use broadcasting for (N, N) distance matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    return min(min_boundary, min_pairwise)

def _objective(x, beta):
    """
    Objective function to minimize: -soft_min(distances).
    x: flat array of length 2*N.
    """
    centers = x.reshape(N, 2)
    
    # Distances to boundary
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    boundary_dists = np.minimum(dx, dy)
    
    # Pairwise distances
    # For speed, we might calculate only upper triangle or use a more efficient method
    # But N=50 is small. 50x50 matrix is 2500 elements.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pairwise_dists = dists / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([pairwise_dists.flatten(), boundary_dists])
    
    # Soft min
    r_soft = _soft_min(all_dists, beta)
    
    return -r_soft

def _hexagonal_init(jitter=0.01):
    """
    Generate a jittered hexagonal lattice initialization.
    """
    # Hexagonal packing density is pi/sqrt(12) ~ 0.9069
    # Square packing is pi/4 ~ 0.785
    # We want to place 50 points.
    # Side length of hexagon packing to fit 50 points in 1x1 square?
    # Approx area per point = 1/50 = 0.02.
    # Hex cell area = sqrt(3)/2 * s^2.
    # s ~ sqrt(0.02 * 2 / sqrt(3)) ~ 0.15.
    
    # Let's generate a grid of hex points and pick the best 50.
    # Or just generate a dense hex grid and select.
    
    # Hex grid generation
    # Spacing dx = s, dy = s * sqrt(3)/2
    # Let's try s = 0.15
    s = 0.15
    dx = s
    dy = s * np.sqrt(3) / 2
    
    points = []
    # Cover the square [-0.5, 0.5]
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more than N points, select N closest to center
    if len(points) > N:
        dists_to_center = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        points = points[indices]
    elif len(points) < N:
        # Add random points to fill
        missing = N - len(points)
        random_points = np.random.uniform(-0.4, 0.4, size=(missing, 2))
        points = np.vstack([points, random_points])
        
    # Jitter
    points += np.random.normal(0, jitter, size=points.shape)
    
    # Project to square
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    return points

def _random_init():
    """
    Random initialization with some repulsion relaxation.
    """
    points = np.random.uniform(-0.45, 0.45, size=(N, 2))
    # Simple relaxation to avoid overlaps
    for _ in range(100):
        # Calculate forces
        forces = np.zeros_like(points)
        for i in range(N):
            for j in range(i + 1, N):
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < 0.1: # Repulsion radius
                    f = d / (dist + 1e-6) * (0.1 - dist)
                    forces[i] += f
                    forces[j] -= f
        points += 0.1 * forces
        points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
        points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    return points

def construct():
    start_time = time.time()
    
    best_radius = 0.0
    best_centers = None
    
    # Strategy: Multi-start local optimization
    # 1. Hexagonal init
    # 2. Random init
    # 3. Jittered Grid
    
    inits = []
    inits.append(_hexagonal_init(jitter=0.05))
    inits.append(_random_init())
    
    # Add a perturbed grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    grid += np.random.normal(0, 0.02, size=grid.shape)
    inits.append(grid)
    
    # Optimization settings
    beta_start = 20.0
    beta_end = 100.0
    
    for i_init, init_centers in enumerate(inits):
        if time.time() - start_time > DEADLINE:
            break
            
        x0 = init_centers.flatten()
        
        # Run local optimization with increasing beta
        # Start with lower beta for smoothness, increase for precision
        current_beta = beta_start
        x_current = x0
        
        # Use L-BFGS-B
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        # We can run a sequence of optimizations
        for step in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            beta = beta_start + (beta_end - beta_start) * (step / 4.0)
            
            res = opt.minimize(
                _objective, 
                x_current, 
                method='L-BFGS-B', 
                bounds=bounds, 
                args=(beta,),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            x_current = res.x
            
            # Calculate actual radius
            r = _compute_radius(x_current.reshape(N, 2))
            if r > best_radius:
                best_radius = r
                best_centers = x_current.reshape(N, 2)
                
        # If time allows, do a final polish with Nelder-Mead on the exact min-distance?
        # Or just high beta L-BFGS-B.
        if time.time() - start_time < DEADLINE - 5:
             res_final = opt.minimize(
                _objective, 
                x_current, 
                method='L-BFGS-B', 
                bounds=bounds, 
                args=(200.0,),
                options={'maxiter': 100, 'ftol': 1e-12}
            )
             x_final = res_final.x
             r_final = _compute_radius(x_final.reshape(N, 2))
             if r_final > best_radius:
                 best_radius = r_final
                 best_centers = x_final.reshape(N, 2)

    # Fallback if no valid center found (should not happen)
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers

if __name__ == "__main__":
    # Quick test
    c = construct()
    print("Radius:", _compute_radius(c))
