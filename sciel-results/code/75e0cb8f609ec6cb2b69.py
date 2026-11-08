import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    best_score = -np.inf
    best_config = None

    # Candidate widths for the rectangle (Perimeter = 4 => H = 2 - W)
    # Square is W=1, H=1.
    # We search around the square and some elongated shapes.
    widths = np.linspace(0.9, 1.1, 5)
    
    # Number of restarts per width
    n_restarts = 10

    for W in widths:
        if time.time() > deadline:
            break
        H = 2.0 - W

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Initialize centers
            # Use a grid initialization for better starting points
            # Approximate grid size
            n_x = int(np.sqrt(N * W / H))
            n_y = int(np.ceil(N / n_x))
            
            # Create grid points
            x_coords = np.linspace(0, W, n_x + 2)[1:-1]
            y_coords = np.linspace(0, H, n_y + 2)[1:-1]
            
            grid_x, grid_y = np.meshgrid(x_coords, y_coords)
            centers = np.column_stack((grid_x.flatten(), grid_y.flatten()))[:N]
            
            # Add small random perturbation to break symmetry
            np.random.seed(42 + restart)
            centers += np.random.normal(0, 0.05, centers.shape)
            
            # Clamp to bounds
            centers[:, 0] = np.clip(centers[:, 0], 0.01, W - 0.01)
            centers[:, 1] = np.clip(centers[:, 1], 0.01, H - 0.01)

            beta = 20.0  # Smoothness parameter for soft-min

            def objective(x_flat):
                x = x_flat.reshape(N, 2)
                total_soft_min = 0.0
                
                # Precompute pairwise distances? 
                # For N=21, computing norms in a loop is okay, but let's try to be efficient.
                
                for i in range(N):
                    # Distances to walls
                    d_walls = np.array([x[i,0], W - x[i,0], x[i,1], H - x[i,1]])
                    
                    # Distances to other circles
                    # Vectorized distance calculation
                    dists = np.linalg.norm(x[i] - x, axis=1) / 2.0
                    dists[i] = np.inf  # Exclude self
                    
                    # Combine
                    all_dists = np.concatenate([d_walls, dists])
                    
                    # Soft-min calculation
                    m = np.max(all_dists)
                    # Numerical stability
                    exp_terms = np.exp(-beta * (all_dists - m))
                    sum_exp = np.sum(exp_terms)
                    soft_min = m - (1.0 / beta) * np.log(sum_exp)
                    
                    total_soft_min += soft_min
                
                return -total_soft_min

            bounds = [(0, W)] * N + [(0, H)] * N

            try:
                res = minimize(objective, centers.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
                
                if res.success:
                    final_centers = res.x.reshape(N, 2)
                    
                    # Calculate exact radii
                    radii = np.zeros(N)
                    for i in range(N):
                        d_walls = np.array([final_centers[i,0], W - final_centers[i,0], final_centers[i,1], H - final_centers[i,1]])
                        dists = np.linalg.norm(final_centers[i] - final_centers, axis=1) / 2.0
                        dists[i] = np.inf
                        all_dists = np.concatenate([d_walls, dists])
                        radii[i] = np.min(all_dists)
                        
                    score = np.sum(radii)
                    if score > best_score:
                        best_score = score
                        best_config = np.column_stack((final_centers, radii))
                        
            except Exception:
                pass

    # Fallback if no valid result
    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
