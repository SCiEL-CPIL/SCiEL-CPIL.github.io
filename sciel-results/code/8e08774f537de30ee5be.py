import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def compute_radius(centers):
        # centers: (N, 2)
        # Compute min distance between pairs
        # Compute min distance to boundary
        # Return min of the two (divided by 2 for pair dist)
        
        # Vectorized distance calculation for pairs is O(N^2) which is fine for N=50
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=-1)
        # min_pair_dist = np.min(dists[np.triu_indices_from(dists, k=1)])
        
        # Boundary distance
        # x_min_dist = 0.5 - centers[:, 0]
        # x_max_dist = 0.5 + centers[:, 0]
        # y_min_dist = 0.5 - centers[:, 1]
        # y_max_dist = 0.5 + centers[:, 1]
        # min_boundary_dist = np.min(np.concatenate([x_min_dist, x_max_dist, y_min_dist, y_max_dist]))
        
        # r = min(min_pair_dist / 2, min_boundary_dist)
        
        # For speed in optimization, we might want a smooth approximation, 
        # but SLSQP handles non-smooth constraints poorly if not careful.
        # Let's stick to the exact definition for the final score, 
        # but for optimization, we use constraints.
        pass

    def objective_and_constraints(x):
        # x: [c_1x, c_1y, ..., c_Nx, c_Ny, r]
        centers = x[:2*N].reshape(N, 2)
        r = x[-1]
        
        # Objective: minimize -r
        return -r

    def make_constraints():
        cons = []
        # Boundary constraints:
        # c_i_x >= -0.5 + r  => c_i_x - r >= -0.5
        # c_i_x <= 0.5 - r   => -c_i_x - r <= -0.5  => c_i_x + r <= 0.5
        # c_i_y >= -0.5 + r
        # c_i_y <= 0.5 - r
        
        # Pair constraints:
        # ||c_i - c_j|| >= 2r
        # ||c_i - c_j||^2 - 4r^2 >= 0
        
        # SLSQP expects constraints in the form:
        # type 'ineq': f(x) >= 0
        # type 'eq': f(x) == 0
        
        # Let's generate them as functions.
        
        def bound_x_min(x):
            centers = x[:2*N].reshape(N, 2)
            r = x[-1]
            return centers[:, 0] - r + 0.5 # >= 0
            
        def bound_x_max(x):
            centers = x[:2*N].reshape(N, 2)
            r = x[-1]
            return 0.5 - centers[:, 0] - r # >= 0
            
        def bound_y_min(x):
            centers = x[:2*N].reshape(N, 2)
            r = x[-1]
            return centers[:, 1] - r + 0.5 # >= 0
            
        def bound_y_max(x):
            centers = x[:2*N].reshape(N, 2)
            r = x[-1]
            return 0.5 - centers[:, 1] - r # >= 0

        cons.append({'type': 'ineq', 'fun': bound_x_min})
        cons.append({'type': 'ineq', 'fun': bound_x_max})
        cons.append({'type': 'ineq', 'fun': bound_y_min})
        cons.append({'type': 'ineq', 'fun': bound_y_max})
        
        # Pair constraints
        indices = np.triu_indices_from(np.ones((N, N)), k=1)
        for i, j in zip(*indices):
            def pair_constraint(x, i=i, j=j):
                centers = x[:2*N].reshape(N, 2)
                r = x[-1]
                diff = centers[i] - centers[j]
                dist_sq = np.dot(diff, diff)
                return dist_sq - 4*r*r # >= 0
            cons.append({'type': 'ineq', 'fun': pair_constraint})
            
        return cons

    constraints = make_constraints()
    
    # Initial guesses
    best_r = 0
    best_centers = None
    
    # 1. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Initial r for grid
    # Grid spacing is 1/m
    # Dist to boundary is 0.5 - (m-0.5)/m = 0.5/m
    # Pair dist is 1/m
    # r_grid = min(1/m / 2, 0.5/m) = 0.5/m
    r_init = 0.5 / m
    
    x0 = np.concatenate([grid.flatten(), [r_init]])
    
    # 2. Perturbed Grids
    np.random.seed(42)
    
    restarts = [x0]
    for _ in range(10):
        noise = np.random.normal(0, 0.01, grid.shape)
        perturbed = grid + noise
        # Clip to square
        perturbed = np.clip(perturbed, -0.5 + r_init, 0.5 - r_init)
        x0_pert = np.concatenate([perturbed.flatten(), [r_init]])
        restarts.append(x0_pert)
        
    # 3. Random
    for _ in range(5):
        rand_centers = np.random.uniform(-0.5 + r_init, 0.5 - r_init, (N, 2))
        x0_rand = np.concatenate([rand_centers.flatten(), [r_init]])
        restarts.append(x0_rand)

    for x0 in restarts:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective_and_constraints, x0, method='SLSQP', constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
            if res.success:
                r_found = res.x[-1]
                centers_found = res.x[:2*N].reshape(N, 2)
                # Verify validity and compute actual score
                # The optimizer might have found a point where constraints are satisfied but r is not maximized locally?
                # SLSQP maximizes r subject to constraints. So r_found should be the local max.
                
                # Let's do a quick check to ensure no constraint violation due to numerical error
                # And compute the true radius
                true_r = compute_radius(centers_found)
                
                if true_r > best_r:
                    best_r = true_r
                    best_centers = centers_found
        except Exception as e:
            pass

    # If no optimization succeeded, return grid
    if best_centers is None:
        return grid
        
    return best_centers

def compute_radius(centers):
    # centers: (N, 2)
    # Compute min distance between pairs
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Mask diagonal
    dists[np.fill_diagonal(dists, np.inf)] = np.inf
    min_pair_dist = np.min(dists)
    
    # Boundary distance
    x_min_dist = 0.5 - centers[:, 0]
    x_max_dist = 0.5 + centers[:, 0]
    y_min_dist = 0.5 - centers[:, 1]
    y_max_dist = 0.5 + centers[:, 1]
    min_boundary_dist = np.min(np.concatenate([x_min_dist, x_max_dist, y_min_dist, y_max_dist]))
    
    r = min(min_pair_dist / 2, min_boundary_dist)
    return r
