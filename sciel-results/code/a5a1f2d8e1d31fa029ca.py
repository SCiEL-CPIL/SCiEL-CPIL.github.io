import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

def construct():
    """
    Constructs the centers of 50 equal circles in a unit square [-0.5, 0.5]^2
    to maximize the minimum distance between circles and to the boundary.
    Uses smooth potential energy minimization with L-BFGS-B.
    """
    
    def objective_and_grad(flat_x):
        """
        Objective: Sum of inverse squared distances between pairs + inverse squared distances to walls.
        This smooth function is minimized to maximize the minimum distance.
        """
        x = flat_x.reshape(N, 2)
        
        # Pairwise distances
        # Calculate all pairwise distances
        dists = cdist(x, x)
        np.fill_diagonal(dists, 1.0) # Avoid div by zero
        
        # Sum of 1/d^2 for i < j
        # We can use the full matrix and divide by 2, or just sum upper triangle
        # Using full matrix is easier for gradient calculation via vectorization
        # But cdist is O(N^2). 
        # Let's compute pairwise distances explicitly for gradient
        
        # Objective value
        obj = 0.0
        grad = np.zeros_like(x)
        
        # Pairwise term
        # To avoid O(N^2) Python loop, we can vectorize
        # d_ij^2 = (x_i - x_j)^2 + (y_i - y_j)^2
        # We want sum_{i<j} 1/d_ij^2
        # Gradient wrt x_i: sum_{j != i} -2 * (x_i - x_j) / d_ij^4
        
        # Let's compute the matrix of squared distances
        # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # d_sq = np.sum(diff**2, axis=2)
        # This creates a (N, N) array.
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero (diagonal)
        np.fill_diagonal(d_sq, 1.0)
        
        # 1/d^2 matrix
        inv_d_sq = 1.0 / d_sq
        np.fill_diagonal(inv_d_sq, 0.0)
        
        # Objective is sum of upper triangle of inv_d_sq
        obj += 0.5 * np.sum(inv_d_sq)
        
        # Gradient of sum_{i<j} 1/d_ij^2 wrt x_i
        # d/dx_i (1/d_ij^2) = -2 * (x_i - x_j) / d_ij^4
        # Sum over j != i
        # Let A = inv_d_sq^2 (which is 1/d^4)
        inv_d_4 = inv_d_sq**2
        np.fill_diagonal(inv_d_4, 0.0)
        
        # grad_i = -2 * sum_j (x_i - x_j) * inv_d_4[i, j]
        # sum_j (x_i - x_j) * w_ij = sum_j x_i w_ij - sum_j x_j w_ij
        # = x_i * sum_j w_ij - sum_j x_j w_ij
        
        w_sum = np.sum(inv_d_4, axis=1)
        w_dot_x = np.sum(inv_d_4[:, :, np.newaxis] * x[np.newaxis, :, :], axis=1)
        
        grad += -2.0 * (x * w_sum[:, np.newaxis] - w_dot_x)
        
        # Wall term
        # sum_i [ 1/(0.5-x_i)^2 + 1/(0.5+x_i)^2 + 1/(0.5-y_i)^2 + 1/(0.5+y_i)^2 ]
        
        dx_pos = 0.5 - x[:, 0]
        dx_neg = 0.5 + x[:, 0]
        dy_pos = 0.5 - x[:, 1]
        dy_neg = 0.5 + x[:, 1]
        
        # Prevent division by zero if points are exactly on boundary (shouldn't happen with bounds)
        # But for safety in gradient calculation if it gets close
        eps = 1e-8
        dx_pos = np.maximum(dx_pos, eps)
        dx_neg = np.maximum(dx_neg, eps)
        dy_pos = np.maximum(dy_pos, eps)
        dy_neg = np.maximum(dy_neg, eps)
        
        wall_obj = np.sum(1.0/dx_pos**2 + 1.0/dx_neg**2 + 1.0/dy_pos**2 + 1.0/dy_neg**2)
        obj += wall_obj
        
        # Gradients for wall term
        # d/dx_i (1/(0.5-x)^2) = 2/(0.5-x)^3
        # d/dx_i (1/(0.5+x)^2) = -2/(0.5+x)^3
        
        grad[:, 0] += 2.0 * (1.0/dx_pos**3 - 1.0/dx_neg**3)
        grad[:, 1] += 2.0 * (1.0/dy_pos**3 - 1.0/dy_neg**3)
        
        return obj, grad

    def score(x):
        """
        Calculate the actual radius r = min(min_pair_dist/2, min_wall_dist)
        """
        x = x.reshape(N, 2)
        dists = cdist(x, x)
        np.fill_diagonal(dists, np.inf)
        min_pair = dists.min() / 2.0
        
        # Distance to walls
        min_wall = np.min(np.min([
            0.5 + x[:, 0], # left
            0.5 - x[:, 0], # right
            0.5 + x[:, 1], # down
            0.5 - x[:, 1]  # up
        ]))
        
        return min(min_pair, min_wall)

    def run_optimization(init_points):
        """
        Run L-BFGS-B from a given initialization.
        """
        flat_init = init_points.flatten()
        
        # Bounds: points must be in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        res = minimize(
            objective_and_grad,
            flat_init,
            method='L-BFGS-B',
            jac=True, # Returns (f, g)
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        return res.x.reshape(N, 2)

    # Initialization strategies
    # 1. Hex grid
    def create_hex_init():
        points = []
        # 7 rows: 8,7,8,7,8,7,5
        config = [8, 7, 8, 7, 8, 7, 5]
        for i, count in enumerate(config):
            y = -(len(config) - 1) / 2.0 + i
            x_start = -(count - 1) / 2.0
            if i % 2 == 1:
                x_start += 0.5
            for j in range(count):
                x = x_start + j
                points.append([x, y])
        
        points = np.array(points)
        points -= points.mean(axis=0)
        
        # Scale to fit roughly
        # Max extent is roughly 7.5 units
        scale = 0.9 / max(points.max(axis=0) - points.min(axis=0))
        points *= scale
        return points

    best_score = -1
    best_points = None
    
    # Try multiple restarts
    # 1. Hex grid
    init_hex = create_hex_init()
    res_points = run_optimization(init_hex)
    s = score(res_points)
    if s > best_score:
        best_score = s
        best_points = res_points

    # 2. Random starts (few)
    np.random.seed(42)
    for _ in range(5):
        init_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        res_points = run_optimization(init_rand)
        s = score(res_points)
        if s > best_score:
            best_score = s
            best_points = res_points
            
    # 3. Grid start
    # 7x7 grid is 49, add 1 in center? Or 5x10?
    # Let's try a simple grid
    grid_x, grid_y = np.meshgrid(np.linspace(-0.4, 0.4, 7), np.linspace(-0.4, 0.4, 8))
    # This is 56 points. We need 50.
    # Let's just use the hex init and maybe a perturbed hex.
    
    # Perturbed Hex
    init_pert = init_hex + np.random.normal(0, 0.05, size=(N, 2))
    init_pert = np.clip(init_pert, -0.45, 0.45)
    res_points = run_optimization(init_pert)
    s = score(res_points)
    if s > best_score:
        best_score = s
        best_points = res_points

    return best_points
