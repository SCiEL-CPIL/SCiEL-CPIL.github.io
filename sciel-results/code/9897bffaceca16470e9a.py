import numpy as np
import itertools
from scipy.optimize import minimize
import time

N = 16
D = 2

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # 10s safety margin
    
    def pairwise_distances_squared(points):
        # points: (N, D)
        # returns (N, N) matrix of squared distances
        # d_ij^2 = ||p_i - p_j||^2
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        return dist_sq

    def soft_max_log(dist_sq, beta):
        # log(sum(exp(beta * dist_sq))) / beta
        # Use log-sum-exp trick for stability
        d_max = np.max(dist_sq)
        # We only care about i < j, but summing all i,j is fine as it's symmetric and includes diagonals (0)
        # Actually, min distance is 0 on diagonal. We must exclude diagonals.
        # Let's mask diagonals.
        mask = ~np.eye(N, dtype=bool)
        d_masked = dist_sq[mask]
        if len(d_masked) == 0:
            return 0.0
        m = np.max(d_masked)
        return np.log(np.sum(np.exp(beta * (d_masked - m)))) + beta * m
        
    def soft_min_log(dist_sq, beta):
        # -log(sum(exp(-beta * dist_sq))) / beta
        mask = ~np.eye(N, dtype=bool)
        d_masked = dist_sq[mask]
        if len(d_masked) == 0:
            return 0.0
        m = np.max(d_masked)
        # sum(exp(-beta * d))
        # = exp(-beta * m) * sum(exp(-beta * (d - m)))
        # log(sum) = -beta * m + log(sum(exp(-beta * (d - m))))
        # -1/beta * log(sum) = m - 1/beta * log(sum(exp(-beta * (d - m))))
        # Wait, soft min is -1/beta log(sum(exp(-beta d)))
        # So we want -1/beta * [ -beta * m + log(sum(exp(-beta * (d - m)))) ]
        # = m - 1/beta * log(sum(exp(-beta * (d - m))))
        return m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_masked - m))))

    def objective_soft(x, beta):
        points = x.reshape(N, D)
        dist_sq = pairwise_distances_squared(points)
        # Avoid numerical issues if points are identical
        # Add small epsilon to min distance if needed?
        # The soft min handles it.
        log_max = soft_max_log(dist_sq, beta)
        log_min = soft_min_log(dist_sq, beta)
        # If log_min is very large (dist_min ~ 0), ratio explodes.
        # We want to minimize log_max - log_min.
        return log_max - log_min

    def objective_exact(x):
        points = x.reshape(N, D)
        dist_sq = pairwise_distances_squared(points)
        mask = ~np.eye(N, dtype=bool)
        d_masked = dist_sq[mask]
        d_min = np.min(d_masked)
        d_max = np.max(d_masked)
        if d_min < 1e-10:
            return 1e10
        return (d_max / d_min) ** 2

    def get_initialization(seed_idx):
        np.random.seed(seed_idx)
        if seed_idx == 0:
            # 4x4 Grid
            side = 4
            grid = list(itertools.product(range(side), repeat=D))[:N]
            return np.array(grid, dtype=float)
        elif seed_idx == 1:
            # Hexagonal-ish
            # 4 rows of 4
            pts = []
            for i in range(4):
                for j in range(4):
                    x = j + (0.5 if i % 2 else 0)
                    y = i * np.sqrt(3) / 2
                    pts.append([x, y])
            return np.array(pts, dtype=float)
        elif seed_idx == 2:
            # Random in square
            return np.random.rand(N, D) * 2 - 1
        elif seed_idx == 3:
            # Concentric circles
            pts = []
            # 1 center? No, distinct.
            # 4 inner, 12 outer?
            for i in range(4):
                angle = 2 * np.pi * i / 4
                pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
            for i in range(12):
                angle = 2 * np.pi * i / 12
                pts.append([1.5 * np.cos(angle), 1.5 * np.sin(angle)])
            return np.array(pts, dtype=float)
        else:
            # Random
            return np.random.rand(N, D) * 2 - 1

    best_score = np.inf
    best_points = None
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        x0 = get_initialization(i)
        
        # Run L-BFGS-B on soft objective
        # Use a moderate beta
        beta = 10.0
        try:
            res = minimize(
                objective_soft, 
                x0.flatten(), 
                args=(beta,), 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            x_opt = res.x.reshape(N, D)
            
            # Polish with Powell on exact objective
            # Powell is derivative-free and handles non-smoothness better
            # But can be slow. Limit iterations.
            res2 = minimize(
                objective_exact, 
                x_opt, 
                method='Powell', 
                options={'maxiter': 50, 'maxfev': 200}
            )
            x_final = res2.x.reshape(N, D)
            
            score = objective_exact(x_final.flatten())
            if score < best_score:
                best_score = score
                best_points = x_final
                
        except Exception as e:
            continue

    if best_points is None:
        return get_initialization(0)
        
    return best_points
# EVOLVE-BLOCK-END
