import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    def hexagonal_init(n):
        rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
        points = []
        for r in range(rows):
            for c in range(int(np.ceil(n / rows)) + 1):
                x = c * 1.0 + (0.5 if r % 2 else 0.0)
                y = r * np.sqrt(3) / 2
                points.append([x, y])
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        points = np.array(points[:n])
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)
        points = (points - (min_x + min_y)) * scale
        points = points - 0.45
        return points

    def objective(params, beta):
        c = params.reshape(-1, 2)
        
        # Wall distances
        d_walls = np.minimum(
            np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
            np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
        )
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Soft min approximation: log-sum-exp of -beta * d
        # Minimizing log(sum(exp(-beta d))) approximates maximizing min(d)
        
        d_flat = dists.flatten()
        d_flat = d_flat[d_flat < np.inf]
        
        if len(d_flat) == 0:
            return 0
            
        # Log-sum-exp for pairwise
        x_pair = -beta * d_flat
        x_pair_max = np.max(x_pair)
        lse_pair = x_pair_max + np.log(np.sum(np.exp(x_pair - x_pair_max)))
        
        # Walls
        x_wall = -beta * d_walls
        x_wall_max = np.max(x_wall)
        lse_wall = x_wall_max + np.log(np.sum(np.exp(x_wall - x_wall_max)))
        
        return lse_pair + lse_wall

    best_params = None
    best_score = -np.inf

    inits = []
    inits.append(hexagonal_init(N))
    
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    rng = np.random.default_rng(42)
    inits.append(rng.uniform(-0.4, 0.4, (N, 2)))
    
    rng = np.random.default_rng(123)
    inits.append(rng.uniform(-0.4, 0.4, (N, 2)))

    beta = 20.0

    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        try:
            res = minimize(objective, init.flatten(), args=(beta,), method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Evaluate the true objective (min distance) to compare
            c = res.x.reshape(-1, 2)
            d_walls = np.minimum(
                np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
                np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
            )
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_d = np.min(np.concatenate([dists.flatten(), d_walls.flatten()]))
            
            if min_d > best_score:
                best_score = min_d
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        return inits[1]

    return best_params.reshape(-1, 2)
# EVOLVE-BLOCK-END
