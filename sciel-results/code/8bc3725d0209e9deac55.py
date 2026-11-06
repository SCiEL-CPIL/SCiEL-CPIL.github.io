"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    def get_radius(centers):
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, 1e10)
        min_pair = np.min(dists)
        
        dx = 0.5 - np.abs(centers[:, 0])
        dy = 0.5 - np.abs(centers[:, 1])
        min_wall = np.min(np.minimum(dx, dy))
        
        r = 0.5 * np.min([min_pair, min_wall])
        return r

    def potential(x, power=8):
        centers = x.reshape(-1, 2)
        
        # Wall distances
        dx = 0.5 - np.abs(centers[:, 0])
        dy = 0.5 - np.abs(centers[:, 1])
        wall_dists = np.minimum(dx, dy)
        
        # Pairwise distances
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, 1e10)
        min_pair_dists = np.min(dists, axis=1)
        
        # Combine
        eps = 1e-6
        wall_dists_safe = np.clip(wall_dists, eps, None)
        min_pair_dists_safe = np.clip(min_pair_dists, eps, None)
        
        V = np.sum(1.0 / (wall_dists_safe ** power)) + np.sum(1.0 / (min_pair_dists_safe ** power))
        
        return V

    def simulated_annealing(x0, steps=2000, start_temp=0.05, end_temp=0.001):
        x = x0.copy()
        centers = x.reshape(-1, 2)
        current_score = get_radius(centers)
        best_x = x.copy()
        best_score = current_score
        
        rng = np.random.default_rng(12345)
        
        for i in range(steps):
            # Cooling schedule
            T = start_temp * (end_temp / start_temp) ** (i / steps)
            
            # Move
            move_idx = rng.integers(0, N)
            move = rng.normal(0, T * 0.5, 2)
            x_new = x.copy()
            x_new[move_idx * 2: move_idx * 2 + 2] += move
            
            # Clip to box to maintain validity during SA
            x_new = np.clip(x_new.reshape(-1, 2), -0.5, 0.5).flatten()
            
            new_score = get_radius(x_new.reshape(-1, 2))
            
            if new_score >= current_score:
                x = x_new
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_x = x_new.copy()
            else:
                # Accept with probability exp(-delta/T)
                delta = current_score - new_score
                if rng.random() < np.exp(-delta / (T + 1e-10)):
                    x = x_new
                    current_score = new_score
        
        return best_x

    def init_hexagonal(d, seed):
        pts = []
        range_x = int(np.ceil(1.0 / d)) + 2
        range_y = int(np.ceil(1.0 / (d * np.sqrt(3)/2))) + 2
        
        for j in range(-range_y, range_y + 1):
            for i in range(-range_x, range_x + 1):
                x = i * d + (j % 2) * d / 2
                y = j * d * np.sqrt(3) / 2
                x -= 0.5
                y -= 0.5
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append((x, y))
        
        if len(pts) < N:
            return None
            
        pts = np.array(pts)
        dists_to_center = np.sum(pts**2, axis=1)
        idx = np.argsort(dists_to_center)
        pts = pts[idx[:N]]
        
        rng = np.random.default_rng(seed)
        pts += rng.normal(0, 0.001, pts.shape)
        pts = np.clip(pts, -0.49, 0.49)
        
        return pts.flatten()

    def init_random(seed):
        rng = np.random.default_rng(seed)
        pts = rng.uniform(-0.45, 0.45, (N, 2))
        return pts.flatten()

    best_x = None
    best_score = -1.0
    
    # Try several initializations
    # 1. Hexagonal with different spacings
    spacings = [0.13, 0.14, 0.15, 0.16]
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    
    initializations = []
    for d in spacings:
        for seed in seeds[:2]: # Limit number of hex inits
            x0 = init_hexagonal(d, seed)
            if x0 is not None:
                initializations.append(x0)
                
    # 2. Random inits
    for seed in seeds:
        initializations.append(init_random(seed))
        
    # Shuffle initializations to mix types
    rng_shuff = np.random.default_rng(99)
    rng_shuff.shuffle(initializations)
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for x0 in initializations:
        if time.time() > deadline - 5: # Leave time for final polish
            break
            
        # Simulated Annealing
        try:
            x_sa = simulated_annealing(x0, steps=1000, start_temp=0.02, end_temp=0.001)
        except:
            x_sa = x0
            
        # L-BFGS-B Polish
        try:
            res = minimize(potential, x_sa, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10})
            x_opt = res.x
        except:
            x_opt = x_sa
            
        centers = x_opt.reshape(-1, 2)
        centers = np.clip(centers, -0.5, 0.5)
        
        r = get_radius(centers)
        
        if r > best_score:
            best_score = r
            best_x = centers.copy()
            
    if best_x is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_x = grid[:N]
        
    return best_x
# EVOLVE-BLOCK-END
