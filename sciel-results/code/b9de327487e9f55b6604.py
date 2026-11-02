# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist


def construct():
    N = 200
    
    def get_score(x):
        # Pairwise distances
        dists = pdist(x)
        min_pair = np.min(dists) / 2.0
        
        # Boundary distances
        boundary = np.minimum(
            np.minimum(0.5 - x[:, 0], x[:, 0] + 0.5),
            np.minimum(0.5 - x[:, 1], x[:, 1] + 0.5)
        )
        min_bound = np.min(boundary)
        
        return min(min_pair, min_bound)

    # Initialize: Jittered Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x = grid[:N].copy()
    
    # Add small jitter
    x += np.random.normal(0, 0.001, x.shape)
    x = np.clip(x, -0.5, 0.5)
    
    r_curr = get_score(x)
    
    # SA Parameters
    steps = 100000
    T_start = 0.005
    T_end = 0.00001
    move_size = 0.02
    
    cool_rate = (T_end / T_start) ** (1.0 / steps)
    
    best_x = x.copy()
    best_r = r_curr
    
    for i in range(steps):
        T = T_start * (cool_rate ** i)
        
        idx = np.random.randint(N)
        old_pos = x[idx].copy()
        
        # Gaussian move
        x[idx] += np.random.normal(0, move_size, 2)
        x[idx] = np.clip(x[idx], -0.5, 0.5)
        
        r_new = get_score(x)
        
        if r_new > r_curr:
            r_curr = r_new
            if r_new > best_r:
                best_r = r_new
                best_x = x.copy()
        else:
            delta = r_new - r_curr
            if delta != 0 and np.random.rand() < np.exp(delta / T):
                r_curr = r_new
            else:
                x[idx] = old_pos
                
    return best_x
# EVOLVE-BLOCK-END
