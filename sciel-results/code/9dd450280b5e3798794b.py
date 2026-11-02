import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.spatial.distance import cdist

N = 50

def construct():
    # Objective: maximize r = min(min_pair_dist/2, min_wall_dist)
    # We minimize -r
    
    def objective(flat):
        c = flat.reshape(N, 3)
        
        # Pairwise distances
        D = cdist(c, c)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D) / 2.0
        
        # Wall distances
        # Distance to nearest face for each sphere: 0.5 - max(|c_ik|)
        # We want the minimum of these distances over all spheres
        min_wall = np.min(0.5 - np.max(np.abs(c), axis=1))
        
        r = min(min_pair, min_wall)
        return -r

    def initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Start with a 5x5x2 grid, jittered
        xs = np.linspace(-0.4, 0.4, 5)
        ys = np.linspace(-0.4, 0.4, 5)
        zs = np.array([-0.2, 0.2])
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        grid = np.array(grid[:N])
        
        # Add random perturbation to break symmetry
        grid += rng.normal(0, 0.05, grid.shape)
        # Clip to valid range to ensure start is inside
        grid = np.clip(grid, -0.49, 0.49)
        return grid.flatten()
    
    best_r = -1e10
    best_c = None
    
    # Run basin hopping for a few seeds
    for seed in range(4):
        x0 = initial_guess(seed)
        
        # Configure local minimizer
        # Nelder-Mead is robust for non-smooth
        minimizer_kwargs = {
            'method': 'Nelder-Mead',
            'options': {'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8}
        }
        
        try:
            # Basin hopping parameters
            # stepsize: scale of the perturbation
            # niter: number of iterations
            res = basinhopping(
                objective, 
                x0, 
                minimizer_kwargs=minimizer_kwargs,
                niter=20, 
                stepsize=0.2,
                interval=5,
                disp=False
            )
            
            c = res.x.reshape(N, 3)
            
            # Calculate final score
            D = cdist(c, c)
            np.fill_diagonal(D, np.inf)
            min_pair = np.min(D) / 2.0
            min_wall = np.min(0.5 - np.max(np.abs(c), axis=1))
            r = min(min_pair, min_wall)
            
            if r > best_r:
                best_r = r
                best_c = c
        except Exception as e:
            continue

    return best_c
