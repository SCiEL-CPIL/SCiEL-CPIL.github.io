import numpy as np
from scipy.optimize import minimize, basinhopping

def construct():
    N = 16
    
    def objective(x):
        pts = x.reshape(N, 2)
        dist_matrix = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
        iu = np.triu_indices(N, 1)
        pairwise_dists = dist_matrix[iu]
        
        min_d = np.min(pairwise_dists)
        max_d = np.max(pairwise_dists)
        
        if min_d < 1e-8:
            return 1e10
        return (max_d / min_d) ** 2

    def local_minimize(x0):
        result = minimize(objective, x0, method='Powell', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
        return {
            'x': result.x,
            'fun': result.fun,
            'nit': result.nit,
            'nfev': result.nfev
        }

    best_x = None
    best_f = np.inf

    # Initializers
    def init_grid():
        pts = []
        for i in range(4):
            for j in range(4):
                pts.append([i - 1.5, j - 1.5])
        return np.array(pts).flatten()

    def init_random(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(-5, 5, (N, 2)).flatten()

    inits = [init_grid(), init_random(1), init_random(2)]

    for x0 in inits:
        # Center
        pts0 = x0.reshape(N, 2)
        pts0 -= np.mean(pts0, axis=0)
        x0 = pts0.flatten()
        
        try:
            # Run Basin Hopping
            # stepsize controls the magnitude of the random kicks
            # We use a slightly larger stepsize to allow escape from grid-like traps
            res = basinhopping(
                objective, 
                x0, 
                minimizer_kwargs={'method': 'Powell', 'options': {'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6}},
                niter=20, 
                stepsize=1.0,
                T=1000.0,
                accept_test=lambda f_new, f_old, T: np.random.rand() < np.exp(-(f_new - f_old) / T) if f_new > f_old else True,
                disp=False
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            continue

    if best_x is None:
        return np.random.uniform(-5, 5, (N, 2))

    pts = best_x.reshape(N, 2)
    pts -= np.mean(pts, axis=0)
    return pts
