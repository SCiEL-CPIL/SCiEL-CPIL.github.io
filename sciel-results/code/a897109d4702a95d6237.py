import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def construct():
       # Deadline guard
       start_time = time.time()
       deadline = 110.0 # 10s margin

       # Initial guess: 4x4 grid
       # Centered at 0
       xs = np.array([0, 1, 2, 3]) - 1.5
       ys = np.array([0, 1, 2, 3]) - 1.5
       xx, yy = np.meshgrid(xs, ys)
       p0 = np.column_stack((xx.ravel(), yy.ravel()))
       
       # Normalize p0? Not strictly necessary if we minimize ratio, but helps numerical stability
       # Let's just use the grid as is.
       
       best_score = np.inf
       best_p = p0

       def objective(params):
           if time.time() > deadline:
               return 1e10
           p = params.reshape(N, D)
           # Calculate pairwise distances
           # p is (N, D)
           # diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
           # dists = np.linalg.norm(diff, axis=2)
           # This creates an N x N matrix.
           # We can use scipy.spatial.distance.pdist
           from scipy.spatial.distance import pdist
           d = pdist(p, 'euclidean')
           min_d = np.min(d)
           max_d = np.max(d)
           if min_d < 1e-9:
               return 1e10
           return (max_d / min_d) ** 2

       # Try Nelder-Mead
       try:
           res = minimize(objective, p0.ravel(), method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
           if res.fun < best_score:
               best_score = res.fun
               best_p = res.x.reshape(N, D)
       except Exception:
           pass

       # Try Powell
       if time.time() < deadline - 5:
           try:
               res = minimize(objective, p0.ravel(), method='Powell', 
                              options={'maxiter': 50, 'maxfev': 2000})
               if res.fun < best_score:
                   best_score = res.fun
                   best_p = res.x.reshape(N, D)
           except Exception:
               pass

       # If we have time, try a few random restarts?
       # Maybe not, to keep it fast.
       
       return best_p
