import numpy as np
from scipy.optimize import minimize, basinhopping
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # Pre-compute constants
    a = np.arange(1, n + 1) / n
    m_vals = np.arange(20000)
    y = (m_vals + 0.5) / 20000.0
    w = (1.0 - y) / n
    
    def compute_area(x):
        # x: shape (n,)
        # L: shape (20000, n)
        L = x[None, :] + a[None, :] * y[:, None]
        
        # Sort L along axis 1
        L_sorted = np.sort(L, axis=1)
        
        # Compute differences
        d = np.diff(L_sorted, axis=1)
        
        # Union length
        w_col = w[:, None]
        mins = np.minimum(d, w_col)
        union_lengths = w + mins.sum(axis=1)
        
        return np.mean(union_lengths)

    # Initial guess: Keich construction
    j = np.arange(1, k + 1)
    positions_keich = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions_keich.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions_keich, dtype=np.float64)
    
    best_x = x0
    best_area = compute_area(x0)
    
    start_time = time.time()
    deadline = start_time + 500  # 500 seconds safety margin
    
    # Basin Hopping setup
    # The step size for the random kicks should be small initially
    # We use a local minimizer (L-BFGS-B) inside Basin Hopping
    
    def local_minimizer(x):
        res = minimize(
            compute_area, 
            x, 
            method='L-BFGS-B', 
            options={'maxiter': 50, 'ftol': 1e-9, 'gtol': 1e-8}
        )
        return res.x, res.fun, {'nfev': res.nfev}
    
    # Define a step size function that cools down
    # Or just use a fixed step size for the kicks
    # Since the landscape is convex-ish in L, but x is permuted, 
    # small kicks are needed to cross permutation boundaries.
    
    rng = np.random.default_rng(42)
    
    # We will run basin hopping for a limited number of iterations
    # or until time limit
    
    # Instead of scipy's basinhopping which might be slow to configure, 
    # I'll implement a simple custom basin hopping loop to control time strictly.
    
    current_x = x0.copy()
    current_area = best_area
    
    max_hops = 100
    step_size = 0.05
    
    for hop in range(max_hops):
        if time.time() > deadline:
            break
            
        # Generate a random kick
        # The kick should be in the x-space
        kick = rng.normal(0, step_size, size=n)
        x_trial = current_x + kick
        
        # Local minimize the trial
        try:
            x_min, area_min, _ = local_minimizer(x_trial)
        except Exception:
            continue
            
        if area_min < current_area:
            current_x = x_min
            current_area = area_min
            if current_area < best_area:
                best_area = current_area
                best_x = current_x.copy()
                
        # Cool down step size
        step_size *= 0.95
        
        # If we've converged tightly, break or reduce step size more
        if step_size < 0.001:
             step_size = 0.001

    if not np.all(np.isfinite(best_x)):
        return x0

    return best_x
