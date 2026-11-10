import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    np.random.seed(42)
    
    def energy_and_grad(coords, r):
        """
        Calculate energy and gradient for target radius r.
        Energy penalizes overlaps and wall violations.
        """
        coords = np.array(coords).reshape(N, 2)
        energy = 0.0
        grad = np.zeros_like(coords)
        
        # Pairwise interactions
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        dist_safe = np.where(dist > 1e-10, dist, 1.0)
        active = (dist < 2 * r) & (dist > 1e-10)
        
        term = np.where(active, (2 * r - dist), 0.0)
        coeff = -2.0 * term / dist_safe
        np.fill_diagonal(coeff, 0.0)
        
        grad_pair = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        energy_pair = 0.5 * np.sum(term**2)
        
        # Wall interactions
        wall_x = 0.5 - np.abs(coords[:, 0])
        active_x = wall_x < r
        term_x = np.where(active_x, (r - wall_x), 0.0)
        energy_x = np.sum(term_x**2)
        
        sign_x = np.sign(coords[:, 0])
        grad_x = 2.0 * term_x * sign_x
        
        wall_y = 0.5 - np.abs(coords[:, 1])
        active_y = wall_y < r
        term_y = np.where(active_y, (r - wall_y), 0.0)
        energy_y = np.sum(term_y**2)
        
        sign_y = np.sign(coords[:, 1])
        grad_y = 2.0 * term_y * sign_y
        
        grad[:, 0] += grad_x
        grad[:, 1] += grad_y
        
        total_energy = energy_pair + energy_x + energy_y
        total_grad = grad_pair + grad
        
        return total_energy, total_grad

    def try_radius(r, max_time=4.0):
        """
        Attempt to find a configuration for radius r.
        Returns (best_coords, min_energy)
        """
        best_energy = np.inf
        best_coords = None
        
        # 1 restart to keep time low
        # Initialize randomly with margin
        margin = 0.05
        coords = np.random.uniform(-0.5 + margin, 0.5 - margin, (N, 2))
        x0 = coords.flatten()
        
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        def func_flat(x):
            c = x.reshape(N, 2)
            e, g = energy_and_grad(c, r)
            return e, g.flatten()
        
        try:
            res = minimize(func_flat, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 150, 'maxfun': 1000, 'disp': False, 'gtol': 1e-6})
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_coords = res.x.reshape(N, 2)
                
        except Exception:
            pass
            
        return best_coords, best_energy

    # Binary Search for Max Radius
    lo = 0.06
    hi = 0.075
    best_r = lo
    best_centers = None
    
    start_global = time.time()
    
    # Run binary search steps
    for _ in range(12):
        if time.time() - start_global > 110:
            break
            
        mid = (lo + hi) / 2.0
        
        coords, energy = try_radius(mid)
        
        if coords is not None and energy < 1e-4:
            best_r = mid
            best_centers = coords
            lo = mid
        else:
            hi = mid
            
    if best_centers is None:
        # Fallback to random if all failed
        best_centers = np.random.uniform(-0.4, 0.4, (N, 2))
        
    return best_centers
# EVOLVE-BLOCK-END
