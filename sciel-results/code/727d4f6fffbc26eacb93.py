import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Precompute constants for area calculation
    a = (np.arange(n) + 1) / n
    M_full = 20000
    M_coarse = 500  # Coarse grid for initial optimization
    M_medium = 2000  # Medium grid for refinement
    
    y_full = (np.arange(M_full) + 0.5) / M_full
    y_coarse = (np.arange(M_coarse) + 0.5) / M_coarse
    y_medium = (np.arange(M_medium) + 0.5) / M_medium
    
    # Precompute A = a[:, None] * y[None, :] for all grids
    A_full = a[:, None] * y_full[None, :]
    A_coarse = a[:, None] * y_coarse[None, :]
    A_medium = a[:, None] * y_medium[None, :]
    w_full = (1.0 - y_full) / n
    w_coarse = (1.0 - y_coarse) / n
    w_medium = (1.0 - y_medium) / n
    
    def calc_area(x_vals, A, w):
        # L_i,m = x_i + A_i,m
        L = x_vals[:, None] + A
        L_sorted = np.sort(L, axis=0)
        d = L_sorted[1:, :] - L_sorted[:-1, :]
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        area = np.mean(union_lens)
        return area
    
    def calc_area_coarse(x_vals):
        return calc_area(x_vals, A_coarse, w_coarse)
    
    def calc_area_medium(x_vals):
        return calc_area(x_vals, A_medium, w_medium)
    
    def calc_area_full(x_vals):
        return calc_area(x_vals, A_full, w_full)
    
    start_time = time.time()
    timeout = 590  # Safety margin before 630s kill
    
    def coordinate_descent(x_init, A, w, area_func, max_sweeps, xatol, bounds_width):
        """Coordinate descent optimization"""
        x = x_init.copy()
        sweep = 0
        while sweep < max_sweeps and time.time() - start_time < timeout - 30:
            for i in range(n):
                if time.time() - start_time > timeout - 30:
                    break
                
                x_i_old = x[i]
                
                def obj_1d(xi):
                    x_new = x.copy()
                    x_new[i] = xi
                    return area_func(x_new)
                
                res = scipy.optimize.minimize_scalar(
                    obj_1d, 
                    bounds=(x_i_old - bounds_width, x_i_old + bounds_width), 
                    method='bounded', 
                    options={'xatol': xatol}
                )
                
                if res.success:
                    x[i] = res.x
            
            sweep += 1
        return x
    
    # Generate multiple initial guesses
    def keich_construction():
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        return np.array(positions, dtype=np.float64)
    
    def random_perturbation(base, sigma):
        return base + np.random.normal(0, sigma, n)
    
    # Multi-start optimization
    best_x = None
    best_area = np.inf
    
    # Start 1: Keich construction
    x0 = keich_construction()
    
    # Phase 1: Coarse grid optimization
    x_coarse = coordinate_descent(x0, A_coarse, w_coarse, calc_area_coarse, max_sweeps=3, xatol=1e-4, bounds_width=0.5)
    
    # Phase 2: Medium grid refinement
    x_medium = coordinate_descent(x_coarse, A_medium, w_medium, calc_area_medium, max_sweeps=5, xatol=1e-5, bounds_width=0.3)
    
    # Phase 3: Full grid final polish
    x_full = coordinate_descent(x_medium, A_full, w_full, calc_area_full, max_sweeps=2, xatol=1e-6, bounds_width=0.2)
    
    area_full = calc_area_full(x_full)
    best_x = x_full.copy()
    best_area = area_full
    
    # Try additional random restarts if time permits
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() - start_time > timeout - 60:
            break
        
        x0_restart = random_perturbation(x0, sigma=0.1)
        
        # Quick coarse optimization
        x_coarse_r = coordinate_descent(x0_restart, A_coarse, w_coarse, calc_area_coarse, max_sweeps=2, xatol=1e-4, bounds_width=0.5)
        
        # Medium refinement
        x_medium_r = coordinate_descent(x_coarse_r, A_medium, w_medium, calc_area_medium, max_sweeps=3, xatol=1e-5, bounds_width=0.3)
        
        # Full grid polish
        x_full_r = coordinate_descent(x_medium_r, A_full, w_full, calc_area_full, max_sweeps=1, xatol=1e-6, bounds_width=0.2)
        
        area_r = calc_area_full(x_full_r)
        
        if area_r < best_area:
            best_area = area_r
            best_x = x_full_r.copy()
    
    return best_x
