# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    We exploit the separability of the tube definitions. The x-coordinate of tube (i,j)
    depends on i and base x, and the y-coordinate depends on j and base y.
    If we set base x to depend only on i (x_k = X_i) and base y only on j (y_k = Y_j),
    the union of tubes is a product of a 1D union in x and a 1D union in y.
    The volume is the integral over c in [0,1] of (length of 1D x-union at c) * (length of 1D y-union at c).
    To minimize the volume, we minimize the 1D integral independently for x and y.
    
    We solve the 1D problem: find positions z_0, ..., z_7 such that the integral
    of the length of the union of intervals [z_m + c*m/8, z_m + 1/8 + c*m/8] is minimized.
    """
    N = 8
    
    # Deadline guard
    start_time = time.time()
    deadline = 600.0
    safety_margin = 15.0
    
    def union_length_1d(z, c):
        """Calculate length of union of intervals for given c."""
        m = np.arange(N)
        starts = z + c * m / N
        ends = starts + 1.0 / N
        order = np.argsort(starts)
        s_sorted = starts[order]
        e_sorted = ends[order]
        
        total_len = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] <= cur_end:
                if e_sorted[k] > cur_end:
                    cur_end = e_sorted[k]
            else:
                total_len += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
        total_len += cur_end - cur_start
        return total_len

    def cost_1d(z):
        """
        Calculate the integral of the union length over c in [0, 1].
        z is array of shape (N-1,), representing z_1 to z_{N-1}. z_0 is fixed to 0.
        """
        full_z = np.zeros(N)
        full_z[1:] = z
        c_samples = np.linspace(0, 1, 401)
        lengths = np.array([union_length_1d(full_z, c) for c in c_samples])
        integral = np.trapz(lengths, c_samples)
        return integral

    # Keich binary construction for initial guess
    def keich_positions():
        pos = int(np.log2(N))
        j_terms = np.arange(1, pos + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        return keich

    keich_full = keich_positions()
    keich_shifted = keich_full - keich_full[0]
    
    best_z = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try multiple starting points with different optimization strategies
    for run in range(12):
        if time.time() > deadline - safety_margin:
            break
            
        if run == 0:
            start = keich_shifted[1:].copy()
        elif run == 1:
            start = np.zeros(N - 1)
        elif run == 2:
            # Try a spread-out configuration
            start = np.linspace(-0.5, 0.5, N - 1)
        elif run == 3:
            # Try a compact configuration
            start = np.linspace(-0.1, 0.1, N - 1)
        else:
            start = rng.uniform(-0.4, 0.4, N - 1)
        
        # Use Nelder-Mead
        try:
            res = scipy.optimize.minimize(cost_1d, start, method='Nelder-Mead', 
                                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
                                                    'adaptive': True})
            if res.fun < best_cost:
                best_cost = res.fun
                best_z = res.x.copy()
        except Exception:
            pass
        
        # Use L-BFGS-B
        if time.time() > deadline - safety_margin:
            break
        try:
            res2 = scipy.optimize.minimize(cost_1d, start, method='L-BFGS-B', 
                                            options={'maxiter': 1000, 'gtol': 1e-8})
            if res2.fun < best_cost:
                best_cost = res2.fun
                best_z = res2.x.copy()
        except Exception:
            pass
        
        # Use Powell's method
        if time.time() > deadline - safety_margin:
            break
        try:
            res3 = scipy.optimize.minimize(cost_1d, start, method='Powell', 
                                            options={'maxiter': 500, 'xtol': 1e-7, 'ftol': 1e-9})
            if res3.fun < best_cost:
                best_cost = res3.fun
                best_z = res3.x.copy()
        except Exception:
            pass
    
    if best_z is None:
        # Fallback to Keich construction
        best_z = keich_shifted[1:]
    
    # Construct full z array (z_0 = 0, z_1..z_7 = best_z)
    z_full = np.zeros(N)
    z_full[1:] = best_z
    
    # Since the problem is symmetric in x and y, use the same optimal 1D positions
    # for both dimensions.
    base_x = z_full
    base_y = z_full
    
    # Construct 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    bx = base_x[i_flat]
    by = base_y[j_flat]
    
    positions = np.stack([bx, by], axis=1)
    
    # Validate: check for non-finite values
    if not np.all(np.isfinite(positions)):
        # Fallback to Keich
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        bx = keich_full[i_flat]
        by = keich_full[j_flat]
        positions = np.stack([bx, by], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
