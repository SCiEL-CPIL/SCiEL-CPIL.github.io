# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    num_tubes = N * N
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

    # Pre-compute tube indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten().astype(np.float64)
    j_flat = j_grid.flatten().astype(np.float64)
    
    # Pair indices for all i < j
    i1_list, j1_list, i2_list, j2_list = [], [], [], []
    for k1 in range(num_tubes):
        for k2 in range(k1 + 1, num_tubes):
            i1_list.append(i_flat[k1])
            j1_list.append(j_flat[k1])
            i2_list.append(i_flat[k2])
            j2_list.append(j_flat[k2])
            
    i1_arr = np.array(i1_list)
    j1_arr = np.array(j1_list)
    i2_arr = np.array(i2_list)
    j2_arr = np.array(j2_list)
    
    num_pairs = len(i1_list)
    
    # Function to compute sum of pairwise overlap volumes and gradient
    # params: (64 x, 64 y)
    def objective_and_grad(params):
        x = params[:num_tubes]
        y = params[num_tubes:]
        
        # Relative offsets at c=0
        # dx0 = x2 - x1
        # dy0 = y2 - y1
        # Slopes in c:
        # dx_c = (i2 - i1) / N
        # dy_c = (j2 - j1) / N
        
        dx0 = x[i2_list] - x[i1_list]
        dy0 = y[j2_list] - y[j1_list]
        
        dx_slope = (i2_arr - i1_arr) / N
        dy_slope = (j2_arr - j1_arr) / N
        
        # We need to integrate A(c) = max(0, L - |dx0 + c*dx_slope|) * max(0, L - |dy0 + c*dy_slope|) over c in [0, 1]
        # L = 1/8
        L = 1.0 / N
        
        # To do this analytically, we find the intervals of c where the terms are non-zero.
        # Term 1: T1(c) = L - |dx0 + c*sx| > 0  => |dx0 + c*sx| < L
        # Term 2: T2(c) = L - |dy0 + c*sy| > 0  => |dy0 + c*sy| < L
        
        # This is a bit complex to handle generally with vectorization for arbitrary slopes.
        # However, slopes are multiples of 1/8.
        # Let's use a robust numerical integration with many points for the VALUE,
        # but for GRADIENT, we can use the analytical derivative of the integrand
        # integrated numerically as well, or just use a fine grid for both.
        # Given 600s, we can afford a decent number of quadrature points.
        # Let's use 100 points for c.
        
        c_vals = np.linspace(0, 1, 101)
        dc = c_vals[1] - c_vals[0]
        
        # dx(c) = dx0 + c * dx_slope
        # Shape: (num_pairs, 101)
        dx_c = dx0[:, np.newaxis] + c_vals[np.newaxis, :] * dx_slope[:, np.newaxis]
        dy_c = dy0[:, np.newaxis] + c_vals[np.newaxis, :] * dy_slope[:, np.newaxis]
        
        # Overlap lengths
        # Lx = max(0, L - |dx_c|)
        Lx = np.maximum(0, L - np.abs(dx_c))
        Ly = np.maximum(0, L - np.abs(dy_c))
        
        Area = Lx * Ly
        
        # Volume sum
        vol = np.sum(Area) * dc
        
        # Gradient calculation
        # d/dx1 (which is -dx0) of Area
        # d/dx1 Lx = d/dx1 max(0, L - |dx0 + c sx|)
        # Let u = dx0 + c sx. d/dx1 u = -1.
        # d/du max(0, L - |u|) = -sign(u) if |u| < L, else 0.
        # So d/dx1 Lx = sign(u) if |u| < L, else 0.
        
        mask_x = np.abs(dx_c) < L
        mask_y = np.abs(dy_c) < L
        
        # dArea/dx1 = (dLx/dx1) * Ly + Lx * (dLy/dx1)
        # dLx/dx1 = sign(dx_c) * mask_x
        # dLy/dx1 = 0 (y doesn't depend on x1)
        
        dLx_dx1 = np.sign(dx_c) * mask_x
        dArea_dx1 = dLx_dx1 * Ly
        
        grad_x1 = np.sum(dArea_dx1, axis=1) * dc
        
        # dArea/dx2: dx2 is in dx0. d/dx2 u = 1.
        # d/dx2 Lx = -sign(u) * mask_x
        dLx_dx2 = -np.sign(dx_c) * mask_x
        dArea_dx2 = dLx_dx2 * Ly
        grad_x2 = np.sum(dArea_dx2, axis=1) * dc
        
        # Similarly for y
        dLy_dy1 = np.sign(dy_c) * mask_y
        dArea_dy1 = dLy_dy1 * Lx
        grad_y1 = np.sum(dArea_dy1, axis=1) * dc
        
        dLy_dy2 = -np.sign(dy_c) * mask_y
        dArea_dy2 = dLy_dy2 * Lx
        grad_y2 = np.sum(dArea_dy2, axis=1) * dc
        
        # Accumulate gradients into full parameter vector
        grad = np.zeros_like(params)
        np.add.at(grad[:num_tubes], i1_list, -grad_x1) # Wait, x1 is index in x array. i1_list are tube indices.
        # Actually, x[i1_list] is x1.
        # The gradient wrt x[i1] is -grad_x1 (since dx0 = x2-x1, d/dx1 dx0 = -1)
        # My derivation: d/dx1 Lx = sign(dx_c). 
        # Let's re-verify signs.
        # Lx = L - |x2 - x1 + c(s2-s1)|.
        # d/dx1 Lx = -1 * d/d(x2-x1) (L - |...|) = -1 * (-sign(...)) = sign(...).
        # So dVol/dx1 = sum(sign * Ly).
        # So grad_x1 calculated above is dVol/dx1.
        # We need to add this to grad[x1].
        # x1 corresponds to index i1_list.
        np.add.at(grad[:num_tubes], i1_list, grad_x1)
        np.add.at(grad[:num_tubes], i2_list, grad_x2)
        
        np.add.at(grad[num_tubes:], j1_list, grad_y1)
        np.add.at(grad[num_tubes:], j2_list, grad_y2)
        
        return vol, grad

    def objective(params):
        val, _ = objective_and_grad(params)
        return val

    def gradient(params):
        _, g = objective_and_grad(params)
        return g

    # Initial guess: Keich-like or Random
    # Let's use a random initial guess but scaled to be compact
    rng_init = np.random.default_rng(42)
    # Start with a grid, then perturb
    init_x = i_flat / N
    init_y = j_flat / N
    # Add small noise to break symmetry
    init_x += rng_init.normal(0, 0.1, num_tubes)
    init_y += rng_init.normal(0, 0.1, num_tubes)
    
    x0 = np.concatenate([init_x, init_y])
    
    # Bounds: Keep positions within a reasonable box to prevent explosion
    # The box is defined by min/max. If we let them go to infinity, volume grows.
    # But L-BFGS might try to expand the box to reduce overlap?
    # No, expanding the box increases the "bounding box area" factor in the MC estimate,
    # but our objective is just sum of overlaps.
    # The task score is 1/V_union.
    # V_union is estimated by MC.
    # If we minimize sum of overlaps, we minimize V_union (approx).
    # However, if we spread them out too much, the bounding box grows,
    # and the MC estimate V = fraction * BoxArea.
    # If tubes are far apart, fraction drops, but BoxArea increases.
    # Ideally, we want to pack them tightly.
    # So we should constrain the positions to a small region, e.g., [0, 1].
    # Actually, the optimal packing likely fits in [0, 1] or slightly larger.
    # Let's bound x, y in [-1, 2].
    
    bounds = [(-1.0, 2.0)] * (2 * num_tubes)
    
    # Run optimization
    best_result = None
    best_val = np.inf
    
    # Try a few different initializations if time permits
    for seed in range(5):
        if time.time() > deadline - 60:
            break
            
        rng_init = np.random.default_rng(seed + 100)
        # Mix of grid and random
        if seed == 0:
            curr_x0 = x0
        else:
            # Random permutation of a grid?
            # Just random in [0, 1]
            rx = rng_init.uniform(0, 1, num_tubes)
            ry = rng_init.uniform(0, 1, num_tubes)
            curr_x0 = np.concatenate([rx, ry])
            
        res = minimize(
            objective, 
            curr_x0, 
            jac=gradient, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_result = res.x
            
        if time.time() > deadline - 30:
            break
            
    if best_result is None:
        # Fallback to initial
        best_result = x0
        
    # Extract positions
    best_x = best_result[:num_tubes]
    best_y = best_result[num_tubes:]
    
    # Ensure finite
    if not np.all(np.isfinite(best_x)) or not np.all(np.isfinite(best_y)):
        # Fallback to Keich
        pos_k = int(np.log2(N))
        j_terms = np.arange(1, pos_k + 1)
        keich = np.empty(N, dtype=np.float64)
        for idx in range(N):
            eps = np.array(list(str(bin(idx))[2:]), dtype=int)
            eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
            keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
        best_x = keich[i_flat.astype(int)]
        best_y = keich[j_flat.astype(int)]
        
    positions = np.stack([best_x, best_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
